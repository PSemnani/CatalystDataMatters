"""Decomposition of the catalyst-based error by component novelty (Table S7, Figure S4).

Every held-out catalyst is classified against the training catalysts of its own split:

    T0  every metal and the support occur in at least one loaded training catalyst
        (only the combination is new)
    T1  the support occurs in no loaded training catalyst
    T2  at least one metal occurs in no loaded training catalyst

A support that appears in training only as a bare (unloaded) material does not count as
seen, since it carries no metal-support interaction. Unloaded materials (bare supports and
the blank reactor) are excluded from the tier comparison. A held-out catalyst with both an
unseen support and an unseen metal would be labelled "T1+T2" and reported separately.

Inputs are the data set and the gathered results (gather_results.py) of a catalyst-split
run of ocm_xgb.py. The training catalysts of each seed are recovered by replaying the split
of ocm_xgb.py; the replayed test catalysts are checked against the results file.

Outputs (in --out_dir):
    novelty_tiers_per_fold.csv   one row per (seed, held-out catalyst): MAE and tier
    novelty_tiers_entries.csv    one row per catalyst-tier entry: mean MAE over its folds
    novelty_tiers_summary.txt    all numbers quoted in the SI text
    table_S7.tex                 body of Table S7
    figure_S4.pdf / .png         Figure S4

Example (run from the repository root):

    python src/novelty_tiers.py \
        --data_path Dataset/OCM-NguyenEtAl_with_descriptors_updated.csv \
        --results experiments/resubmission/split_catalyst/xgb_cv_random_50/gathered_results.csv \
        --random_results experiments/resubmission/split_random/xgb_cv_random_50/gathered_results.csv \
        --out_dir experiments/resubmission/novelty_tiers
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from utils import split_data

NAME_COL = "Name"
TARGET_COL = "C2y"
SUPPORT_COL = "Support"
METAL_COLS = ["M1", "M2", "M3"]  # element symbols; "<site>_mol%" gives the loading
REFERENCE_SUPPORT = "SiO2"
REFERENCE_METAL = "Mn"

TIERS = ["T0", "T1", "T2"]
TIER_COLORS = {"T0": "#2a7f6f", "T1": "#cc7a2e", "T2": "#8e5aa8"}
TIER_HEADERS = {"T0": "T0: new combination", "T1": "T1: unseen support", "T2": "T2: unseen metal"}
TIER_YLABELS = {
    "T0": "T0  new combination\nof known components",
    "T1": "T1  unseen support",
    "T2": "T2  unseen metal",
}


# --------------------------------------------------------------------------------------
# composition and classification
# --------------------------------------------------------------------------------------
def catalyst_compositions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per catalyst: set of metals, support and whether it carries any metal."""
    rows = []
    for name, grp in df.groupby(NAME_COL, sort=False):
        metals = set()
        for site in METAL_COLS:
            present = grp[f"{site}_mol%"] > 0
            metals.update(grp.loc[present, site].astype(str).unique())
        supports = grp[SUPPORT_COL].unique()
        if len(supports) != 1:
            raise ValueError(f"Catalyst {name} has more than one support: {supports}")
        rows.append(
            {NAME_COL: name, "metals": frozenset(metals), "support": supports[0], "loaded": bool(metals)}
        )
    return pd.DataFrame(rows).set_index(NAME_COL)


def classify(test_name: str, train_names: list[str], comp: pd.DataFrame) -> str:
    train = comp.loc[train_names]
    train = train[train["loaded"]]
    seen_metals = set().union(*train["metals"])
    seen_supports = set(train["support"])
    unseen_support = comp.at[test_name, "support"] not in seen_supports
    unseen_metal = not comp.at[test_name, "metals"] <= seen_metals
    if unseen_support and unseen_metal:
        return "T1+T2"
    return "T1" if unseen_support else "T2" if unseen_metal else "T0"


# --------------------------------------------------------------------------------------
# splits and per-fold errors
# --------------------------------------------------------------------------------------
def replay_split(df: pd.DataFrame, seed: int, n_train: int, n_test: int) -> tuple[list, list]:
    """Reproduce the catalyst split of ocm_xgb.py (no conditions) for one seed."""
    rng = random.Random(seed)
    rng.randint(0, 1000000)  # drawn for the XGBoost seed in ocm_xgb.py before splitting
    out = split_data(
        df, [TARGET_COL], TARGET_COL, "catalyst", rng,
        n_train_catalysts=n_train, n_val_catalysts=0, n_test_catalysts=n_test,
        return_indices=True, conditions={},
    )
    names = df[NAME_COL].values
    return list(pd.unique(names[out[6]])), list(pd.unique(names[out[8]]))


def filter_results(df: pd.DataFrame, args) -> pd.DataFrame:
    mask = (
        (df["model_type"] == args.model_type)
        & (df["feature_set"] == args.feature_set)
        & (df["augmentation"].astype(str).str.lower() == args.augmentation.lower())
        & (df["n_train_catalysts"] == args.n_train_catalysts)
    )
    if "condition" in df:
        mask &= df["condition"].isna() | df["condition"].astype(str).isin(["", "None"])
    out = df[mask]
    if out.empty:
        raise ValueError("No results left after filtering; check the filter arguments.")
    if out["seed"].duplicated().any():
        raise ValueError("Duplicate seeds after filtering; run gather_results.py first.")
    return out


def per_fold_errors(results: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (seed, held-out catalyst) with its MAE."""
    idx = sorted(int(c.rsplit("_", 1)[1]) for c in results.columns if c.startswith("test_catalyst_"))
    rows = [
        {"seed": int(r["seed"]), NAME_COL: r[f"test_catalyst_{i}"], "mae": float(r[f"mae_test_catalyst_{i}"])}
        for _, r in results.iterrows()
        for i in idx
        if pd.notna(r[f"test_catalyst_{i}"])
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# table and figure
# --------------------------------------------------------------------------------------
def latex_name(name: str) -> str:
    """Mn-Na2WO4/SiO2 -> Mn$-$Na$_2$WO$_4$/SiO$_2$, also MoO$_x$, SiC$_\\mathrm{nf}$, ZSM-5."""
    if name == "ZSM-5" or name.endswith("/ZSM-5"):
        return latex_name(name[: -len("ZSM-5")]) + "ZSM-5" if name != "ZSM-5" else "ZSM-5"
    s = re.sub(r"(\d+)", r"$_{\1}$", name)
    s = s.replace("Ox", "O$_x$").replace("SiCnf", r"SiC$_\mathrm{nf}$")
    return s.replace("-", "$-$")


def write_latex_table(entries: pd.DataFrame, path: Path) -> None:
    twins = set(entries.loc[entries["in_several_tiers"], NAME_COL])
    cols = {t: entries[entries["tier"] == t].sort_values("mae") for t in TIERS}
    n_rows = max(len(c) for c in cols.values())
    lines = [
        r"\begin{tabular}{lr@{\hspace{2em}}lr@{\hspace{2em}}lr}",
        r"\toprule",
        " & ".join(rf"\multicolumn{{2}}{{c}}{{{TIER_HEADERS[t]} ({len(cols[t])})}}" for t in TIERS) + r" \\",
        r"\cmidrule(r){1-2}\cmidrule(lr){3-4}\cmidrule(l){5-6}",
        r"Catalyst & MAE & Catalyst & MAE & Catalyst & MAE \\",
        r"\midrule",
    ]
    for i in range(n_rows):
        cells = []
        for t in TIERS:
            if i < len(cols[t]):
                r = cols[t].iloc[i]
                star = "$^*$" if r[NAME_COL] in twins else ""
                cells += [latex_name(r[NAME_COL]) + star, f"{r['mae']:.2f}"]
            else:
                cells += ["", ""]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        " & ".join(f"mean & {cols[t]['mae'].mean():.2f}" for t in TIERS) + r" \\",
        " & ".join(f"range & {cols[t]['mae'].min():.2f}--{cols[t]['mae'].max():.2f}" for t in TIERS) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_figure(entries: pd.DataFrame, random_mae: float, annotate: list[str], path: Path, seed: int = 0) -> None:
    """One point per catalyst-tier entry, tiers as rows (T0 bottom), bar and label at each
    tier mean, dashed line at the random-split MAE, selected catalysts labelled."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    positions = {}
    for row, t in enumerate(TIERS):
        sel = entries[entries["tier"] == t]
        y = row + rng.uniform(-0.15, 0.15, len(sel))
        positions.update({(n, t): (m, yy) for n, m, yy in zip(sel[NAME_COL], sel["mae"], y)})
        ax.scatter(sel["mae"], y, s=50, color=TIER_COLORS[t], alpha=0.75,
                   edgecolor="white", linewidth=0.8, zorder=3)
        mean = sel["mae"].mean()
        ax.plot([mean, mean], [row - 0.32, row + 0.32], color=TIER_COLORS[t], lw=3, zorder=4)
        ax.text(mean, row + 0.38, f"{mean:.2f}", color=TIER_COLORS[t], ha="center",
                va="bottom", fontsize=12, fontweight="bold")
    ax.axvline(random_mae, ls="--", color="0.3", lw=1.5, zorder=1)
    ax.text(random_mae + 0.05, -0.55, f"random split\n{random_mae:.2f}",
            color="0.3", ha="left", va="bottom", fontsize=10.5)
    for name in annotate:
        hits = [(k, v) for k, v in positions.items() if k[0] == name]
        if not hits:
            print(f"WARNING: catalyst to annotate not found among entries: {name}")
        for (_, _t), (m, yy) in hits:
            label = re.sub(r"(\d+)", r"$_{\1}$", name)
            ax.annotate(f"{label}\n{m:.2f}", xy=(m, yy), xytext=(m + 0.3, yy + 0.42),
                        fontsize=10, ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
    ax.set_yticks(range(len(TIERS)))
    ax.set_yticklabels([TIER_YLABELS[t] for t in TIERS], fontsize=11)
    ax.set_ylim(-0.6, len(TIERS) - 0.15)
    lo = min(entries["mae"].min(), random_mae)
    ax.set_xlim(max(0.0, np.floor(lo * 2) / 2 - 0.1), np.ceil(entries["mae"].max() * 2) / 2 + 0.3)
    ax.set_xlabel("MAE (C$_2$ yield), one point per catalyst and tier", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True)
    p.add_argument("--results", required=True, help="gathered_results.csv of the catalyst-split run")
    p.add_argument("--random_results", required=True, help="gathered_results.csv of the random-split run")
    p.add_argument("--model_type", default="xgboost")
    p.add_argument("--feature_set", default="base+descriptors")
    p.add_argument("--augmentation", default="yes")
    p.add_argument("--n_train_catalysts", type=int, default=49)
    p.add_argument("--annotate", nargs="*", default=None,
                   help="catalysts to label in Figure S4 (default: best T2 and worst T0 entry)")
    p.add_argument("--out_dir", default="experiments/resubmission/novelty_tiers")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    df = pd.read_csv(args.data_path)
    comp = catalyst_compositions(df)
    results = filter_results(pd.read_csv(args.results), args)
    random_results = filter_results(pd.read_csv(args.random_results), args)
    random_mae = float(random_results["mae"].mean())
    n_test = int(results["n_test_catalysts"].iloc[0])

    # ---- classify every held-out catalyst against its own training split ------------
    folds = per_fold_errors(results)
    tiers = []
    for seed, grp in folds.groupby("seed"):
        train, test = replay_split(df, seed, args.n_train_catalysts, n_test)
        if set(test) != set(grp[NAME_COL]):
            raise RuntimeError(f"Seed {seed}: replayed test catalysts differ from the results file. "
                               "Use the exact data file of the run.")
        tiers += [
            {"seed": seed, NAME_COL: n, "tier": classify(n, train, comp) if comp.at[n, "loaded"] else "unloaded"}
            for n in grp[NAME_COL]
        ]
    folds = folds.merge(pd.DataFrame(tiers), on=["seed", NAME_COL])
    folds.to_csv(out_dir / "novelty_tiers_per_fold.csv", index=False)

    entries = (
        folds[folds["tier"].isin(TIERS)]
        .groupby([NAME_COL, "tier"])
        .agg(mae=("mae", "mean"), mae_std=("mae", "std"), n_folds=("mae", "size"))
        .reset_index()
    )
    entries["in_several_tiers"] = entries[NAME_COL].duplicated(keep=False)
    entries = entries.sort_values(["tier", "mae"]).reset_index(drop=True)
    entries.to_csv(out_dir / "novelty_tiers_entries.csv", index=False)

    # ---- summary ----------------------------------------------------------------------
    log(f"Setting: {args.model_type}, {args.feature_set}, augmentation={args.augmentation}, "
        f"{args.n_train_catalysts} training / {n_test} test catalysts, {len(results)} seeds")
    log(f"Random-split MAE: {random_mae:.2f} ({len(random_results)} seeds)")
    n_outside = int((folds["tier"] == "T1+T2").sum())
    log(f"Held-out entries with both an unseen support and an unseen metal: {n_outside}")

    log(f"\n{entries[NAME_COL].nunique()} loaded catalysts -> {len(entries)} catalyst-tier entries")
    for t in TIERS:
        e = entries.loc[entries["tier"] == t, "mae"]
        log(f"  {t}: n={len(e):2d}  mean={e.mean():.2f}  range={e.min():.2f}-{e.max():.2f}")
    twins = entries[entries["in_several_tiers"]].sort_values([NAME_COL, "tier"])
    log("Catalysts in more than one tier (MAE, folds):")
    for name, g in twins.groupby(NAME_COL):
        log("  " + name + ": " + ", ".join(f"{r.tier} {r.mae:.2f} ({r.n_folds})" for r in g.itertuples()))
    thin = entries[entries["n_folds"] < 3]
    if len(thin):
        log("Entries based on fewer than 3 folds: " +
            ", ".join(f"{r.Name} {r.tier} ({r.n_folds})" for r in thin.itertuples()))

    log("\nTier comparisons on catalyst-tier entries:")
    for a, b in [("T0", "T2"), ("T0", "T1"), ("T1", "T2")]:
        x, y = (entries.loc[entries["tier"] == t, "mae"] for t in (a, b))
        log(f"  {a} vs {b}: Welch p = {ttest_ind(x, y, equal_var=False).pvalue:.2f}, "
            f"Mann-Whitney p = {mannwhitneyu(x, y, alternative='two-sided').pvalue:.2f}")

    per_cat = folds.groupby(NAME_COL)["mae"].mean().to_frame().join(comp)
    loaded = per_cat["loaded"]
    with_ref_metal = per_cat["metals"].apply(lambda m: REFERENCE_METAL in m)
    on_ref_support = per_cat["support"] == REFERENCE_SUPPORT
    log("\nPer-catalyst MAE (mean over its folds), averaged over catalysts:")
    for label, mask in [
        ("unloaded materials", ~loaded),
        ("loaded catalysts", loaded),
        (f"loaded, with {REFERENCE_METAL}", loaded & with_ref_metal),
        (f"loaded, without {REFERENCE_METAL}", loaded & ~with_ref_metal),
        (f"loaded, on {REFERENCE_SUPPORT}", loaded & on_ref_support),
        (f"loaded, other supports", loaded & ~on_ref_support),
    ]:
        log(f"  {label:<26} n={int(mask.sum()):2d}  MAE={per_cat.loc[mask, 'mae'].mean():.2f}")

    metal_counts = pd.Series([m for ms in comp["metals"] for m in ms]).value_counts()
    support_counts = comp.loc[loaded, "support"].value_counts()
    log("\nComponent frequencies (number of catalysts):")
    log("  metals: " + ", ".join(f"{m} {n}" for m, n in metal_counts.items()))
    log(f"  metals in exactly one catalyst: {int((metal_counts == 1).sum())}")
    log(f"  supports, all catalysts: {comp['support'].value_counts().to_dict()}")
    log(f"  supports, loaded catalysts: {support_counts.to_dict()}")

    single = set(metal_counts[metal_counts == 1].index)
    unique_central = entries[(entries["tier"] == "T2") & entries[NAME_COL].map(
        lambda n: bool(comp.at[n, "metals"] & single))]
    log("\nT2 entries whose unseen metal occurs in no other catalyst, by MAE:")
    log("  " + ", ".join(f"{r.Name} {r.mae:.2f}" for r in unique_central.itertuples()))

    # ---- table and figure ------------------------------------------------------------
    if args.annotate is None:
        best_t2 = entries[entries["tier"] == "T2"].nsmallest(1, "mae")[NAME_COL].iloc[0]
        worst_t0 = entries[entries["tier"] == "T0"].nlargest(1, "mae")[NAME_COL].iloc[0]
        args.annotate = [best_t2, worst_t0]
    write_latex_table(entries, out_dir / "table_S7.tex")
    plot_figure(entries, random_mae, args.annotate, out_dir / "figure_S4")
    (out_dir / "novelty_tiers_summary.txt").write_text("\n".join(log_lines) + "\n")
    print(f"\nWrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
