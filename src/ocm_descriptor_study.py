#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCM — 3 Arms (Only Atom # | Only Descriptors | Both) with Multi Test Sets
==========================================================================
Trains XGBoost with version-agnostic early stopping (xgb.train) across 5 fixed
test sets (5 catalysts each). For each arm and each (size, repeat), logs MAE,
SHAP (|mean SHAP|), and XGBoost "gain" importances. Aggregates results across
test sets and plots averaged learning curves and top-K importances vs size.

Arms:
  A) Only atom numbers (Base):       [M1_atom_number, M2_atom_number, M3_atom_number]
  B) Only descriptors (+ support SA): 15 descriptor columns + support_surface_area
  C) Both: atom numbers + 15 descriptors + support_surface_area
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import List, Sequence, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # OneHot kept for completeness

# ───────── config ─────────
CSV_PATH      = "/Users/parastoo/DataMatters/Dataset/OCM-NguyenEtAl_with_descriptors.csv"
TARGET_COL    = "C2y"
RANDOM_SEED   = 1

# NOTE: includes tiny sizes 1 and 3 as requested
TRAIN_SIZES   = sorted(set([1, 3] + list(range(5, 55, 5)) + [54]))
N_REPEATS     = 20

# Multiple fixed test sets
TESTSET_SIZE  = 10
N_TESTSETS    = 10
DISJOINT      = True  # try to make them disjoint if enough catalysts

# Optional hyperparam tuning (kept off)
N_TRIALS      = 0
CV_FOLDS_MAX  = 5

TOP_K_FEATS   = 7

# ───────── descriptors ─────────
DESCRIPTOR_COLUMNS = [
    "M1_electronegativity","M1_inverse_ionization","M1_d_electron_count","M1_ionic_radius","M1_oxidation_state",
    "M2_electronegativity","M2_inverse_ionization","M2_d_electron_count","M2_ionic_radius","M2_oxidation_state",
    "M3_electronegativity","M3_inverse_ionization","M3_d_electron_count","M3_ionic_radius","M3_oxidation_state"
]

# ───────── support SA LUT (for arm B and C) ─────────
SUPPORT_SA_LUT = {
    "Al2O3":413,"BEA":512,"BN":1513,"CeO2":102,"MgO":194,
    "SiC":22,"SiCnf":124,"SiO2":350,"TiO2":417,"ZrO2":37,
    "ZSM-5":285,"NbO5":190,"n.a.":0,
}

# ───────── feature sets for the 3 arms ─────────
# We isolate variables to study their effects.
BASE_PROCESS = [
    "Temp","CT","CH4/O2","Ar_flow","O2_flow","CH4_flow","Total_flow",
    "M1_mol%","M2_mol%","M3_mol%"
]

# ───────── feature sets for the 3 arms (all include BASE_PROCESS) ─────────
# appended BASE_PROCESS to each arm definition
ARM_A_ONLY_ATOM = BASE_PROCESS + ["M1_atom_number","M2_atom_number","M3_atom_number"] + ["Support_ID"]
ARM_B_ONLY_DESC = BASE_PROCESS + DESCRIPTOR_COLUMNS + ["support_surface_area"]
ARM_C_BOTH      = BASE_PROCESS + ["M1_atom_number","M2_atom_number","M3_atom_number"] + DESCRIPTOR_COLUMNS + ["support_surface_area"]

# ───────── helpers ─────────
def ensure_support_surface_area(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure 'support_surface_area' exists (compute from 'Support' if needed)."""
    if "support_surface_area" in df.columns:
        df["support_surface_area"] = pd.to_numeric(df["support_surface_area"], errors="coerce").fillna(0.0)
        return df
    if "Support" in df.columns:
        df = df.copy()
        df["support_surface_area"] = df["Support"].map(lambda s: SUPPORT_SA_LUT.get(str(s).strip(), 0)).astype(float)
        print("[info] Created 'support_surface_area' from Support via LUT.")
        return df
    df = df.copy()
    df["support_surface_area"] = 0.0
    print("[warn] Neither 'support_surface_area' nor 'Support' present; filled with zeros.")
    return df

def validate_features(df: pd.DataFrame, features: List[str]) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns:\n  - " + "\n  - ".join(missing))

def build_preprocessor(cols: Sequence[str], df_: pd.DataFrame) -> ColumnTransformer:
    """
    Scales numeric columns; if any categorical present in other experiments,
    we keep OneHotEncoder ready (not used in current 3 arms).
    """
    num = [c for c in cols if df_[c].dtype != "object"]
    cat = [c for c in cols if df_[c].dtype == "object"]
    tr = []
    if num: tr.append(("num", StandardScaler(), num))
    if cat: tr.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(tr, verbose_feature_names_out=False)

def default_xgb_params(seed: int) -> Dict:
    """Parameters for xgb.train (version-agnostic)."""
    return dict(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        eta=0.05,       # lr
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        alpha=0.0,
        lambda_=1.0,           
        seed=seed,
        random_state=seed,
    )

def make_disjoint_test_sets(names: List[str], test_n: int, k_sets: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    pool = names.copy(); rng.shuffle(pool)
    sets = []
    for _ in range(k_sets):
        if len(pool) < test_n: break
        sets.append(pool[:test_n]); pool = pool[test_n:]
    return sets

def make_seeded_test_sets(names: List[str], test_n: int, k_sets: int, seed: int) -> List[List[str]]:
    sets = []
    for i in range(k_sets):
        rng = random.Random(seed + i); shuf = names.copy(); rng.shuffle(shuf)
        sets.append(shuf[:test_n])
    return sets

def define_multiple_tests(df: pd.DataFrame, test_n: int, k_sets: int, seed: int, disjoint: bool) -> List[List[str]]:
    names = df["Name"].dropna().astype(str).unique().tolist()
    if len(names) < test_n:
        raise ValueError(f"Dataset must have ≥{test_n} unique catalysts.")
    if disjoint:
        sets = make_disjoint_test_sets(names, test_n, k_sets, seed)
        if len(sets) < k_sets:
            sets += make_seeded_test_sets(names, test_n, k_sets - len(sets), seed + 10_000)
    else:
        sets = make_seeded_test_sets(names, test_n, k_sets, seed)
    return sets

def get_feature_names(prep: ColumnTransformer) -> List[str]:
    try:
        return list(prep.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in prep.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    out = list(trans.get_feature_names_out(cols))
                except Exception:
                    out = list(cols)
            else:
                out = list(cols if isinstance(cols, (list, tuple)) else [cols])
            names.extend(out)
        return names

def shap_importance_from_booster(booster: xgb.Booster, X_te: np.ndarray, feat_names: List[str]) -> pd.Series:
    X_df = pd.DataFrame(X_te, columns=feat_names)
    expl = shap.TreeExplainer(booster)
    sv = expl(X_df, check_additivity=False)
    values = sv.values if hasattr(sv, "values") else sv
    mean_abs = np.abs(values).mean(axis=0)
    return pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)

def gain_importance_from_booster(booster: xgb.Booster, feat_names: List[str]) -> pd.Series:
    imp_dict = booster.get_score(importance_type="gain")
    values = [imp_dict.get(name, 0.0) for name in feat_names]
    return pd.Series(values, index=feat_names).sort_values(ascending=False)

def get_early_stopping_split(train_df: pd.DataFrame, feats: List[str], seed: int = 12345):
    groups = train_df["Name"].values
    uniq = np.unique(groups)
    if len(uniq) < 2:
        n = len(train_df); cut = max(1, int(0.8 * n))
        idx = np.arange(n); rng = np.random.default_rng(seed); rng.shuffle(idx)
        return idx[:cut], idx[cut:]
    gkf = GroupKFold(n_splits=min(5, len(uniq)))
    tr_idx, va_idx = next(iter(gkf.split(train_df[feats], train_df[TARGET_COL], groups)))
    return tr_idx, va_idx

def train_with_es(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    params: Dict,
    feature_names: List[str],
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50
) -> xgb.Booster:
    if "lambda_" in params and "lambda" not in params:
        params = params.copy(); params["lambda"] = params.pop("lambda_")
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dvalid = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster

# ───────── single (size, repeat) fit for a given test set ─────────
def train_one_size(
    df: pd.DataFrame, feats: List[str],
    train_cats: List[str], test_cats: List[str],
    rep_seed: int
) -> Tuple[float, pd.Series, pd.Series]:
    train_df = df[df["Name"].isin(train_cats)].reset_index(drop=True)
    test_df  = df[df["Name"].isin(test_cats)].reset_index(drop=True)

    tr_idx, va_idx = get_early_stopping_split(train_df, feats, seed=rep_seed + 12345)
    X_tr, y_tr = train_df.iloc[tr_idx][feats], train_df.iloc[tr_idx][TARGET_COL]
    X_va, y_va = train_df.iloc[va_idx][feats], train_df.iloc[va_idx][TARGET_COL]

    prep = build_preprocessor(feats, train_df)
    X_tr_p = prep.fit_transform(X_tr)
    X_va_p = prep.transform(X_va)
    X_te_p = prep.transform(test_df[feats])
    feat_names = get_feature_names(prep)

    params = default_xgb_params(rep_seed)

    # (Optional) light tuning — disabled by default
    if N_TRIALS and N_TRIALS > 0:
        groups = train_df["Name"].values
        uniq = len(np.unique(groups))
        cv = GroupKFold(n_splits=min(CV_FOLDS_MAX, max(2, uniq))) if uniq >= 3 else LeaveOneOut()

        def objective(trial):
            trial_params = dict(
                **params,
                eta=trial.suggest_float("lr", 0.03, 0.12, log=True),
                max_depth=trial.suggest_categorical("max_depth", [4, 6, 8]),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight=trial.suggest_float("min_child_weight", 0.5, 3.0, log=True),
                alpha=trial.suggest_float("reg_alpha", 0.0, 0.5),
                lambda_=trial.suggest_float("reg_lambda", 0.5, 2.5),
            )
            fold_mae = []
            for tr, va in cv.split(train_df[feats], train_df[TARGET_COL], groups):
                Xtr, ytr = train_df.iloc[tr][feats], train_df.iloc[tr][TARGET_COL]
                Xva, yva = train_df.iloc[va][feats], train_df.iloc[va][TARGET_COL]
                prep_cv = build_preprocessor(feats, train_df)
                Xtr_p = prep_cv.fit_transform(Xtr)
                Xva_p = prep_cv.transform(Xva)
                names_cv = get_feature_names(prep_cv)
                booster = train_with_es(
                    Xtr_p, np.asarray(ytr), Xva_p, np.asarray(yva),
                    trial_params, feature_names=names_cv,
                    num_boost_round=2200, early_stopping_rounds=50
                )
                dval = xgb.DMatrix(Xva_p, feature_names=names_cv)
                pred = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
                fold_mae.append(mean_absolute_error(yva, pred))
            return float(np.mean(fold_mae))

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=rep_seed))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best = study.best_params
        params.update(
            eta=best["lr"], max_depth=int(best["max_depth"]),
            subsample=best["subsample"], colsample_bytree=best["colsample_bytree"],
            min_child_weight=best["min_child_weight"], alpha=best["reg_alpha"], lambda_=best["reg_lambda"],
        )

    booster = train_with_es(
        X_tr_p, np.asarray(y_tr), X_va_p, np.asarray(y_va),
        params, feature_names=feat_names, num_boost_round=2000, early_stopping_rounds=50
    )

    dtest = xgb.DMatrix(X_te_p, feature_names=feat_names)
    te_pred = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    mae = mean_absolute_error(test_df[TARGET_COL], te_pred)

    shap_imp = shap_importance_from_booster(booster, X_te_p, feat_names)
    gain_imp = gain_importance_from_booster(booster, feat_names)

    return float(mae), shap_imp, gain_imp

# ───────── run a full arm across K test sets and aggregate ─────────
def run_experiment_multi(
    df: pd.DataFrame, feature_list: List[str], out_root: Path,
    random_seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, Dict[int,pd.Series], Dict[int,pd.Series], Dict[int,pd.Series], Dict[int,pd.Series]]:
    """
    Returns (averaged across test sets):
        mae_df_agg (size, mean, std),
        shap_mean_by_size, shap_std_by_size,
        gain_mean_by_size, gain_std_by_size
    Writes per-test-set results under out_root/testset_i and aggregated CSVs under out_root/aggregated_over_testsets.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root/"aggregated_over_testsets").mkdir(parents=True, exist_ok=True)
    validate_features(df, feature_list + ["Name"])

    test_sets = define_multiple_tests(df, TESTSET_SIZE, N_TESTSETS, RANDOM_SEED, DISJOINT)
    pd.DataFrame(
        [{"testset_id": tid, "Name": n} for tid, names in enumerate(test_sets) for n in names]
    ).to_csv(out_root/"fixed_test_sets.csv", index=False)
    print(f"[{out_root.name}] Created {len(test_sets)} fixed test sets (size={TESTSET_SIZE}, disjoint={DISJOINT}).")

    rng_master = random.Random(random_seed)
    per_ts_mae = []
    per_ts_shap: Dict[int, Dict[int, List[pd.Series]]] = {}
    per_ts_gain: Dict[int, Dict[int, List[pd.Series]]] = {}

    all_names = df["Name"].astype(str).unique().tolist()

    for tid, test_names in enumerate(test_sets):
        ts_dir = out_root / f"testset_{tid}"
        ts_dir.mkdir(parents=True, exist_ok=True)

        pool_names = [n for n in all_names if n not in test_names]
        sizes = sorted(s for s in TRAIN_SIZES if s <= len(pool_names))
        print(f"[{out_root.name}] Testset {tid}: using train sizes {sizes} (pool {len(pool_names)}).")

        per_ts_shap[tid] = {}
        per_ts_gain[tid] = {}

        for size in sizes:
            per_ts_shap[tid][size] = []
            per_ts_gain[tid][size] = []
            for rep in range(N_REPEATS):
                rng = random.Random(rng_master.randint(0, 1_000_000))
                train_cats = rng.sample(pool_names, size)
                mae, shap_imp, gain_imp = train_one_size(
                    df, feature_list, train_cats, test_names,
                    rep_seed=rng.randint(0, 1_000_000)
                )
                per_ts_mae.append({"testset_id": tid, "size": size, "repeat": rep, "mae": mae})
                per_ts_shap[tid][size].append(shap_imp)
                per_ts_gain[tid][size].append(gain_imp)

                (ts_dir/f"shap_size{size}_rep{rep+1}.csv").write_text(shap_imp.to_csv(header=False))
                (ts_dir/f"gain_size{size}_rep{rep+1}.csv").write_text(gain_imp.to_csv(header=False))
                print(f"[{out_root.name} | TS{tid}] size={size:2d} rep={rep+1}  "
                      f"MAE={mae:.4f}; SHAP top1={shap_imp.index[0]} ({shap_imp.iloc[0]:.4f}); "
                      f"Gain top1={gain_imp.index[0]} ({gain_imp.iloc[0]:.4f})")

        # Save per-testset MAEs
        pd.DataFrame([r for r in per_ts_mae if r["testset_id"] == tid]).to_csv(ts_dir/"learning_curve_results.csv", index=False)

        # Save per-testset mean/std importances
        for size, series_list in per_ts_shap[tid].items():
            if series_list:
                M = pd.concat(series_list, axis=1).fillna(0.0)
                M.mean(axis=1).to_csv(ts_dir/f"shap_mean_size{size}.csv")
                M.std(axis=1).to_csv(ts_dir/f"shap_std_size{size}.csv")
        for size, series_list in per_ts_gain[tid].items():
            if series_list:
                G = pd.concat(series_list, axis=1).fillna(0.0)
                G.mean(axis=1).to_csv(ts_dir/f"gain_mean_size{size}.csv")
                G.std(axis=1).to_csv(ts_dir/f"gain_std_size{size}.csv")

    # Aggregate MAE over test sets (first avg over repeats within each set)
    mae_df_all = pd.DataFrame(per_ts_mae)
    mae_df_all.to_csv(out_root/"learning_curve_results_all_testsets.csv", index=False)
    mae_size_ts_mean = mae_df_all.groupby(["testset_id","size"])["mae"].mean().reset_index()
    agg_over_ts = mae_size_ts_mean.groupby("size")["mae"].agg(["mean","std"]).reset_index()
    agg_over_ts.to_csv(out_root/"aggregated_over_testsets"/"aggregated_mae.csv", index=False)

    # Aggregate SHAP & Gain across test sets
    shap_mean_by_size: Dict[int, pd.Series] = {}
    shap_std_by_size:  Dict[int, pd.Series] = {}
    gain_mean_by_size: Dict[int, pd.Series] = {}
    gain_std_by_size:  Dict[int, pd.Series] = {}

    sizes_union = sorted(set(mae_df_all["size"].unique().tolist()))

    for size in sizes_union:
        per_ts_series_shap = []
        per_ts_series_gain = []
        for tid in range(len(test_sets)):
            if size in per_ts_shap[tid] and per_ts_shap[tid][size]:
                S = pd.concat(per_ts_shap[tid][size], axis=1).fillna(0.0).mean(axis=1)  # avg across repeats in this testset
                per_ts_series_shap.append(S)
            if size in per_ts_gain[tid] and per_ts_gain[tid][size]:
                G = pd.concat(per_ts_gain[tid][size], axis=1).fillna(0.0).mean(axis=1)
                per_ts_series_gain.append(G)

        if per_ts_series_shap:
            M = pd.concat(per_ts_series_shap, axis=1).fillna(0.0)  # columns=testsets
            shap_mean_by_size[size] = M.mean(axis=1)
            shap_std_by_size[size]  = M.std(axis=1)
            shap_mean_by_size[size].to_csv(out_root/"aggregated_over_testsets"/f"shap_mean_size{size}.csv")
            shap_std_by_size[size].to_csv(out_root/"aggregated_over_testsets"/f"shap_std_size{size}.csv")

        if per_ts_series_gain:
            Gm = pd.concat(per_ts_series_gain, axis=1).fillna(0.0)
            gain_mean_by_size[size] = Gm.mean(axis=1)
            gain_std_by_size[size]  = Gm.std(axis=1)
            gain_mean_by_size[size].to_csv(out_root/"aggregated_over_testsets"/f"gain_mean_size{size}.csv")
            gain_std_by_size[size].to_csv(out_root/"aggregated_over_testsets"/f"gain_std_size{size}.csv")

    return agg_over_ts, shap_mean_by_size, shap_std_by_size, gain_mean_by_size, gain_std_by_size

# ───────── orchestrator for 3 arms ─────────
def main():
    raw = pd.read_csv(CSV_PATH)
    if TARGET_COL not in raw.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    df = raw.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Ensure support SA (needed by arms B and C)
    df_sa = ensure_support_surface_area(df.copy())

    # Validate features for all arms
    validate_features(df, ARM_A_ONLY_ATOM + ["Name"])
    validate_features(df_sa, ARM_B_ONLY_DESC + ["Name"])
    validate_features(df_sa, ARM_C_BOTH + ["Name"])

    ROOT = Path("results_OCM_3arms_multi")
    (ROOT/"comparison").mkdir(parents=True, exist_ok=True)

    # Arm folders with requested names
    outA = ROOT / "A_only_atom_number_Base"     # NOTE: renamed as requested
    outB = ROOT / "B_only_descriptors"          # NOTE: renamed/new arm
    outC = ROOT / "C_both_atom_and_descriptors" # NOTE: renamed as requested

    # Run experiments
    maeA, shapMeanA, shapStdA, gainMeanA, gainStdA = run_experiment_multi(df,    ARM_A_ONLY_ATOM, outA)
    maeB, shapMeanB, shapStdB, gainMeanB, gainStdB = run_experiment_multi(df_sa, ARM_B_ONLY_DESC, outB)
    maeC, shapMeanC, shapStdC, gainMeanC, gainStdC = run_experiment_multi(df_sa, ARM_C_BOTH,      outC)

    # ----- Averaged learning curves (MAE) over 5 test sets -----
    plt.figure(figsize=(8,4.8))
    plt.errorbar(maeA["size"], maeA["mean"], yerr=maeA["std"], fmt="-o", capsize=4, label="A) Only atom number (Base)")
    plt.errorbar(maeB["size"], maeB["mean"], yerr=maeB["std"], fmt="-o", capsize=4, label="B) Only descriptors")
    plt.errorbar(maeC["size"], maeC["mean"], yerr=maeC["std"], fmt="-o", capsize=4, label="C) Both")
    plt.xlabel("# training catalysts"); plt.ylabel("MAE (C₂ yield)")
    plt.title("Learning curves — averaged over 5 test sets")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT/"comparison"/"learning_curve_comparison_3arms_averaged.png")
    plt.show()

    comp = (
        maeA.rename(columns={"mean":"mae_mean_A","std":"mae_std_A"})
        .merge(maeB.rename(columns={"mean":"mae_mean_B","std":"mae_std_B"}), on="size", how="outer")
        .merge(maeC.rename(columns={"mean":"mae_mean_C","std":"mae_std_C"}), on="size", how="outer")
    )
    comp.to_csv(ROOT/"comparison"/"learning_curve_comparison_3arms_averaged.csv", index=False)

    # ----- SHAP Top-K vs size (averaged over test sets) for each arm -----
    def plot_topk_series(mean_map: Dict[int,pd.Series], label: str, outfile: Path):
        sizes_sorted = sorted(mean_map.keys())
        if not sizes_sorted:
            return
        all_means = pd.concat([mean_map[s].rename(s) for s in sizes_sorted], axis=1).fillna(0.0)
        global_topK = all_means.mean(axis=1).sort_values(ascending=False).head(TOP_K_FEATS).index.tolist()
        plt.figure(figsize=(8, 4 + 0.3*TOP_K_FEATS))
        for feat in global_topK:
            y = [mean_map[s].get(feat, 0.0) for s in sizes_sorted]
            plt.errorbar(sizes_sorted, y, marker="o", linestyle="-", capsize=4, label=feat)
        plt.xlabel("# training catalysts")
        plt.ylabel("mean |SHAP| on fixed tests (avg over 5 sets)")
        plt.title(f"Top-{TOP_K_FEATS} |SHAP| vs size — {label} (averaged)")
        plt.legend(title="Feature", bbox_to_anchor=(1.02,1), loc="upper left")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(outfile); plt.show()

    plot_topk_series(shapMeanA, "A) Only atom number (Base)", ROOT/"comparison"/f"shap_top{TOP_K_FEATS}_A_averaged.png")
    plot_topk_series(shapMeanB, "B) Only descriptors",       ROOT/"comparison"/f"shap_top{TOP_K_FEATS}_B_averaged.png")
    plot_topk_series(shapMeanC, "C) Both",                   ROOT/"comparison"/f"shap_top{TOP_K_FEATS}_C_averaged.png")

    # ----- XGB Gain Top-K vs size (averaged) for each arm -----
    def plot_topk_gain(mean_map: Dict[int,pd.Series], label: str, outfile: Path):
        sizes_sorted = sorted(mean_map.keys())
        if not sizes_sorted:
            return
        all_means = pd.concat([mean_map[s].rename(s) for s in sizes_sorted], axis=1).fillna(0.0)
        global_topK = all_means.mean(axis=1).sort_values(ascending=False).head(TOP_K_FEATS).index.tolist()
        plt.figure(figsize=(8, 4 + 0.3*TOP_K_FEATS))
        for feat in global_topK:
            y = [mean_map[s].get(feat, 0.0) for s in sizes_sorted]
            plt.errorbar(sizes_sorted, y, marker="o", linestyle="-", capsize=4, label=feat)
        plt.xlabel("# training catalysts")
        plt.ylabel("XGB gain importance (avg over 5 sets)")
        plt.title(f"Top-{TOP_K_FEATS} XGB gain vs size — {label} (averaged)")
        plt.legend(title="Feature", bbox_to_anchor=(1.02,1), loc="upper left")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(outfile); plt.show()

    plot_topk_gain(gainMeanA, "A) Only atom number (Base)", ROOT/"comparison"/f"gain_top{TOP_K_FEATS}_A_averaged.png")
    plot_topk_gain(gainMeanB, "B) Only descriptors",         ROOT/"comparison"/f"gain_top{TOP_K_FEATS}_B_averaged.png")
    plot_topk_gain(gainMeanC, "C) Both",                     ROOT/"comparison"/f"gain_top{TOP_K_FEATS}_C_averaged.png")

    print("\n✅ Finished 3-arm multi-test-set study! See results in:", ROOT)

# ───────── run ─────────
if __name__ == "__main__":
    main()
