#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCM – append five Mendeleev descriptors for M1 / M2 / M3
========================================================
Descriptors
-----------
1) electronegativity         (Pauling χ)
2) ionic_radius              (Å, radius of the ion with |charge| max)
3) oxidation_state           (maximum oxidation state)
4) d_electron_count          (ground-state d occupancy)
5) inverse_ionization        (1 / first ionisation energy, eV⁻¹)

© 2025 – minimalist, debug-friendly version
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
from mendeleev import element

# ─── Descriptor keys ───────────────────────────────────────────────────────────
DESC_KEYS = (
    "electronegativity",  # pauling
    "ionic_radius",
    "oxidation_state",  # max oxidation state
    "d_electron_count",
    "inverse_ionization",
)
ZERO_DICT = {k: 0.0 for k in DESC_KEYS}

# Support surface areas (m2/g)
SUPPORT_SURFACE_AREAS = {
    "Al2O3": 413.0,
    "BEA": 512.0,
    "BN": 1513.0,
    "CeO2": 102.0,
    "MgO": 194.0,
    "Nb2O5": 0.0,
    "SiC": 22.0,
    "SiCnf": 124.0,
    "SiO2": 350.0,
    "TiO2": 417.0,
    "ZSM-5": 285.0,
    "ZrO2": 37.0,
    "n.a.": 0.0,  # no support (blank)
}


def descriptors(sym: str) -> Dict[str, float]:
    """
    Return the five descriptors for `sym`.
    On ANY problem -> all zeros (never raises).
    """
    if sym in ("n.a.", "nan", "NaN", None) or sym != sym:
        return ZERO_DICT.copy()

    try:
        el = element(sym.strip().capitalize())

        # 1) electronegativity (Pauling χ)
        chi = el.en_pauling or 0.0

        # 2) ionic radius (Å)
        rad = 0.0
        if el.ionic_radii:
            # charge attribute renamed to ion_charge in ≥0.15
            best = max(
                el.ionic_radii,
                key=lambda r: abs(getattr(r, "ion_charge", getattr(r, "charge", 0))),
            )
            # radius attribute renamed to ionic_radius; crystal_radius is a fallback
            rad = getattr(best, "ionic_radius", None)
            if rad is None:
                rad = getattr(best, "crystal_radius", 0.0)

        # 3) maximum oxidation state
        ox = max(el.oxistates) if el.oxistates else 0.0

        # 4) d-electron count
        # 4) d-electron count
        d_cnt = 0.0

        # (A) Preferred: API method if available
        try:
            if hasattr(el, "ec") and hasattr(el.ec, "get_subshell_occupancy"):
                d_val = el.ec.get_subshell_occupancy("d")
                if d_val is not None:
                    d_cnt = float(d_val)
        except Exception:
            pass

        # (B) If still zero, parse conf tuples (len 2 or 3, labels may be '3d', '4d', etc.)
        if d_cnt == 0.0 and hasattr(el, "ec") and hasattr(el.ec, "conf"):
            try:
                for item in el.ec.conf:
                    subshell, electrons = item[
                        -2:
                    ]  # last two entries are label & count
                    if isinstance(subshell, str) and "d" in subshell:
                        d_cnt = float(electrons)
                        break
            except Exception:
                pass

        # (C) Last fallback: regex from a string representation
        if d_cnt == 0.0:
            cfg = ""
            if hasattr(el, "ec") and hasattr(el.ec, "conf_str"):
                cfg = el.ec.conf_str or ""
            elif hasattr(el, "econf"):
                cfg = str(el.econf)  # sometimes a dict, sometimes a string
            if cfg:
                import re

                m = re.findall(r"d(\d+)", cfg)
                if m:
                    d_cnt = float(m[-1])

        # 5) inverse first ionization energy
        ie1 = el.ionenergies.get(1, 0)
        inv_ie = 1.0 / ie1 if ie1 else 0.0

        return {
            "electronegativity": chi,
            "ionic_radius": rad,
            "oxidation_state": ox,
            "d_electron_count": d_cnt,
            "inverse_ionization": inv_ie,
        }

    except Exception as err:
        print(f"[warn]   {sym}: {err}", file=sys.stderr)
        return ZERO_DICT.copy()


def build_cache(symbols):
    return {s: descriptors(str(s)) for s in symbols}


def main(csv_path: Path):
    df = pd.read_csv(csv_path)
    # the original file has a trailing space in the column name "Support "
    df.columns = df.columns.str.strip()

    # build cache of all M1/M2/M3 symbols
    symbols = pd.unique(df[["M1", "M2", "M3"]].values.ravel())
    cache = build_cache(symbols)

    # append descriptor-columns
    for idx, col in enumerate(("M1", "M2", "M3"), start=1):
        base = f"M{idx}"
        for k in DESC_KEYS:
            new_col = f"{base}_{k}"
            df[new_col] = df[col].map(lambda s, kk=k: cache.get(str(s), ZERO_DICT)[kk])

    # append support surface area
    unknown = set(df["Support"]) - set(SUPPORT_SURFACE_AREAS)
    if unknown:
        raise ValueError(f"No surface area given for supports: {unknown}")
    df["support_surface_area"] = df["Support"].map(SUPPORT_SURFACE_AREAS)

    # show a quick preview
    pd.set_option("display.max_columns", None, "display.max_rows", 8)
    print(df.head())

    # save to a new CSV (works on Python <3.9)
    out_csv = csv_path.with_name(f"{csv_path.stem}_with_descriptors{csv_path.suffix}")
    df.to_csv(out_csv, index=False)
    print(f"\nAugmented file → {out_csv.resolve()}")

    # plot distributions
    plt.figure(figsize=(10, 8))
    for i, k in enumerate(DESC_KEYS, 1):
        vals = pd.concat([df[f"M1_{k}"], df[f"M2_{k}"], df[f"M3_{k}"]])
        plt.subplot(3, 2, i)
        plt.hist(vals, bins=30, edgecolor="k")
        plt.title(k.replace("_", " ").title())
        plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    import argparse

    ap = argparse.ArgumentParser(
        description="Append 5 Mendeleev descriptors for M1/M2/M3"
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("Dataset/OCM-NguyenEtAl.csv"),
        help="Path to CSV file (default: Dataset/OCM-NguyenEtAl.csv, relative to the repository root)",
    )

    args, _ = ap.parse_known_args()
    main(args.csv)
