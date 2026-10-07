"""
Compute XenonPy compositional descriptors of the active phase (metals M1-M3) and of
the support for every catalyst in the original OCM data and store them in a table
with one row per catalyst (column "Name" and the descriptor columns).

- Active phase (columns xp_active_*): metal composition weighted by M1_mol%, M2_mol%,
  M3_mol%. The compositions are taken from the rows at 900 C, since the original data
  contains inconsistent mol% for some catalysts at other temperatures. Catalysts without
  data at 900 C use their other rows, which must then have a consistent composition.
- Support (columns xp_support_*): stoichiometric composition of the support.
For both, XenonPy computes the weighted average, weighted variance, maximum, and minimum
of 58 elemental properties. Catalysts without active phase (bare supports, blank) or
without support (blank) get zeros for the respective descriptors.

Requires the "xenonpy" environment (xenonpy, pymatgen, rdkit) and the element data
(from xenonpy.datatools import preset; preset.sync("elements_completed")).
Usage: python src/data_scripts/compute_xenonpy_descriptors.py [--csv Dataset/OCM-NguyenEtAl.csv]
           [--output Dataset/xenonpy_descriptors.csv]
"""
import argparse
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition
from xenonpy.descriptor import Compositions

SITES = ["M1", "M2", "M3"]
MOL_COLS = [f"{site}_mol%" for site in SITES]
EMPTY_SITE = "n.a."
NO_SUPPORT = "n.a."
REFERENCE_TEMPERATURE = 900
# supports whose name is not a chemical formula
# The zeolite SiO2/Al2O3 ratios and forms are taken from the experimental section
# of the original publication (Nguyen et al.), i.e. M2O*Al2O3*nSiO2 with
# n = SiO2/Al2O3 = 104 for BEA (M = H) and n = 90 for ZSM-5 (M = Na).
# Nominal framework compositions: Si/Al = n / 2, one counter-cation (H+ or Na+)
# per Al, and 2 * (Si + Al) oxygen atoms (corner-sharing TO4 tetrahedra).
SUPPORT_FORMULAS = {
    "SiCnf": "SiC",  # SiC nanofibers
    "BEA": "HAlSi52O106",  # H-form, SiO2/Al2O3 = 104 (HSZ-960HOA)
    "ZSM-5": "NaAlSi45O92",  # Na-form, SiO2/Al2O3 = 90 (JRC-Z5-90NA)
}
# compositions are normalized to fractions, so WeightedSum would equal WeightedAverage
FEATURIZERS = ["WeightedAverage", "WeightedVariance", "MaxPooling", "MinPooling"]


def catalyst_rows(df):
    """One row per catalyst with its composition, preferably from the rows at 900 C."""
    rows = []
    for name, group in df.groupby("Name", sort=False):
        reference = group[group["Temp"] == REFERENCE_TEMPERATURE]
        if len(reference) == 0:
            reference = group
            print(f"{name}: no data at {REFERENCE_TEMPERATURE} C, using all rows")
        composition_cols = SITES + MOL_COLS + ["Support"]
        if len(reference[composition_cols].drop_duplicates()) != 1:
            raise ValueError(f"{name}: inconsistent composition in the reference rows")
        rows.append(reference.iloc[0])
    return pd.DataFrame(rows).set_index("Name")


def active_composition(row):
    comp = {
        row[site]: row[f"{site}_mol%"]
        for site in SITES
        if row[site] != EMPTY_SITE and row[f"{site}_mol%"] > 0
    }
    return Composition(comp).fractional_composition if comp else None


def support_composition(support):
    if support == NO_SUPPORT:
        return None
    return Composition(SUPPORT_FORMULAS.get(support, support)).fractional_composition


def featurize(compositions, prefix):
    """Featurize a Series of compositions (None for missing ones, which get zeros)."""
    calculator = Compositions(featurizers=FEATURIZERS, n_jobs=1)
    present = compositions.dropna()
    features = calculator.transform(present.tolist())
    features.index = present.index
    features = features.reindex(compositions.index, fill_value=0.0)
    features.columns = [f"{prefix}_{c.replace(':', '_')}" for c in features.columns]
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("Dataset/OCM-NguyenEtAl.csv"),
                        help="Path to the original data (default: Dataset/OCM-NguyenEtAl.csv)")
    parser.add_argument("--output", type=Path, default=Path("Dataset/xenonpy_descriptors.csv"),
                        help="Path of the output table (default: Dataset/xenonpy_descriptors.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # the original file has a trailing space in the column name "Support "
    df.columns = df.columns.str.strip()
    catalysts = catalyst_rows(df)

    active = featurize(catalysts.apply(active_composition, axis=1), "xp_active")
    support = featurize(catalysts["Support"].map(support_composition), "xp_support")
    descriptors = pd.concat([active, support], axis=1)
    descriptors.index.name = "Name"
    if descriptors.isna().any().any():
        raise ValueError("NaN in XenonPy descriptors.")

    descriptors.to_csv(args.output)
    print(
        f"Saved {active.shape[1]} active-phase and {support.shape[1]} support descriptors "
        f"for {len(descriptors)} catalysts to {args.output}"
    )


if __name__ == "__main__":
    main()
