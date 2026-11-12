"""
Build the OCM dataset with descriptors of the metals and the support from the original
data (Dataset/OCM-NguyenEtAl.csv) and store it as <input>_with_descriptors.csv next to
the input file.

Corrections of the original data:
- M1/M2/M3_mol%: catalysts with different metal compositions in different rows are
  set to the composition that is consistent with the ratio of the metal loadings
  M2_mol / M3_mol. In the original data (as of Oct. 1, 2026), Mn-MgWO4, Mn-CaWO4,
  Mn-SrWO4, Mn-BaWO4, Mn-FeMoO4, and Mn-ZnMoO4 (on SiO2) have 40/40/20 mol% at
  700-850 C instead of 50/25/25 mol% (as at 900 C and in line with their loadings).
  This might be fixed in future updates of the data set; catalysts with a consistent
  composition are left unchanged, so the result is the same for corrected data.

Missing values, i.e. the descriptors of empty metal sites ("n.a.") and the support surface
area of the catalyst without support (blank), are 0 by default (usable by all models). With
--missing_value nan they are NaN instead (handled by XGBoost, but not by all models).

Descriptors of the metals M1/M2/M3, all elemental data from pymatgen:
- electronegativity: Pauling electronegativity.
- ionic_radius: Shannon radius (in pm) for coordination number 6 (CN VI) at the
  oxidation state below. If radii are given for several spin states (e.g. Fe3+, Co3+),
  the high spin radius is used.
- oxidation_state: highest common oxidation state of the element.
- d_electron_count: number of d electrons outside the noble-gas core of the neutral
  atom in its ground state, e.g. 1 for Y ([Kr] 4d1 5s2).
- inverse_ionization: 1 / first ionization energy (in 1/eV).
Support:
- support_surface_area: BET surface areas (in m2/g) of the bare supports as reported
  in Table 1 of the original publication of the data [1]. For the SiC nanofibers (SiCnf),
  no value is reported; a geometric lower bound is estimated from the manufacturer's
  specifications [2] (D < 2.5 um, L/D >= 20, rho = 3.22 g/cm3) for long, dense and smooth
  cylinders, SSA = 4 / (rho * D) ~ 0.5 m2/g for D = 2.5 um (the neglected end faces would
  contribute at most 2.5 % for L/D >= 20).

References:
[1] Nguyen, T. N., Nhat, T. T. P., Takimoto, K., Thakur, A., Nishimura, S., Ohyama, J.,
    ... & Taniike, T. (2020). High-throughput experimentation and catalyst informatics
    for oxidative coupling of methane. ACS Catalysis, 10(2), 921-932.
[2] Sigma-Aldrich. Silicon carbide, nanofiber, D <2.5 um, L/D >= 20, 98% trace metals
    basis, Product No. 776742. https://www.sigmaaldrich.com/product/aldrich/776742
    (accessed 2026-10-28).

Usage: python src/data_scripts/build_descriptor_data.py [--csv Dataset/OCM-NguyenEtAl.csv]
           [--missing_value nan]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Element, Species

SITES = ["M1", "M2", "M3"]
EMPTY_SITE = "n.a."
DESC_KEYS = (
    "electronegativity",
    "ionic_radius",
    "oxidation_state",
    "d_electron_count",
    "inverse_ionization",
)
# coordination number for the Shannon radii
CN = "VI"
# BET surface areas (m2/g) of the bare supports from the original publication
SUPPORT_SURFACE_AREAS = {
    "BN": 5.2,
    "MgO": 5.5,
    "Al2O3": 150.0,  # gamma-Al2O3
    "SiO2": 650.0,  # silica gel 60N
    "SiC": 1.5,
    "SiCnf": 0.5,  # not reported, geometric lower bound for D < 2.5 um, L/D >= 20 (see above)
    "BEA": 560.0,
    "ZSM-5": 300.0,
    "TiO2": 17.4,
    "ZrO2": 3.2,
    "Nb2O5": 4.7,
    "CeO2": 3.9,
    "n.a.": None,  # no support (blank)
}


def fix_mol_fractions(df):
    """Set the mol% of catalysts with different compositions in different rows to
    the composition whose M2:M3 ratio is consistent with the loadings M2_mol:M3_mol."""
    df = df.copy()
    mol_cols = [f"{site}_mol%" for site in SITES]
    for name, group in df.groupby("Name"):
        compositions = group[mol_cols].drop_duplicates()
        if len(compositions) == 1:
            continue
        loadings = group[["M2_mol", "M3_mol"]].drop_duplicates()
        if len(loadings) != 1:
            raise ValueError(f"{name}: different metal loadings in different rows")
        m2_load, m3_load = loadings.iloc[0]
        consistent = compositions[
            np.isclose(
                compositions["M2_mol%"] * m3_load,
                compositions["M3_mol%"] * m2_load,
                rtol=0.05,
            )
        ]
        if len(consistent) != 1:
            raise ValueError(
                f"{name}: cannot determine the composition consistent with the loadings "
                f"from {compositions.values.tolist()}"
            )
        composition = consistent.iloc[0]
        mask = (df["Name"] == name) & (df[mol_cols] != composition.values).any(axis=1)
        print(
            f"{name}: mol% {df.loc[mask, mol_cols].drop_duplicates().values.tolist()} -> "
            f"{composition.values.tolist()} ({mask.sum()} rows)"
        )
        df.loc[mask, mol_cols] = composition.values
    return df


def shannon_radius_pm(symbol, oxidation_state):
    species = Species(symbol, oxidation_state)
    try:
        radius = species.get_shannon_radius(cn=CN, radius_type="ionic")
    except KeyError:
        # radii are given for several spin states: use high spin
        radius = species.get_shannon_radius(cn=CN, spin="High Spin", radius_type="ionic")
    return round(float(radius) * 100, 2)  # Angstrom -> pm


def valence_d_electron_count(symbol):
    # d electrons outside the noble-gas core of the neutral atom, e.g. "[Kr].4d1.5s2" -> 1
    subshells = Element(symbol).electronic_structure.split(".")[1:]
    return float(sum(int(subshell.split("d")[1]) for subshell in subshells if "d" in subshell))


def descriptors(symbol):
    if symbol == EMPTY_SITE:
        return {key: None for key in DESC_KEYS}
    el = Element(symbol)
    oxidation_state = max(el.common_oxidation_states)
    return {
        "electronegativity": float(el.X),
        "ionic_radius": shannon_radius_pm(symbol, oxidation_state),
        "oxidation_state": float(oxidation_state),
        "d_electron_count": valence_d_electron_count(symbol),
        "inverse_ionization": 1.0 / float(el.ionization_energies[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Dataset/OCM-NguyenEtAl.csv"),
        help="Path to the original data (default: Dataset/OCM-NguyenEtAl.csv, relative to the repository root)",
    )
    parser.add_argument(
        "--missing_value",
        type=float,
        default=0.0,
        help="Value for missing descriptors (empty metal sites, surface area of the blank). "
        "Default: 0; use nan to keep them missing. For values other than 0, the output file "
        "name gets the suffix _missing<value> (e.g. _missingnan).",
    )
    args = parser.parse_args()
    suffix = "" if args.missing_value == 0 else f"_missing{args.missing_value:g}"
    output_path = args.csv.with_name(f"{args.csv.stem}_with_descriptors{suffix}.csv")

    df = pd.read_csv(args.csv)
    # the original file has a trailing space in the column name "Support "
    df.columns = df.columns.str.strip()
    df = fix_mol_fractions(df)

    # append metal descriptors (M1_electronegativity, ..., M3_inverse_ionization)
    cache = {symbol: descriptors(symbol) for symbol in pd.unique(df[SITES].values.ravel())}
    for site in SITES:
        for key in DESC_KEYS:
            df[f"{site}_{key}"] = df[site].map(lambda symbol, key=key: cache[symbol][key])

    # append support surface area
    unknown = set(df["Support"]) - set(SUPPORT_SURFACE_AREAS)
    if unknown:
        raise ValueError(f"No surface area given for supports: {unknown}")
    df["support_surface_area"] = df["Support"].map(SUPPORT_SURFACE_AREAS)

    # missing values (None/NaN) of the computed descriptors: NaN or the given value
    descriptor_cols = [f"{site}_{key}" for site in SITES for key in DESC_KEYS] + ["support_surface_area"]
    df[descriptor_cols] = df[descriptor_cols].astype(float)
    if not np.isnan(args.missing_value):
        df[descriptor_cols] = df[descriptor_cols].fillna(args.missing_value)
    print(f"missing values: {int(df[descriptor_cols].isna().sum().sum())} NaN entries in the descriptors")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
