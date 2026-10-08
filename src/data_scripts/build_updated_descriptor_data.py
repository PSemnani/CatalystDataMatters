"""
Build the OCM dataset with updated descriptors directly from the original data
(Dataset/OCM-NguyenEtAl.csv) and store it as <input>_with_descriptors_updated.csv
next to the input file.

Corrections of the original data:
- M1/M2/M3_mol%: catalysts with different metal compositions in different rows are
  set to the composition that is consistent with the ratio of the metal loadings
  M2_mol / M3_mol. In the original data, Mn-MgWO4, Mn-CaWO4, Mn-SrWO4, Mn-BaWO4,
  Mn-FeMoO4, and Mn-ZnMoO4 (on SiO2) have 40/40/20 mol% at 700-850 C instead of
  50/25/25 mol% (as at 900 C and in line with their loadings).

Missing values, i.e. the descriptors of empty metal sites ("n.a."), the support descriptors
of catalysts without support (blank), and the optical basicity of non-oxidic supports, are
NaN by default (handled by XGBoost). With --missing_value 0 they are set to 0 instead,
similar to the previous version of the data.

Descriptors of the metals M1/M2/M3, all elemental data from pymatgen:
- electronegativity: Pauling electronegativity.
- ionic_radius: Shannon radius (in pm) for coordination number 6 (CN VI) at the
  oxidation state below. If radii are given for several spin states (e.g. Fe3+, Co3+),
  the high spin radius is used.
- oxidation_state: highest common oxidation state of the element.
- d_electron_count: number of d electrons outside the noble-gas core of the neutral
  atom in its ground state, e.g. 1 for Y ([Kr] 4d1 5s2).
- inverse_ionization: 1 / first ionization energy (in 1/eV).
- inverse_second_ionization: 1 / second ionization energy (in 1/eV).
- mulliken_electronegativity: Mulliken electronegativity (IE1 + EA) / 2 (in eV) with the
  first ionization energy IE1 and the electron affinity EA. Negative electron affinities
  (elements without stable anion) are set to 0, as for the support band center below.
Support:
- support_surface_area: BET surface areas (in m2/g) of the bare supports as reported
  in the experimental section of the original publication of the data [1].
- support_band_center: estimate of the absolute band center of the support (in eV) from
  the electronegativities of its elements [3]. Conventions:
  * Mulliken electronegativity of every element: chi = (IE1 + EA) / 2 with the first
    ionization energy IE1 and the electron affinity EA from pymatgen.
  * Negative electron affinities are set to 0 (elements without stable anion, e.g. Mg
    with EA = -0.42 eV and N with EA = -0.07 eV in pymatgen), i.e. an unbound anion does
    not contribute to the electronegativity.
  * Band center = geometric mean of the electronegativities of the elements, weighted
    with their atomic fractions in the support formula (e.g. SiO2: chi_Si^(1/3) *
    chi_O^(2/3)). The value is stored as a positive number; the band center relative to
    the vacuum level is its negative (e.g. 6.47 for SiO2 means -6.47 eV).
  * Supports whose name is not a chemical formula use the compositions in
    SUPPORT_FORMULAS (SiC nanofibers: SiC; zeolites: nominal framework compositions).
  * No support (blank): missing value.
- support_optical_basicity: optical basicity of the (oxidic) support from Table 4 of [2];
  for the zeolites computed with Equation 5 of [2] (see SUPPORT_OPTICAL_BASICITY). Not
  defined for non-oxides (BN, SiC, SiC nanofibers) and the blank, which get the missing
  value.
- support_class_oxide, support_class_zeolite, support_class_carbide,
  support_class_nitride, support_class_none: one-hot encoding (0/1) of the material
  class of the support (see SUPPORT_CLASS), e.g. support_class_oxide = 1 for SiO2.

References:
[1] Nguyen, T. N., Nhat, T. T. P., Takimoto, K., Thakur, A., Nishimura, S., Ohyama, J.,
    ... & Taniike, T. (2020). High-throughput experimentation and catalyst informatics
    for oxidative coupling of methane. ACS Catalysis, 10(2), 921-932.
[2] Lebouteiller, A., & Courtine, P. (1998). Improvement of a bulk optical basicity table
    for oxidic systems. Journal of Solid State Chemistry, 137(1), 94-103.
[3] Butler, M. A., & Ginley, D. S. (1978). Prediction of flatband potentials at
    semiconductor-electrolyte interfaces from atomic electronegativities. Journal of the
    Electrochemical Society, 125(2), 228-232.

Usage: python src/data_scripts/build_updated_descriptor_data.py [--csv Dataset/OCM-NguyenEtAl.csv]
           [--missing_value 0]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Species

SITES = ["M1", "M2", "M3"]
EMPTY_SITE = "n.a."
DESC_KEYS = (
    "electronegativity",
    "ionic_radius",
    "oxidation_state",
    "d_electron_count",
    "inverse_ionization",
    "inverse_second_ionization",
    "mulliken_electronegativity",
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
    "SiCnf": 1.5,  # not reported for the SiC nanofibers, value of SiC is used
    "BEA": 560.0,
    "ZSM-5": 300.0,
    "TiO2": 17.4,
    "ZrO2": 3.2,
    "Nb2O5": 4.7,
    "CeO2": 3.9,
    "n.a.": None,  # no support (blank)
}
NO_SUPPORT = "n.a."
# compositions of supports whose name is not a chemical formula (for the band center)
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
# Optical basicities of the cations used for the zeolites, from Lebouteiller & Courtine,
# J. Solid State Chem. 137, 94-103 (1998): Si4+ CN4 and Al3+ CN6 from Table 2, Na+ CN6
# from Table 3. Tetrahedral framework Al3+ and H+ are not
# tabulated, so the values of CN6 Al and of H2O are used; both carry under 2% weight, so
# neither changes the third decimal.
CATION_OPTICAL_BASICITY = {"Si": 0.48, "Al": 0.60, "H": 0.40, "Na": 1.10}
CATION_CHARGE = {"Si": 4, "Al": 3, "H": 1, "Na": 1}


def zeolite_optical_basicity(formula):
    """Optical basicity of a zeolite from its framework composition (e.g. HAlSi52O106) with
    the charge-neutralisation rule (Equation 5 of Lebouteiller & Courtine, 1998): every cation is weighted with
    the share of the negative charge of the oxygen atoms (2 per O) that it neutralises."""
    amounts = Composition(formula).get_el_amt_dict()
    total_charge = 2 * amounts["O"]
    return sum(
        amount * CATION_CHARGE[el] / total_charge * CATION_OPTICAL_BASICITY[el]
        for el, amount in amounts.items()
        if el != "O"
    )


# optical basicity of the supports (None: not defined, i.e. non-oxides and no support)
# oxides: Table 4 of Lebouteiller & Courtine, J. Solid State Chem. 137, 94-103 (1998);
# zeolites: computed from their framework compositions
SUPPORT_OPTICAL_BASICITY = {
    "Al2O3": 0.60,  # Al3+ CN6. NOTE: Table 4 is corundum; yours is gamma (~30% tetrahedral Al)
    "BEA": zeolite_optical_basicity(SUPPORT_FORMULAS["BEA"]),  # H2Al2Si104O212: 0.4813
    "BN": None,  # nitride
    "CeO2": 0.65,  # Ce4+ CN8. From the d0 hypothesis line, authors advise caution
    "MgO": 0.78,  # Mg2+ CN6
    "Nb2O5": 0.61,  # Nb5+ CN6. d0 line, authors advise caution
    "SiC": None,  # carbide
    "SiCnf": None,  # carbide
    "SiO2": 0.48,  # Si4+ CN4
    "TiO2": 0.75,  # Ti4+ CN6, rutile. Anatase noted as slightly below 0.75
    "ZSM-5": zeolite_optical_basicity(SUPPORT_FORMULAS["ZSM-5"]),  # Na2Al2Si90O184: 0.4853
    "ZrO2": 0.71,  # monoclinic (CN8) and tetragonal (CN7) both give 0.71. d0 line, authors advise caution
    "n.a.": None,  # unsupported
}
# material classes of the supports (one-hot encoded as support_class_<class>)
SUPPORT_CLASSES = ["oxide", "zeolite", "carbide", "nitride", "none"]
SUPPORT_CLASS = {
    "Al2O3": "oxide",
    "CeO2": "oxide",
    "MgO": "oxide",
    "Nb2O5": "oxide",
    "SiO2": "oxide",
    "TiO2": "oxide",
    "ZrO2": "oxide",
    "BEA": "zeolite",
    "ZSM-5": "zeolite",
    "SiC": "carbide",
    "SiCnf": "carbide",
    "BN": "nitride",
    "n.a.": "none",  # no support (blank)
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


def mulliken_electronegativity(symbol):
    # (IE1 + EA) / 2 in eV; negative electron affinities (no stable anion) are set to 0
    el = Element(symbol)
    return (float(el.ionization_energy) + max(float(el.electron_affinity), 0.0)) / 2


def support_band_center(support):
    """Band center estimate (Butler & Ginley) of the support as a positive number in eV
    (geometric mean of the Mulliken electronegativities weighted with atomic fractions)."""
    if support == NO_SUPPORT:
        return None
    composition = Composition(SUPPORT_FORMULAS.get(support, support)).fractional_composition
    return float(np.exp(sum(
        fraction * np.log(mulliken_electronegativity(str(el))) for el, fraction in composition.items()
    )))


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
        "inverse_second_ionization": 1.0 / float(el.ionization_energies[1]),
        "mulliken_electronegativity": mulliken_electronegativity(symbol),
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
        default=float("nan"),
        help="Value for missing descriptors (empty metal sites, no support, optical basicity of "
        "non-oxides). Default: NaN; use 0 for a dataset similar to the previous version. For values "
        "other than NaN, the output file name gets the suffix _missing<value>.",
    )
    args = parser.parse_args()
    suffix = "" if np.isnan(args.missing_value) else f"_missing{args.missing_value:g}"
    output_path = args.csv.with_name(f"{args.csv.stem}_with_descriptors_updated{suffix}.csv")

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

    # append support band center
    band_centers = {support: support_band_center(support) for support in df["Support"].unique()}
    df["support_band_center"] = df["Support"].map(band_centers)
    print("support band centers (eV, absolute values):",
          {support: round(value, 3) for support, value in
           sorted(((s, v) for s, v in band_centers.items() if v is not None), key=lambda x: x[1])})

    # append support optical basicity (missing where not defined)
    unknown = set(df["Support"]) - set(SUPPORT_OPTICAL_BASICITY)
    if unknown:
        raise ValueError(f"No optical basicity given for supports: {unknown}")
    df["support_optical_basicity"] = df["Support"].map(SUPPORT_OPTICAL_BASICITY)

    # append one-hot encoded support class
    unknown = set(df["Support"]) - set(SUPPORT_CLASS)
    if unknown:
        raise ValueError(f"No support class given for supports: {unknown}")
    support_class = df["Support"].map(SUPPORT_CLASS)
    for cls in SUPPORT_CLASSES:
        df[f"support_class_{cls}"] = (support_class == cls).astype(int)

    # missing values (None/NaN) of the computed descriptors: NaN or the given value
    descriptor_cols = [f"{site}_{key}" for site in SITES for key in DESC_KEYS] + [
        "support_surface_area",
        "support_band_center",
        "support_optical_basicity",
    ]
    df[descriptor_cols] = df[descriptor_cols].astype(float)
    if not np.isnan(args.missing_value):
        df[descriptor_cols] = df[descriptor_cols].fillna(args.missing_value)
    print(f"missing values: {int(df[descriptor_cols].isna().sum().sum())} NaN entries in the descriptors")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
