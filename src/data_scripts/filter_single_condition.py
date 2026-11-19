"""
Filter the OCM dataset down to exactly one row per catalyst, at the fixed
condition T=800, CH4/O2=4, CT=0.5, Total_flow=15, Ar_flow=6 (Ar_flow is
needed in addition to Total_flow/CT/CH4_O2 since those three alone are
satisfied by 3 different flow combinations per catalyst).
"""
import argparse
from pathlib import Path

import pandas as pd

CONDITION = {
    "Temp": 800,
    "CH4/O2": 4,
    "CT": 0.5,
    "Total_flow": 15,
    "Ar_flow": 6,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Dataset/OCM-NguyenEtAl_with_descriptors.csv"),
        help="Path to the data with descriptors (default: Dataset/OCM-NguyenEtAl_with_descriptors.csv, "
        "relative to the repository root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the filtered output (default: input path with the suffix _single_condition, e.g. "
        "Dataset/OCM-NguyenEtAl_with_descriptors_single_condition.csv)",
    )
    args = parser.parse_args()
    output_path = args.output or args.input.with_name(f"{args.input.stem}_single_condition.csv")

    df = pd.read_csv(args.input)

    mask = pd.Series(True, index=df.index)
    for col, value in CONDITION.items():
        mask &= df[col] == value
    filtered = df[mask]

    n_catalysts = df["Name"].nunique()
    duplicated = filtered["Name"].value_counts()
    duplicated = duplicated[duplicated > 1]
    if len(duplicated) > 0:
        raise ValueError(f"Multiple rows found for catalysts: {duplicated.index.tolist()}")
    missing = set(df["Name"].unique()) - set(filtered["Name"].unique())
    if missing:
        raise ValueError(f"No matching row found for catalysts: {missing}")

    print(f"Filtered {len(df)} rows down to {len(filtered)} rows ({n_catalysts} catalysts).")
    filtered.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
