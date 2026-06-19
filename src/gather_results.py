from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import get_results_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather results from results.csv files in experiment subfolders."
    )
    parser.add_argument("experiment_path", help="Path to the experiment folder")
    args = parser.parse_args()
    experiment_path = Path(args.experiment_path)
    #load data
    if not (experiment_path / "gathered_results.csv").exists():
        # go through all folders and load file "results_summary.csv" and concatenate them into a single dataframe
        all_results = get_results_csv(experiment_path)
        print(f"Found {len(all_results)} result files.")
        # concatenate all results
        results_df = pd.concat(all_results, ignore_index=True)
        # save gathered results
        results_df.to_csv(experiment_path / "gathered_results.csv", index=False)
        print(f"Saved gathered results to {experiment_path / 'gathered_results.csv'}.")
    else:
        print(f"Found existing gathered results at {experiment_path / 'gathered_results.csv'}.")
        print("Please rename or delete the existing gathered results file if you want to gather results again.")


if __name__ == "__main__":
    main()
