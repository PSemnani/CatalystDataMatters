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
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Skip duplicate-row removal entirely, even if the columns needed to identify duplicates are present.",
    )
    args = parser.parse_args()
    experiment_path = Path(args.experiment_path)
    #load data
    if not (experiment_path / "gathered_results.csv").exists():
        # go through all folders and load file "training_results.csv" and concatenate them into a single dataframe
        all_results = get_results_csv(experiment_path)
        print(f"Found {len(all_results)} result files.")
        if len(all_results) == 0:
            print("ERROR: No result files found. Did you choose the correct folder? Exiting.")
            return
        # concatenate all results
        results_df = pd.concat(all_results, ignore_index=True)
        # remove duplicate runs (e.g. from reruns), keeping the row with the fewest NaNs
        dedup_cols = [
            "model_type",
            "feature_set",
            "augmentation",
            "n_train_catalysts",
            "n_test_catalysts",
            "seed",
            "condition",
        ]
        available_dedup_cols = [col for col in dedup_cols if col in results_df.columns]
        missing_dedup_cols = [col for col in dedup_cols if col not in results_df.columns]
        if args.no_dedup:
            print("Skipping duplicate removal (--no_dedup was given).")
        elif not available_dedup_cols:
            print(
                f"WARNING: None of the columns {dedup_cols} needed to identify duplicates are present. "
                "Skipping duplicate removal."
            )
        else:
            if missing_dedup_cols:
                print(
                    f"WARNING: Columns {missing_dedup_cols} needed to identify duplicates are missing. "
                    f"Deduplicating using only {available_dedup_cols} instead."
                )
            n_before = len(results_df)
            nan_counts = results_df.isna().sum(axis=1)
            results_df = (
                results_df.assign(_nan_count=nan_counts)
                .sort_values("_nan_count", kind="stable")
                .drop_duplicates(subset=available_dedup_cols, keep="first")
                .drop(columns="_nan_count")
                .sort_index()
            )
            n_removed = n_before - len(results_df)
            if n_removed > 0:
                print(f"Removed {n_removed} duplicate rows (kept the row with fewest NaN values for each duplicate).")
        # save gathered results
        results_df.to_csv(experiment_path / "gathered_results.csv", index=False)
        print(f"Saved gathered results to {experiment_path / 'gathered_results.csv'}.")
    else:
        print(f"Found existing gathered results at {experiment_path / 'gathered_results.csv'}.")
        print("Please rename or delete the existing gathered results file if you want to gather results again.")


if __name__ == "__main__":
    main()
