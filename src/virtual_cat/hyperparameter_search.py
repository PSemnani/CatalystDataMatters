"""
Hyperparameter Search for Virtual Catalyst Data
================================================
Tests 50 random hyperparameter combinations to find optimal XGBoost settings.
Uses a fixed train/test split and evaluates all parameter combinations.
"""

import random
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils_virtual_catalyst import (
    ALL_FEATURES,
    TARGET,
    get_random_xgb_params,
    split_data_catalyst,
    scale_data,
    compute_metrics,
)
import xgboost as xgb


def train_with_params(
    X_train, y_train,
    X_test, y_test,
    feature_cols,
    random_state,
    params,
):
    """Train XGBoost with specific hyperparameters"""
    # Scale data
    X_train, _, X_test, scaler = scale_data(
        X_train, None, X_test, feature_cols
    )
    
    # Initialize model with custom params
    model_seed = random_state.randint(0, 1000000)
    model = xgb.XGBRegressor(
        random_state=model_seed,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        **params
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, preds)
    
    return metrics


def main(
    data_path,
    seeds,
    n_train_catalysts=200,
    n_test_catalysts=10,
    n_param_samples=50,
    output_folder="hyperparameter_search",
    fast_test=False,
):
    """
    Run hyperparameter search.
    
    Args:
        data_path: Path to CSV data
        seeds: List of random seeds
        n_train_catalysts: Number of training catalysts
        n_test_catalysts: Number of test catalysts
        n_param_samples: Number of random parameter combinations to try
        output_folder: Output directory
        fast_test: If True, use only 10 parameter samples and 2 seeds
    """
    if fast_test:
        print("\n" + "="*60)
        print("FAST TEST MODE ENABLED")
        print("="*60)
        n_param_samples = 10
        seeds = seeds[:2]
        output_folder = output_folder + "_fast_test"
        print(f"Using {n_param_samples} parameter samples")
        print(f"Using {len(seeds)} seeds")
        print(f"Output folder: {output_folder}")
        print("="*60 + "\n")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Training catalysts: {n_train_catalysts}")
    print(f"Test catalysts: {n_test_catalysts}")
    
    # Create results directory
    results_path = Path(f"./{output_folder}/")
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Generate random hyperparameter combinations
    print(f"\nGenerating {n_param_samples} random hyperparameter combinations...")
    param_combinations = get_random_xgb_params(n_samples=n_param_samples, seed=42)
    
    # Save parameter combinations
    params_df = pd.DataFrame(param_combinations)
    params_df.to_csv(results_path / "parameter_combinations.csv", index=False)
    print(f"Saved parameter combinations to {results_path / 'parameter_combinations.csv'}")
    
    # Feature columns
    feature_cols = ALL_FEATURES
    target_col = TARGET
    
    # Storage for results
    results_rows = []
    
    print(f"\n{'='*60}")
    print(f"Running hyperparameter search:")
    print(f"  {len(seeds)} seeds × {n_param_samples} param combinations = {len(seeds) * n_param_samples} experiments")
    print(f"{'='*60}\n")
    
    # Loop over seeds
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        rng = random.Random(seed)
        xgb_rng = random.Random(rng.randint(0, 1000000))
        
        # Split data once per seed
        X_train, y_train, _, _, X_test, y_test, _, _, _ = split_data_catalyst(
            df,
            feature_cols,
            target_col,
            rng,
            n_train_catalysts=n_train_catalysts,
            n_val_catalysts=0,
            n_test_catalysts=n_test_catalysts,
        )
        
        print(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")
        print(f"Testing {n_param_samples} parameter combinations...")
        
        # Loop over parameter combinations with progress bar
        for param_idx, params in enumerate(tqdm(param_combinations, desc=f"Seed {seed}")):
            # Create a fresh random state for this model
            model_rng = random.Random(xgb_rng.randint(0, 1000000))
            
            try:
                metrics = train_with_params(
                    X_train.copy(), y_train.copy(),
                    X_test.copy(), y_test.copy(),
                    feature_cols,
                    random_state=model_rng,
                    params=params,
                )
                
                # Store results
                results_rows.append({
                    "seed": seed,
                    "param_idx": param_idx,
                    "n_train_catalysts": n_train_catalysts,
                    "n_test_catalysts": n_test_catalysts,
                    "n_train_samples": len(y_train),
                    "n_test_samples": len(y_test),
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "mse": metrics["mse"],
                    "rmse": metrics["rmse"],
                    **params,
                })
            except Exception as e:
                print(f"\nError with params {param_idx}: {e}")
                continue
    
    # Save results
    results_df = pd.DataFrame(results_rows)
    csv_path = results_path / "results_all.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved all results to {csv_path}")
    
    # Analyze results
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*60)
    
    # Best parameters by mean performance across seeds
    mean_performance = results_df.groupby("param_idx").agg({
        "r2": ["mean", "std"],
        "mae": ["mean", "std"],
    }).round(4)
    
    # Find best parameter set
    best_idx = mean_performance[("r2", "mean")].idxmax()
    best_params = param_combinations[best_idx]
    best_r2 = mean_performance.loc[best_idx, ("r2", "mean")]
    best_mae = mean_performance.loc[best_idx, ("mae", "mean")]
    
    print(f"\nBest parameter set (index {best_idx}):")
    print(f"  R²:  {best_r2:.4f} ± {mean_performance.loc[best_idx, ('r2', 'std')]:.4f}")
    print(f"  MAE: {best_mae:.4f} ± {mean_performance.loc[best_idx, ('mae', 'std')]:.4f}")
    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    best_params_df = pd.DataFrame([best_params])
    best_params_df["best_r2"] = best_r2
    best_params_df["best_mae"] = best_mae
    best_params_df.to_csv(results_path / "best_parameters.csv", index=False)
    print(f"\nSaved best parameters to {results_path / 'best_parameters.csv'}")
    
    # Save summary statistics
    summary_df = mean_performance.reset_index()
    summary_df = summary_df.sort_values(("r2", "mean"), ascending=False)
    summary_path = results_path / "results_summary_by_params.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to {summary_path}")
    
    # Top 10 parameter sets
    print("\n" + "="*60)
    print("TOP 10 PARAMETER SETS (by R²)")
    print("="*60)
    top10 = summary_df.head(10)
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        param_idx = row["param_idx"]
        r2_mean = row[("r2", "mean")]
        mae_mean = row[("mae", "mean")]
        print(f"{i}. Param {param_idx}: R²={r2_mean:.4f}, MAE={mae_mean:.4f}")
    
    print("\nHyperparameter search completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for Virtual Catalyst Data"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV data file",
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Random seeds for experiments (default: [1,2,3,4,5])",
    )
    
    parser.add_argument(
        "--n_train_catalysts",
        type=int,
        default=200,
        help="Number of training catalysts (default: 200)",
    )
    
    parser.add_argument(
        "--n_test_catalysts",
        type=int,
        default=10,
        help="Number of test catalysts (default: 10)",
    )
    
    parser.add_argument(
        "--n_param_samples",
        type=int,
        default=50,
        help="Number of random parameter combinations to test (default: 50)",
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        default="hyperparameter_search",
        help="Output folder name (default: hyperparameter_search)",
    )
    
    parser.add_argument(
        "--fast_test",
        action="store_true",
        help="Run fast test with 10 param samples and 2 seeds",
    )
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        seeds=args.seeds,
        n_train_catalysts=args.n_train_catalysts,
        n_test_catalysts=args.n_test_catalysts,
        n_param_samples=args.n_param_samples,
        output_folder=args.output_folder,
        fast_test=args.fast_test,
    )
