"""
Baseline Random Split Study for Virtual Catalyst Data
======================================================
Simple 80-20 random split of all data points as a baseline comparison.
This ignores catalyst structure and treats all data points independently.
"""

import random
import argparse
from pathlib import Path
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_virtual_catalyst import (
    ALL_FEATURES,
    TARGET,
    optimized_xgb,
    default_xgb,
    split_data_random,
    scale_data,
    compute_metrics,
)
import xgboost as xgb


def train_and_evaluate(
    X_train, y_train,
    X_test, y_test,
    feature_cols,
    random_state,
    model_type="optimized",
):
    """Train XGBoost and evaluate"""
    # Scale data
    X_train, _, X_test, scaler = scale_data(
        X_train, None, X_test, feature_cols
    )
    
    # Initialize model
    model_seed = random_state.randint(0, 1000000)
    
    if model_type == "optimized":
        model = optimized_xgb(seed=model_seed)
    elif model_type == "default":
        model = default_xgb(seed=model_seed)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, preds)
    
    return model, preds, metrics


def main(
    data_path,
    seeds,
    train_fraction=0.8,
    model_type="optimized",
    output_folder="baseline_random_split",
):
    """
    Run baseline random split experiments.
    
    Args:
        data_path: Path to CSV data
        seeds: List of random seeds
        train_fraction: Fraction of data for training (default 0.8)
        model_type: Model configuration
        output_folder: Output directory
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Using {train_fraction*100:.0f}% for training, {(1-train_fraction)*100:.0f}% for testing")
    
    # Create results directory
    results_path = Path(f"./{output_folder}/")
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Feature columns
    feature_cols = ALL_FEATURES
    target_col = TARGET
    
    # Storage for results
    results_rows = []
    collected_models = {}
    
    # Create figure for plotting
    n_seeds = len(seeds)
    n_cols = min(5, n_seeds)
    n_rows = (n_seeds + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False
    )
    axes = axes.flatten()
    
    print(f"\nRunning {n_seeds} experiments with random 80-20 splits...")
    
    # Loop over seeds
    for i, seed in enumerate(seeds):
        print(f"\nSeed {seed} ({i+1}/{n_seeds})...")
        rng = random.Random(seed)
        xgb_rng = random.Random(rng.randint(0, 1000000))
        
        # Split data randomly
        X_train, y_train, X_test, y_test, train_idx, test_idx = split_data_random(
            df, feature_cols, target_col, rng, train_fraction=train_fraction
        )
        
        print(f"  Train: {len(y_train)} samples, Test: {len(y_test)} samples")
        
        # Train and evaluate
        model, preds, metrics = train_and_evaluate(
            X_train, y_train,
            X_test, y_test,
            feature_cols,
            random_state=xgb_rng,
            model_type=model_type,
        )
        
        print(f"  R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")
        
        # Plot
        ax = axes[i]
        ax.scatter(y_test, preds, alpha=0.6, s=20)
        
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        margin = (max_val - min_val) * 0.05
        
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        
        ax.text(
            0.02, 0.98,
            f"R²={metrics['r2']:.3f}\nMAE={metrics['mae']:.3f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("True C2y")
        ax.set_ylabel("Predicted C2y")
        ax.grid(True, alpha=0.3)
        
        # Store results
        results_rows.append({
            "seed": seed,
            "train_fraction": train_fraction,
            "n_train_samples": len(y_train),
            "n_test_samples": len(y_test),
            "r2": metrics["r2"],
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "model_type": model_type,
        })
        
        # Save model
        model_id = f"seed_{seed:04d}"
        collected_models[model_id] = model
    
    # Hide unused subplots
    for i in range(n_seeds, len(axes)):
        axes[i].set_visible(False)
    
    # Save figure
    plt.tight_layout()
    fig_path = results_path / f"baseline_random_split_seeds{seeds[0]}-{seeds[-1]}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {fig_path}")
    plt.close()
    
    # Save results
    results_df = pd.DataFrame(results_rows)
    csv_path = results_path / "results_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Save models
    models_path = results_path / f"models_seeds{seeds[0]}-{seeds[-1]}.joblib"
    dump(collected_models, models_path)
    print(f"Saved {len(collected_models)} models to {models_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Mean R²:   {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
    print(f"Mean MAE:  {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}")
    print(f"Mean RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}")
    print("="*60)
    
    print("\nBaseline experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline Random Split Study for Virtual Catalyst Data"
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
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="optimized",
        choices=["optimized", "default"],
        help="Model configuration to use (default: optimized)",
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        default="baseline_random_split",
        help="Output folder name (default: baseline_random_split)",
    )
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        seeds=args.seeds,
        train_fraction=args.train_fraction,
        model_type=args.model_type,
        output_folder=args.output_folder,
    )
