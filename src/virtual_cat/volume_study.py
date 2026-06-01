"""
Dataset Volume Study for Virtual Catalyst Data
===============================================
Studies the effect of training set size on model performance.

Training set sizes: 1, 3, 5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 300, max
Test set: Fixed 10 catalysts
Validation set: None (combined with training)

Fast test mode: Run with --fast_test to use only 2 seeds and 2 training sizes
"""

import random
import argparse
from pathlib import Path
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from utils_virtual_catalyst import (
    PROCESS_CONDITIONS,
    DESCRIPTORS,
    ALL_FEATURES,
    TARGET,
    CATALYST_ID,
    optimized_xgb,
    default_xgb,
    get_random_xgb_params,
    split_data_catalyst,
    scale_data,
    compute_metrics,
)


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_cols,
    random_state,
    model_type="optimized",
    model_kwargs=None,
):
    """
    Train XGBoost model and evaluate on test set.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (can be None)
        X_test, y_test: Test data
        feature_cols: List of feature names
        random_state: Random state object
        model_type: "optimized", "default", or "custom"
        model_kwargs: Custom hyperparameters if model_type="custom"
    
    Returns:
        Dict with model, scaler, metrics, and predictions
    """
    # Combine train and validation sets for XGBoost training
    if X_val is not None and y_val is not None:
        X_train = np.vstack([X_train, X_val])
        y_train = np.hstack([y_train, y_val])
        X_val = None
        y_val = None
    
    # Scaling
    X_train, X_val, X_test, scaler = scale_data(
        X_train, X_val, X_test, feature_cols
    )
    
    # Initialize model
    model_seed = random_state.randint(0, 1000000)
    
    if model_type == "optimized":
        model = optimized_xgb(seed=model_seed)
    elif model_type == "default":
        model = default_xgb(seed=model_seed)
    elif model_type == "custom":
        if model_kwargs is None:
            raise ValueError("model_kwargs required for custom model type")
        model = xgb.XGBRegressor(random_state=model_seed, **model_kwargs)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    preds = model.predict(X_test)
    test_metrics = compute_metrics(y_test, preds)
    
    print(f"Test R²: {test_metrics['r2']:.4f}, MAE: {test_metrics['mae']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
    
    results = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "test_metrics": test_metrics,
        "preds_test": preds,
        "y_test": y_test,
    }
    
    return results


def plot_test_results(ax, y_true, y_pred):
    """Create scatter plot of predictions vs true values"""
    metrics = compute_metrics(y_true, y_pred)
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Set axis limits based on data
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    
    # Add metrics text
    ax.text(
        0.02, 0.98,
        f"R²={metrics['r2']:.3f}\nMAE={metrics['mae']:.3f}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    
    ax.set_xlabel("True C2y")
    ax.set_ylabel("Predicted C2y")
    ax.grid(True, alpha=0.3)
    
    return metrics


def main(
    data_path,
    seeds,
    train_catalyst_sizes,
    n_test_catalysts=10,
    model_type="optimized",
    output_folder="volume_study",
    fast_test=False,
):
    """
    Main function to run dataset volume study.
    
    Args:
        data_path: Path to CSV data file
        seeds: List of random seeds
        train_catalyst_sizes: List of training set sizes to test
        n_test_catalysts: Number of catalysts for test set
        model_type: "optimized", "default", or "custom"
        output_folder: Folder name for saving results
        fast_test: If True, run quick test with limited parameters
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Unique catalysts: {df[CATALYST_ID].nunique()}")
    print(f"Target variable: {TARGET}")
    
    # Create results directory
    results_path = Path(f"./{output_folder}/")
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Feature set: use all features
    feature_cols = ALL_FEATURES
    target_col = TARGET
    
    print(f"\nFeature set: {len(feature_cols)} features")
    print(f"Process conditions: {PROCESS_CONDITIONS}")
    print(f"Descriptors: {len(DESCRIPTORS)} descriptors (D1-D{len(DESCRIPTORS)})")
    
    # Storage for results
    results_rows = []
    collected_models = {}
    
    # Create figure for plotting (one row per seed, one column per training size)
    n_seeds = len(seeds)
    n_sizes = len(train_catalyst_sizes)
    
    fig, axes = plt.subplots(
        n_seeds, n_sizes, 
        figsize=(4 * n_sizes, 4 * n_seeds),
        squeeze=False
    )
    
    print(f"\n{'='*60}")
    print(f"Running experiments: {n_seeds} seeds × {n_sizes} training sizes")
    print(f"Training sizes: {train_catalyst_sizes}")
    print(f"Test catalysts: {n_test_catalysts}")
    print(f"{'='*60}\n")
    
    # Loop over seeds
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")
        rng = random.Random(seed)
        xgb_rng = random.Random(rng.randint(0, 1000000))
        
        # Loop over training set sizes
        for j, n_train in enumerate(train_catalyst_sizes):
            print(f"\n  Training with {n_train} catalysts ({j+1}/{n_sizes})...")
            
            # Split data
            try:
                X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx = split_data_catalyst(
                    df,
                    feature_cols,
                    target_col,
                    rng,
                    n_train_catalysts=n_train,
                    n_val_catalysts=0,  # No validation set
                    n_test_catalysts=n_test_catalysts,
                )
            except Exception as e:
                print(f"  ERROR: Could not split data: {e}")
                continue
            
            print(f"  Train: {len(y_train)} samples, Test: {len(y_test)} samples")
            
            # Train model
            try:
                xgb_results = train_xgboost(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    feature_cols,
                    random_state=xgb_rng,
                    model_type=model_type,
                )
            except Exception as e:
                print(f"  ERROR: Training failed: {e}")
                continue
            
            # Plot results
            metrics = plot_test_results(
                ax=axes[i, j],
                y_true=xgb_results["y_test"],
                y_pred=xgb_results["preds_test"],
            )
            axes[i, j].set_title(f"Seed={seed}, N_train={n_train}")
            
            # Store results
            results_rows.append({
                "seed": seed,
                "n_train_catalysts": n_train,
                "n_test_catalysts": n_test_catalysts,
                "n_train_samples": len(y_train),
                "n_test_samples": len(y_test),
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "model_type": model_type,
            })
            
            # Save model
            model_id = f"n{n_train:04d}_s{seed:04d}"
            collected_models[model_id] = xgb_results["model"]
    
    # Save figure
    plt.tight_layout()
    fig_path = results_path / f"volume_study_seeds{seeds[0]}-{seeds[-1]}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {fig_path}")
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame(results_rows)
    csv_path = results_path / "results_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Save models
    models_path = results_path / f"models_seeds{seeds[0]}-{seeds[-1]}.joblib"
    dump(collected_models, models_path)
    print(f"Saved {len(collected_models)} models to {models_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    summary = results_df.groupby("n_train_catalysts").agg({
        "r2": ["mean", "std"],
        "mae": ["mean", "std"],
    }).round(4)
    print(summary)
    
    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset Volume Study for Virtual Catalyst Data"
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
        "--train_sizes",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 300],
        help="Training set sizes to test (default: [1,3,5,10,20,30,40,50,70,100,150,200,300])",
    )
    
    parser.add_argument(
        "--n_test_catalysts",
        type=int,
        default=10,
        help="Number of test catalysts (default: 10)",
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
        default="volume_study",
        help="Output folder name (default: volume_study)",
    )
    
    parser.add_argument(
        "--fast_test",
        action="store_true",
        help="Run fast test with limited parameters (2 seeds, 3 training sizes)",
    )
    
    args = parser.parse_args()
    
    # Adjust parameters for fast test
    if args.fast_test:
        print("\n" + "="*60)
        print("FAST TEST MODE ENABLED")
        print("="*60)
        seeds = [1, 2]
        train_sizes = [5, 20, 50]
        output_folder = args.output_folder + "_fast_test"
        print(f"Using seeds: {seeds}")
        print(f"Using train sizes: {train_sizes}")
        print(f"Output folder: {output_folder}")
        print("="*60 + "\n")
    else:
        seeds = args.seeds
        train_sizes = args.train_sizes
        output_folder = args.output_folder
    
    # Run the study
    main(
        data_path=args.data_path,
        seeds=seeds,
        train_catalyst_sizes=train_sizes,
        n_test_catalysts=args.n_test_catalysts,
        model_type=args.model_type,
        output_folder=output_folder,
        fast_test=args.fast_test,
    )
