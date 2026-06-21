"""
================================================================================
XGBOOST TRAINING FOR VIRTUAL CATALYST DATASETS
With 5-fold cross-validation, SHAP analysis, and comprehensive evaluation
================================================================================
Author: Parastoo
Date: 2026-02-02
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap
import warnings
import random
from datetime import datetime

from utils import (
    get_cross_validation_param_sets,
    split_data,
    scale_data,
    get_cross_validation_masks,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURATION
# ============================================================================

CATALYST_ID = "cat_ID"
TARGET = "C2y"
DESCRIPTORS = [f"D{i}" for i in range(1, 17)]
PROCESS_CONDITIONS = ["T", "Q", "CH4_O2", "InertFraction"]
ALL_FEATURES = DESCRIPTORS + PROCESS_CONDITIONS

# Fixed experiment parameters
N_TEST_CATALYSTS = 10
N_CV_FOLDS = 5

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def cross_validate_params(X_train, y_train, params, cross_val_masks, model_seed):
    """
    Perform k-fold cross-validation for given parameters.

    Args:
        X_train: Training features
        y_train: Training targets
        params: Hyperparameter dictionary
        cross_val_masks: List of validation masks for cross-validation (entries belonging to the validation split are marked as True)
        model_seed: Random seed for model initialization

    Returns:
        Mean MAE across folds
    """
    fold_maes = []
    
    for val_mask in cross_val_masks:
        train_mask = ~val_mask
        X_tr, X_val = X_train[train_mask], X_train[val_mask]
        y_tr, y_val = y_train[train_mask], y_train[val_mask]

        model = XGBRegressor(
            random_state=model_seed,
            **params,
        )

        model.fit(X_tr, y_tr, verbose=False)
        y_val_pred = model.predict(X_val)
        fold_mae = mean_absolute_error(y_val*100, y_val_pred*100)  # Convert to percentage for MAE
        fold_maes.append(fold_mae)

    return np.mean(fold_maes)


def train_and_evaluate_xgb(X_train, y_train, X_test, y_test, params, seed):
    """
    Train XGBoost model on full training set and evaluate on test set.

    Returns:
        Dictionary with model, predictions and metrics
    """
    model = XGBRegressor(
        random_state=seed,
        **params,
    )

    # Train on full training set
    model.fit(X_train, y_train, verbose=False)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    results = {
        "model": model,
        "train_r2": r2_score(y_train, y_train_pred),
        "train_mae": mean_absolute_error(y_train*100, y_train_pred*100),  # Convert to percentage for MAE
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_r2": r2_score(y_test, y_test_pred),
        "test_mae": mean_absolute_error(y_test*100, y_test_pred*100),  # Convert to percentage for MAE
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }

    return results


def compute_shap_values(model, X_test, feature_cols):
    """
    Compute SHAP values for the full test set.

    Args:
        model: Trained XGBoost model
        X_test: Test features (full set)
        feature_cols: List of feature names

    Returns:
        Mean absolute SHAP values per feature
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test, check_additivity=False)

    # Extract values (compatibility with different SHAP versions)
    values = shap_values.values if hasattr(shap_values, "values") else shap_values

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.mean(np.abs(values), axis=0)

    return shap_values, mean_abs_shap


def plot_shap_analysis(shap_values, X_test, feature_cols, output_dir, top_k=5):
    """
    Create SHAP visualizations.

    Args:
        shap_values: SHAP values from TreeExplainer
        X_test: Test features
        feature_cols: List of feature names
        output_dir: Directory to save plots
        top_k: Number of top features to show in dependence plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary bar plot (feature importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Beeswarm plot (detailed feature effects)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_cols, show=False, max_display=20
    )
    plt.title("SHAP Feature Effects", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Dependence plots for top features
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    mean_abs_shap = np.mean(np.abs(values), axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-top_k:][::-1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_features_idx):
        if i >= 6:  # Only plot top 6
            break
        shap.dependence_plot(
            feat_idx, values, X_test, feature_names=feature_cols, ax=axes[i], show=False
        )
        axes[i].set_title(f"{feature_cols[feat_idx]}", fontsize=12, fontweight="bold")

    # Hide unused subplots
    for i in range(len(top_features_idx), 6):
        axes[i].axis("off")

    plt.suptitle(
        f"SHAP Dependence Plots - Top {top_k} Features",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_dependence_top_features.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"  ✓ SHAP plots saved to {output_dir}")


def plot_performance(results_list, output_dir):
    """
    Create performance visualization plots.

    Args:
        results_list: List of result dictionaries from all seeds
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics
    seeds = [r["seed"] for r in results_list]
    test_r2 = [r["test_r2"] for r in results_list]
    test_mae = [r["test_mae"] for r in results_list]
    test_rmse = [r["test_rmse"] for r in results_list]

    cv_mae = [r["cv_mae_mean"] for r in results_list]

    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: R² across seeds
    ax1 = axes[0, 0]
    ax1.plot(
        seeds,
        test_r2,
        "o-",
        linewidth=2,
        markersize=10,
        color="steelblue",
        label="Test R²",
    )
    ax1.axhline(
        np.mean(test_r2),
        color="steelblue",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"Mean Test R²={np.mean(test_r2):.4f}",
    )
    ax1.set_xlabel("Seed", fontweight="bold", fontsize=12)
    ax1.set_ylabel("R²", fontweight="bold", fontsize=12)
    ax1.set_title("R² Performance Across Seeds", fontweight="bold", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: MAE across seeds (Test vs CV)
    ax2 = axes[0, 1]
    ax2.plot(
        seeds,
        test_mae,
        "o-",
        linewidth=2,
        markersize=10,
        color="green",
        label="Test MAE",
    )
    ax2.plot(
        seeds,
        cv_mae,
        "s-",
        linewidth=2,
        markersize=8,
        color="orange",
        label="CV MAE",
        alpha=0.7,
    )
    ax2.axhline(
        np.mean(test_mae),
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"Mean Test MAE={np.mean(test_mae):.4f}",
    )
    ax2.set_xlabel("Seed", fontweight="bold", fontsize=12)
    ax2.set_ylabel("MAE", fontweight="bold", fontsize=12)
    ax2.set_title("MAE Performance Across Seeds", fontweight="bold", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Plot 3: Box plots for metrics
    ax3 = axes[1, 0]
    box_data = [test_r2, test_mae, test_rmse]
    positions = [1, 2, 3]
    bp = ax3.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        labels=["R²", "MAE", "RMSE"],
    )
    for patch, color in zip(bp["boxes"], ["steelblue", "green", "coral"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel("Metric Value", fontweight="bold", fontsize=12)
    ax3.set_title("Test Metrics Distribution", fontweight="bold", fontsize=14)
    ax3.grid(alpha=0.3, axis="y")

    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis("off")

    summary_text = f"""
PERFORMANCE SUMMARY

Cross-Validation: {N_CV_FOLDS}-Fold

Test Set Metrics (n={len(results_list)} seeds):
  R²:
    Mean:  {np.mean(test_r2):.4f}
    Std:   {np.std(test_r2):.4f}
    Min:   {np.min(test_r2):.4f}
    Max:   {np.max(test_r2):.4f}
    
  MAE:
    Mean:  {np.mean(test_mae):.4f}
    Std:   {np.std(test_mae):.4f}
    Min:   {np.min(test_mae):.4f}
    Max:   {np.max(test_mae):.4f}

Best Test R²:  {np.max(test_r2):.4f} (seed {seeds[np.argmax(test_r2)]})
Best Test MAE: {np.min(test_mae):.4f} (seed {seeds[np.argmin(test_mae)]})
"""

    ax4.text(
        0.1,
        0.5,
        summary_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(
        f"XGBoost Performance Analysis ({N_CV_FOLDS}-Fold CV)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "performance_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Performance plots saved to {output_dir}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def main(
    data_path,
    n_train_catalysts,
    seeds,
    compute_shap=True,
    plot_performance=True,
):
    """
    Main training function for virtual catalyst dataset.

    Args:
        data_path: Path to dataset CSV
        output_dir: Output directory for results
        n_train_catalysts: Number of training catalysts
        seeds: List of random seeds
        compute_shap: Whether to compute SHAP values
        plot_performance: Whether to create performance visualization plots
    """
    print("=" * 80)
    print("XGBOOST TRAINING FOR VIRTUAL CATALYST DATASET")
    print(f"WITH {N_CV_FOLDS}-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    folder_name = f"virtual_xgb_random_50"
    output_dir = Path(f"./{folder_name}/{n_train_catalysts}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  Training catalysts: {n_train_catalysts}")
    print(f"  Test catalysts: {N_TEST_CATALYSTS}")
    print(f"  CV folds: {N_CV_FOLDS}")
    print(f"  Seeds: {seeds}")
    print(f"  Compute SHAP: {compute_shap}")
    print(f"  Plot Performance: {plot_performance}")

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(data_path)
    print(f"\nDataset:")
    print(f"  Samples: {len(df):,}")
    print(f"  Catalysts: {df[CATALYST_ID].nunique()}")
    print(f"  Features: {len(ALL_FEATURES)}")
    print(f"  Target: {TARGET}")

    # Run experiments for each seed
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)

    all_results = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*80}")

        print("\n1. Splitting data...")
        rng = random.Random(seed)
        model_seed = rng.randint(0, 1000000)
        # Data splitting
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            train_indices,
            val_indices,
            test_indices,
        ) = split_data(
            df,
            feature_cols=ALL_FEATURES,
            target_col=TARGET,
            split_strategy="catalyst",
            rng=rng,
            n_train_catalysts=n_train_catalysts,
            n_val_catalysts=0,
            n_test_catalysts=N_TEST_CATALYSTS,
            return_indices=True,
            catalyst_name_column=CATALYST_ID,
        )
        # get cross-validation masks for training set
        cross_val_masks = get_cross_validation_masks(
            df,
            train_indices,
            split_strategy="catalyst",
            rng=rng,
            n_folds=N_CV_FOLDS,
            catalyst_name_column=CATALYST_ID,
        )
        # Split data (no validation set)
        train_cats = df.loc[train_indices, CATALYST_ID].unique()
        test_cats = df.loc[test_indices, CATALYST_ID].unique()

        print(f"  Train: {len(X_train):,} samples from {len(train_cats)} catalysts")
        print(f"  Test:  {len(X_test):,} samples from {len(test_cats)} catalysts")

        # Scale data
        print("\n2. Scaling data...")
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(
            X_train,
            X_val,
            X_test,
            feature_cols=ALL_FEATURES,
            passthrough_cols=[],
        )

        # Hyperparameter search with 5-fold CV
        # Draw 50 random combinations from the grid
        xgb_param_combinations = get_cross_validation_param_sets("random_50", seed=seed)
        print(
            f"\n3. Hyperparameter search with {N_CV_FOLDS}-fold CV ({len(xgb_param_combinations)} randomly drawn combinations)..."
        )

        best_cv_mae = float("inf")
        best_params = None

        for idx, params in enumerate(xgb_param_combinations):
            if (idx + 1) % 10 == 0:
                print(f"  Tested {idx+1}/{len(xgb_param_combinations)} combinations...")

            # Cross-validate
            cv_mae = cross_validate_params(
                X_train_scaled, y_train, params, cross_val_masks, model_seed,
            )

            if cv_mae < best_cv_mae:
                best_cv_mae = cv_mae
                best_params = params

        print(f"\n  ✓ Best CV MAE: {best_cv_mae:.4f}")
        print(f"  ✓ Best parameters: {best_params}")

        # Retrain on full training set with best parameters
        print(f"\n4. Retraining on full training set with best parameters...")
        best_model_results = train_and_evaluate_xgb(
            X_train_scaled, y_train, X_test_scaled, y_test, best_params, model_seed
        )

        print(f"  ✓ Train R² = {best_model_results['train_r2']:.4f}")
        print(f"  ✓ Test R² = {best_model_results['test_r2']:.4f}")
        print(f"  ✓ Test MAE = {best_model_results['test_mae']:.4f}")

        # Store results
        result_dict = {
            "seed": seed,
            "n_train_catalysts": n_train_catalysts,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "cv_mae_mean": best_cv_mae,
            "train_r2": best_model_results["train_r2"],
            "train_mae": best_model_results["train_mae"],
            "train_rmse": best_model_results["train_rmse"],
            "test_r2": best_model_results["test_r2"],
            "test_mae": best_model_results["test_mae"],
            "test_rmse": best_model_results["test_rmse"],
            **{f"hp_{k}": v for k, v in best_params.items()},
        }

        # SHAP analysis
        if compute_shap:
            print(
                f"\n5. Computing SHAP values on full test set ({len(X_test)} samples)..."
            )
            shap_values, mean_abs_shap = compute_shap_values(
                best_model_results["model"], X_test_scaled, ALL_FEATURES
            )

            # Add mean SHAP values to results
            for feat, shap_val in zip(ALL_FEATURES, mean_abs_shap):
                result_dict[f"shap_{feat}"] = shap_val

            # Create SHAP plots (only for first seed to save time)
            if seed_idx == 0:
                print("  Creating SHAP visualizations...")
                plot_shap_analysis(
                    shap_values,
                    X_test_scaled,
                    ALL_FEATURES,
                    output_dir / "shap_plots",
                    top_k=5,
                )

        all_results.append(result_dict)

        print(f"\n✓ Seed {seed} complete")
        print(f"  CV MAE = {result_dict['cv_mae_mean']:.4f}")
        print(f"  Test R² = {result_dict['test_r2']:.4f}")
        print(f"  Test MAE = {result_dict['test_mae']:.4f}")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_df = pd.DataFrame(all_results)
    results_path = output_dir / "training_results.csv"
    if results_path.exists():
        results_df.to_csv(results_path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    # Create performance plots
    if plot_performance:
        print("\nCreating performance visualizations...")
        plot_performance(all_results, output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(f"\nDataset: {Path(data_path).name}")
    print(f"Training catalysts: {n_train_catalysts}")
    print(f"Seeds tested: {len(seeds)}")
    print(f"CV folds: {N_CV_FOLDS}")

    print(f"\n📊 FINAL RESULTS:")
    print(
        f"  CV MAE:    {results_df['cv_mae_mean'].mean():.4f} ± {results_df['cv_mae_mean'].std():.4f}"
    )
    print(
        f"  Test R²:   {results_df['test_r2'].mean():.4f} ± {results_df['test_r2'].std():.4f}"
    )
    print(
        f"  Test MAE:  {results_df['test_mae'].mean():.4f} ± {results_df['test_mae'].std():.4f}"
    )

    print(f"\n✓ All outputs saved to: {output_dir}")
    print("=" * 80)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost on virtual catalyst dataset with 5-fold CV hyperparameter tuning"
    )

    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset CSV file"
    )

    parser.add_argument(
        "--n_train_catalysts",
        type=int,
        required=True,
        help="Number of training catalysts",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of random seeds (default: [1, 2, 3, 4, 5])",
    )

    parser.add_argument(
        "--compute_shap",
        action="store_true",
        help="Compute SHAP values (adds significant time)",
    )

    parser.add_argument(
        "--plot_performance",
        action="store_true",
        help="Create performance visualization plots",
    )

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        n_train_catalysts=args.n_train_catalysts,
        seeds=args.seeds,
        compute_shap=args.compute_shap,
        plot_performance=args.plot_performance,
    )
