import random
import argparse
from pathlib import Path
from joblib import dump
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from utils import (
    BASE_PROCESS,
    ATOM_NUMBERS,
    SUPPORT,
    DESCRIPTORS,
    CONDITIONS_TEMP_SINGLE,
    CONDITIONS_TEMP_PAIRS,
    CONDITIONS_CH4_O2_RATIO,
    get_cross_validation_param_sets,
    scale_data,
    split_data,
    augment_data,
    get_cross_validation_masks,
    settings_to_filename_map,
)


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_cols,
    random_state=random.Random(42),
    augment_data_flag=False,
    model_kwargs={},
    cross_val_masks=None,
    cross_val_params=None,
):
    # Combine train and validation sets for XGBoost training
    if X_val is not None and y_val is not None:
        if cross_val_masks is not None:
            raise ValueError(
                "Cannot provide validation set when using cross-validation."
            )
        X_train = np.vstack([X_train, X_val])
        y_train = np.hstack([y_train, y_val])
        X_val = None
        y_val = None

    # Data augmentation (permutations of M1, M2, M3 related features)
    if augment_data_flag:
        X_train, y_train, X_val, y_val, X_test, y_test, cross_val_masks = augment_data(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols,
            other_lists=cross_val_masks,
        )

    # Scaling
    X_train, X_val, X_test, scaler = scale_data(
        X_train,
        X_val,
        X_test,
        feature_cols,
        passthrough_cols=[],
    )

    model_seed = random_state.randint(0, 1000000)
    # run cross-validation if masks are provided to select best hyperparameters
    if cross_val_masks is not None:
        if cross_val_params is None or len(cross_val_params) == 0:
            raise ValueError(
                "cross_val_params must be provided when using cross-validation."
            )
        best_params = None
        best_cv_mse = float("inf")
        for params in cross_val_params:
            cv_mses = []
            for val_mask in cross_val_masks:
                # get train/val split for this fold
                train_mask = ~val_mask
                X_tr = X_train[train_mask]
                y_tr = y_train[train_mask]
                X_val_cv = X_train[val_mask]
                y_val_cv = y_train[val_mask]
                # initialize and train model
                model = xgb.XGBRegressor(random_state=model_seed, **params)
                model.fit(X_tr, y_tr)
                # evaluate on validation fold
                val_preds = model.predict(X_val_cv)
                val_mse = float(np.mean((val_preds - y_val_cv) ** 2))
                cv_mses.append(val_mse)
            # check if this param set is the best so far
            avg_cv_mse = np.mean(cv_mses)
            if avg_cv_mse < best_cv_mse:
                best_cv_mse = avg_cv_mse
                best_params = params
        print(f"Best CV MSE: {best_cv_mse:.4f} with params: {best_params}")
        # init model with best hyperparameters found for training on full train set
        model = xgb.XGBRegressor(random_state=model_seed, **best_params)
    else:
        model = xgb.XGBRegressor(random_state=model_seed, **model_kwargs)

    # model fitting
    model.fit(X_train, y_train)

    # evaluate on test set
    preds = model.predict(X_test)
    test_mse = float(np.mean((preds - y_test) ** 2))
    test_mae = float(np.mean(np.abs(preds - y_test)))
    test_rmse = float(np.sqrt(test_mse))
    ss_res = np.sum((preds - y_test) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    test_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

    results = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "history": [],
        "test_metrics": {
            "mse": test_mse,
            "mae": test_mae,
            "rmse": test_rmse,
            "r2": test_r2,
        },
        "preds_test": preds,
        "y_test": y_test,
    }
    return results


def plot_test_results(ax, y_true, y_pred, accumulate_permutations=False):
    if accumulate_permutations:
        # average predictions over permutations
        y_pred = y_pred.reshape(-1, 6).mean(axis=1)
        y_true = y_true[::6]
    # compute R2, MAE, RMSE
    r2 = 1.0 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    ax.scatter(y_true, y_pred, alpha=0.6)
    max_val = 22.5  # max(max(y_true), max(y_pred))
    min_val = -2.5  # min(min(y_true), min(y_pred))
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.plot([min_val, max_val], [min_val, max_val], "r--")  # y=x line
    # put metrics in the top-left inside the axes instead of using the title
    ax.text(
        0.02,
        0.98,
        f"R²={r2:.2f}, MAE={mae:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )
    # ax.set_title(f"R²={r2:.2f}, MAE={mae:.2f}")
    ax.grid(True)
    return r2, mae, mse, np.abs(y_pred - y_true)


def main(
    data_path,
    seeds,
    n_train_catalysts,
    n_test_catalysts,
    split_strategy,
    n_folds=5,
    cross_val_params=None,
    conditions={},
    folder_name=None,
    feature_sets=["base+atom_numbers+support", "base+descriptors", "all"],
    augmentations=[True, False],
    store_plots=False,
    store_models=False,
):
    # read data
    df = pd.read_csv(data_path)
    # storage path for results
    folder_name = f"xgb_cv_{cross_val_params}"
    results_path = Path(f"./{folder_name}/{n_train_catalysts}/")
    results_path.mkdir(parents=True, exist_ok=True)
    # collect results
    results_rows = []  # list of dicts for each experiment
    collected_models = {}  # dict of models
    collected_splits = {}  # dict of data splits
    for cond_name, conds in conditions.items():
        if cond_name != "":
            print(f"Running experiments for condition: {cond_name}...")
            _collected_models = {}  # dict of models per condition
            collected_models[cond_name] = _collected_models
            _collected_splits = {}  # dict of data splits per condition
            collected_splits[cond_name] = _collected_splits
        else:
            _collected_models = collected_models
            _collected_splits = collected_splits
        # setup composite figure for all seeds
        num_seeds = len(seeds)
        print(f"Running experiments with {num_seeds} seeds...")
        n_cols = len(feature_sets) * len(augmentations)
        figsize = (3.0 * n_cols, 3.0 * num_seeds)
        fig, axes = plt.subplots(
            nrows=num_seeds,
            ncols=n_cols,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )
        # ensure 2D array for consistent indexing
        if num_seeds == 1 and n_cols == 1:
            axes = np.array([[axes]])
        if num_seeds == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        # add column titles and axis labels
        col_titles = [
            "Atom Numbers+Aug",
            "Atom Numbers",
            "Descriptors+Aug",
            "Descriptors",
            "All+Aug",
            "All",
        ]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title)
        for ax in axes[:, 0]:
            ax.set_ylabel("Predicted C2y")
        for ax in axes[-1, :]:
            ax.set_xlabel("True C2y")
        for i, seed in enumerate(seeds):
            print(f"Running experiments for seed {seed}...")
            # load cross validation parameter sets if needed
            _cross_val_params = get_cross_validation_param_sets(
                cross_val_params, seed=seed
            )  # e.g., "best_cases", "avg_best", "random_30"
            for j, feature_set in enumerate(feature_sets):
                # run experiments per feature set
                if feature_set == "base+descriptors":
                    feature_cols = BASE_PROCESS + DESCRIPTORS
                elif feature_set == "base+atom_numbers+support":
                    feature_cols = BASE_PROCESS + ATOM_NUMBERS + SUPPORT
                elif feature_set == "all":  # all features
                    feature_cols = BASE_PROCESS + ATOM_NUMBERS + DESCRIPTORS + SUPPORT
                else:
                    raise ValueError(f"Invalid feature set: {feature_set}")
                target_col = "C2y"

                # initialize random seed (for data splitting, we want the same splits each run)
                rng = random.Random(seed)
                # also use the same random state for training xgboost with/without augmentation
                xgb_rng = random.Random(rng.randint(0, 1000000))

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
                    feature_cols,
                    target_col,
                    split_strategy,
                    rng,
                    n_train_catalysts=n_train_catalysts,
                    n_val_catalysts=0,  # we will not use a separate validation set since we do cross-validation on the training set
                    n_test_catalysts=n_test_catalysts,
                    return_indices=True,
                    conditions=conds,
                )
                # get cross-validation masks for training set if needed
                cross_val_masks = get_cross_validation_masks(
                    df,
                    train_indices,
                    split_strategy,
                    rng=rng,
                    n_folds=n_folds,
                    conditions=conds,
                )

                for k, augm in enumerate(augmentations):
                    # Train xgboost model
                    print(f"Training XGBoost model for feature set: {feature_set}...")
                    print(
                        f"Training {'with' if augm else 'without'} data augmentation..."
                    )
                    start_time = time()
                    xgb_results = train_xgboost(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                        feature_cols,
                        random_state=xgb_rng,
                        augment_data_flag=augm,
                        cross_val_masks=cross_val_masks,
                        cross_val_params=_cross_val_params,
                    )
                    elapsed_time = time() - start_time
                    print(f"Training completed in {elapsed_time:.2f} seconds.")
                    r2, mae, mse, absolute_errors = plot_test_results(
                        ax=axes[i, j * len(augmentations) + k],
                        y_true=xgb_results["y_test"],
                        y_pred=xgb_results["preds_test"],
                        accumulate_permutations=augm,
                    )
                    # compute MAE per catalyst and store in results
                    test_df = df.iloc[test_indices].reset_index(drop=True)
                    test_df["absolute_error"] = absolute_errors
                    mae_by_catalyst = test_df.groupby("Name")["absolute_error"].mean()
                    results_rows.append(
                        {
                            "model_type": "xgboost",
                            "feature_set": feature_set,
                            "augmentation": "yes" if augm else "no",
                            "n_train_catalysts": n_train_catalysts,
                            "n_val_catalysts": 0,
                            "n_test_catalysts": n_test_catalysts,
                            "seed": seed,
                            "r2": r2,
                            "mae": mae,
                            "mse": mse,
                            "condition": cond_name,
                            # merged per-catalyst entries
                            **{
                                f"test_catalyst_{i}": name
                                for i, name in enumerate(mae_by_catalyst.index)
                            },
                            **{
                                f"mae_test_catalyst_{i}": mae_val
                                for i, mae_val in enumerate(mae_by_catalyst)
                            },
                            "training_time": elapsed_time,
                        }
                    )
                    # collect model and splits
                    model_id = (
                        f"{settings_to_filename_map[(feature_set, augm)]}_{seed:04d}"
                    )
                    _collected_models[model_id] = xgb_results["model"]
                    if seed not in _collected_splits:
                        _collected_splits[seed] = {
                            "train_indices": train_indices,
                            "val_indices": val_indices,
                            "test_indices": test_indices,
                        }

        # save figure
        if store_plots:
            fig.savefig(
                results_path / f"xgb_test_results_{seeds[0]}-{seeds[-1]}_{cond_name}.png"
            )
    # save results dataframe to csv
    results_df = pd.DataFrame(results_rows)
    summary_path = results_path / "results_summary.csv"
    if summary_path.exists():
        results_df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(summary_path, index=False)
    # save models to disk
    models_path = results_path / f"xgb_models_{seeds[0]}-{seeds[-1]}.joblib"
    if store_models:
        dump(collected_models, models_path)
    # save data splits to disk
    splits_path = results_path / f"data_splits_{seeds[0]}-{seeds[-1]}.joblib"
    dump(collected_splits, splits_path)
    print("All experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training/evaluation")
    # add argument for datapath
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the input data file",
    )
    # add argument for seeds: accepts space separated ints or range syntax like 1:100 or 1:10:2
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=["1"],
        help="Seeds for running the experiments. Accepts ints (e.g. --seeds 1 2 3) or ranges (e.g. --seeds 1:100 or 1:10:2)",
    )
    # add arguments for n_train_catalysts, n_test_catalysts
    parser.add_argument(
        "--n_train_catalysts",
        type=int,
        default=49,
        help="Number of training catalysts (default: 49)",
    )
    parser.add_argument(
        "--n_test_catalysts",
        type=int,
        default=10,
        help="Number of test catalysts (default: 10)",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="catalyst",
        help="Data split strategy: 'catalyst', or 'catalyst_random' (default: 'catalyst')",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--cross_val_params",
        type=str,
        default="random_50",
        help="Parameter set for cross-validation: 'random_x' where x is the number of random samples from all combinations (default: 'random_30')",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="None",
        help="Conditions to include in the model. Choose from 'None', 'temp_single', 'temp_pairs', 'ch4_o2_ratio'.",
    )
    parser.add_argument(
        "--feature_sets",
        type=str,
        nargs="+",
        default=["base+atom_numbers+support", "base+descriptors", "all"],
        help="Feature sets to run (default: all three sets)",
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        default=["yes", "no"],
        help="Augmentation options to run (default: both yes and no)",
    )
    parser.add_argument(
        "--store_plots",
        action="store_true",
        help="Whether to store the plots (default: False)",
    )
    parser.add_argument(
        "--store_models",
        action="store_true",
        help="Whether to store the models (default: False)",
    )

    def parse_seeds(tokens):
        out = []
        for t in tokens:
            if ":" in t:
                parts = t.split(":")
                if len(parts) not in (2, 3):
                    raise ValueError(f"Invalid seed range: {t}")
                start = int(parts[0])
                stop = int(parts[1])
                step = int(parts[2]) if len(parts) == 3 else 1
                if step == 0:
                    raise ValueError("step in seed range cannot be 0")
                if start <= stop and step > 0:
                    rng = range(start, stop + 1, step)
                elif start >= stop and step > 0:
                    rng = range(start, stop - 1, -step)
                else:
                    rng = range(start, stop + (1 if step > 0 else -1), step)
                out.extend(list(rng))
            else:
                out.append(int(t))
        # remove duplicates while preserving order
        return list(dict.fromkeys(out))

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if args.split_strategy != "catalyst" and args.conditions != "None":
        raise ValueError("Conditions can only be used with 'catalyst' split strategy.")
    # set conditions based on argument
    if args.conditions == "None":
        conditions = {"": {}}  # no conditions
    elif args.conditions == "temp_single":
        conditions = CONDITIONS_TEMP_SINGLE
    elif args.conditions == "temp_pairs":
        conditions = CONDITIONS_TEMP_PAIRS
    elif args.conditions == "ch4_o2_ratio":
        conditions = CONDITIONS_CH4_O2_RATIO
    else:
        raise ValueError(f"Invalid conditions argument: {args.conditions}")

    augmentations = [augm.lower() in ["yes", "1"] for augm in args.augmentations]
    main(
        data_path=args.data_path,
        seeds=seeds,
        n_train_catalysts=args.n_train_catalysts,
        n_test_catalysts=args.n_test_catalysts,
        split_strategy=args.split_strategy,
        n_folds=args.n_folds,
        cross_val_params=args.cross_val_params,
        conditions=conditions,
        feature_sets=args.feature_sets,
        augmentations=augmentations,
        store_plots=args.store_plots,
        store_models=args.store_models,
    )
