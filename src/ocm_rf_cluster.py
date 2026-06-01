import random
import argparse
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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


def train_random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_cols,
    random_state=random.Random(42),
    augment_data_flag=False,
    model_type="default",
    model_kwargs={},
    cross_val_masks=None,
    cross_val_params=None,
):
    # Combine train and validation sets for training
    if X_val is not None and y_val is not None:
        if cross_val_masks is not None:
            raise ValueError(
                "Cannot provide validation set when using cross-validation."
            )
        X_train = np.vstack([X_train, X_val])
        y_train = np.hstack([y_train, y_val])
        X_val = None
        y_val = None

    # Data augmentation
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

    # Cross-validation over provided parameter sets (if any)
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
                train_mask = ~val_mask
                X_tr = X_train[train_mask]
                y_tr = y_train[train_mask]
                X_val_cv = X_train[val_mask]
                y_val_cv = y_train[val_mask]
                # initialize and train model
                model = RandomForestRegressor(random_state=model_seed, **params)
                model.fit(X_tr, y_tr)
                val_preds = model.predict(X_val_cv)
                val_mse = float(np.mean((val_preds - y_val_cv) ** 2))
                cv_mses.append(val_mse)
            avg_cv_mse = np.mean(cv_mses)
            if avg_cv_mse < best_cv_mse:
                best_cv_mse = avg_cv_mse
                best_params = params
        print(f"Best CV MSE: {best_cv_mse:.4f} with params: {best_params}")
        model = RandomForestRegressor(random_state=model_seed, **best_params)
    else:
        # fixed model selection
        if model_type == "default":
            model = RandomForestRegressor(n_estimators=100, random_state=model_seed)
        elif model_type == "custom":
            model = RandomForestRegressor(random_state=model_seed, **model_kwargs)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

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


def main(
    data_path,
    seeds,
    n_train_catalysts,
    n_val_catalysts,
    n_test_catalysts,
    split_strategy,
    model_type="default",
    cross_validation=False,
    n_folds=5,
    cross_val_params=None,
    conditions={},
    folder_name=None,
    feature_sets=["base+atom_numbers+support", "base+descriptors", "all"],
    augmentations=[True, False],
):
    df = pd.read_csv(data_path)
    if cross_validation:
        folder_name = f"rf_cv_{cross_val_params}"
    else:
        folder_name = f"rf_{model_type}"
    results_path = Path(f"./{folder_name}/{n_train_catalysts+n_val_catalysts}/")
    results_path.mkdir(parents=True, exist_ok=True)
    results_rows = []
    collected_models = {}
    collected_splits = {}
    for cond_name, conds in conditions.items():
        if cond_name != "":
            print(f"Running experiments for condition: {cond_name}...")
            _collected_models = {}
            collected_models[cond_name] = _collected_models
            _collected_splits = {}
            collected_splits[cond_name] = _collected_splits
        else:
            _collected_models = collected_models
            _collected_splits = collected_splits
        num_seeds = len(seeds)
        print(f"Running experiments with {num_seeds} seeds...")
        for i, seed in enumerate(seeds):
            print(f"Running experiments for seed {seed}...")
            if cross_validation:
                _cross_val_params = get_cross_validation_param_sets(
                    cross_val_params, seed=seed
                )
            else:
                _cross_val_params = None
            for j, feature_set in enumerate(feature_sets):
                if feature_set == "base+descriptors":
                    feature_cols = BASE_PROCESS + DESCRIPTORS
                elif feature_set == "base+atom_numbers+support":
                    feature_cols = BASE_PROCESS + ATOM_NUMBERS + SUPPORT
                elif feature_set == "all":
                    feature_cols = BASE_PROCESS + ATOM_NUMBERS + DESCRIPTORS + SUPPORT
                else:
                    raise ValueError(f"Invalid feature set: {feature_set}")
                target_col = "C2y"
                rng = random.Random(seed)
                rf_rng = random.Random(rng.randint(0, 1000000))

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
                    n_val_catalysts=n_val_catalysts,
                    n_test_catalysts=n_test_catalysts,
                    return_indices=True,
                    conditions=conds,
                )

                if cross_validation:
                    cross_val_masks = get_cross_validation_masks(
                        df,
                        train_indices,
                        split_strategy,
                        rng=rng,
                        n_folds=n_folds,
                        conditions=conds,
                    )
                else:
                    cross_val_masks = None

                for k, augm in enumerate(augmentations):
                    print(f"Training RF model for feature set: {feature_set}...")
                    print(
                        f"Training {'with' if augm else 'without'} data augmentation..."
                    )
                    start_time = time()
                    rf_results = train_random_forest(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                        feature_cols,
                        random_state=rf_rng,
                        augment_data_flag=augm,
                        model_type=model_type,
                        cross_val_masks=cross_val_masks,
                        cross_val_params=_cross_val_params,
                    )
                    elapsed_time = time() - start_time
                    print(f"Training completed in {elapsed_time:.2f} seconds.")
                    # compute metrics without plotting
                    y_true = rf_results["y_test"]
                    y_pred = rf_results["preds_test"]
                    if augm:
                        # average predictions over permutations
                        y_pred = y_pred.reshape(-1, 6).mean(axis=1)
                        y_true = y_true[::6]
                    ss_res = np.sum((y_pred - y_true) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                    mae = float(np.mean(np.abs(y_pred - y_true)))
                    mse = float(np.mean((y_pred - y_true) ** 2))
                    absolute_errors = np.abs(y_pred - y_true)
                    test_df = df.iloc[test_indices].reset_index(drop=True)
                    test_df["absolute_error"] = absolute_errors
                    mae_by_catalyst = test_df.groupby("Name")["absolute_error"].mean()
                    results_rows.append(
                        {
                            "model_type": "rf",
                            "feature_set": feature_set,
                            "augmentation": "yes" if augm else "no",
                            "n_train_catalysts": n_train_catalysts,
                            "n_val_catalysts": n_val_catalysts,
                            "n_test_catalysts": n_test_catalysts,
                            "seed": seed,
                            "r2": r2,
                            "mae": mae,
                            "mse": mse,
                            "condition": cond_name,
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
                    model_id = (
                        f"{settings_to_filename_map[(feature_set, augm)]}_{seed:04d}"
                    )
                    _collected_models[model_id] = rf_results["model"]
                    if seed not in _collected_splits:
                        _collected_splits[seed] = {
                            "train_indices": train_indices,
                            "val_indices": val_indices,
                            "test_indices": test_indices,
                        }
    results_df = pd.DataFrame(results_rows)
    summary_path = results_path / "results_summary.csv"
    if summary_path.exists():
        results_df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(summary_path, index=False)
    # models_path = results_path / f"rf_models_{seeds[0]}-{seeds[-1]}.joblib"
    # dump(collected_models, models_path)
    # splits_path = results_path / f"data_splits_{seeds[0]}-{seeds[-1]}.joblib"
    # dump(collected_splits, splits_path)
    print("All experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training/evaluation for RF")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the input data file",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=["1"],
        help="Seeds for running the experiments. Accepts ints (e.g. --seeds 1 2 3) or ranges (e.g. --seeds 1:100 or 1:10:2)",
    )
    parser.add_argument(
        "--n_train_catalysts",
        type=int,
        default=39,
        help="Number of training catalysts (default: 39)",
    )
    parser.add_argument(
        "--n_val_catalysts",
        type=int,
        default=10,
        help="Number of validation catalysts (default: 10)",
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
        help="Data split strategy: 'random', 'catalyst', or 'catalyst_random' (default: 'catalyst')",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="default",
        help="Model type: 'default', 'custom' (default: 'default')",
    )
    parser.add_argument(
        "--cross_validation",
        action="store_true",
        help="Whether to use cross-validation for hyperparameter tuning",
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
        default="rf_50",
        help="Parameter set for cross-validation",
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
        return list(dict.fromkeys(out))

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if args.split_strategy != "catalyst" and args.conditions != "None":
        raise ValueError("Conditions can only be used with 'catalyst' split strategy.")
    if args.conditions == "None":
        conditions = {"": {}}
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
        n_val_catalysts=args.n_val_catalysts,
        n_test_catalysts=args.n_test_catalysts,
        split_strategy=args.split_strategy,
        model_type=args.model_type,
        cross_validation=args.cross_validation,
        n_folds=args.n_folds,
        cross_val_params=args.cross_val_params,
        conditions=conditions,
        feature_sets=args.feature_sets,
        augmentations=augmentations,
    )
