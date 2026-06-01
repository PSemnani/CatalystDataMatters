from pathlib import Path
from joblib import load
import shap
import argparse

import pandas as pd
import numpy as np

from utils import (
    BASE_PROCESS,
    ATOM_NUMBERS,
    DESCRIPTORS,
    SUPPORT,
    settings_to_filename_map,
    augment_data,
    scale_data,
)


def get_test_splits(path: Path, conditions=None):
    # prepare dicts for gathering splits
    splits = {
        condition: {} for condition in (conditions if conditions is not None else [])
    }
    # gather splits for all seeds from all files
    # test samples are the same for all n_training_catalysts, load them from n=49
    n_training_catalysts = 49
    split_files = list((path / f"{n_training_catalysts}/").glob("data_splits_*.joblib"))
    for split_file in split_files:
        split_data = load(split_file)
        if conditions is not None:
            for condition in conditions:
                splits[condition].update(split_data[str(condition)])
        else:
            splits.update(split_data)
    # now extract test indices
    if conditions is not None:
        test_splits = {}
        for condition in conditions:
            test_splits[condition] = {
                seed: splits[condition][seed]["test_indices"]
                for seed in splits[condition]
            }
    else:
        test_splits = {seed: splits[seed]["test_indices"] for seed in splits}
    return test_splits


def get_train_splits(path: Path, n_train_catalysts: int, conditions=None):
    # prepare dicts for gathering splits
    splits = {
        condition: {} for condition in (conditions if conditions is not None else [])
    }
    # gather splits from all files
    split_files = list((path / f"{n_train_catalysts}/").glob("data_splits_*.joblib"))
    for split_file in split_files:
        split_data = load(split_file)
        if conditions is not None:
            for condition in conditions:
                splits[condition].update(split_data[str(condition)])
        else:
            splits.update(split_data)
    # now extract train indices
    if conditions is not None:
        train_splits = {}
        for condition in conditions:
            train_splits[condition] = {
                seed: splits[condition][seed]["train_indices"]
                for seed in splits[condition]
            }
    else:
        train_splits = {seed: splits[seed]["train_indices"] for seed in splits}
    return train_splits


def load_models(path: Path, n_train_catalysts: int, conditions=None):
    # prepare dicts for gathering models
    models = {
        condition: {} for condition in (conditions if conditions is not None else [])
    }
    # gather models from all files
    model_files = list((path / f"{n_train_catalysts}/").glob("xgb_models_*.joblib"))
    for model_file in model_files:
        model_data = load(model_file)
        if conditions is not None:
            for condition in conditions:
                models[condition].update(model_data[str(condition)])
        else:
            models.update(model_data)
    return models


def load_augment_scale_data(
    train_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    augmentation: bool,
    train_indices: list,
    test_indices: list,
):
    # extract training and test data
    X_train = train_df[feature_cols].values[train_indices]
    y_train = train_df[target_col].values[train_indices]
    X_test = train_df[feature_cols].values[test_indices]
    y_test = train_df[target_col].values[test_indices]
    X_val = None
    y_val = None
    # apply augmentation if needed
    if augmentation:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = augment_data(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols,
        )
    # Scaling
    X_train, X_val, X_test, scaler = scale_data(
        X_train,
        X_val,
        X_test,
        feature_cols,
    )
    return X_train, y_train, X_test, y_test


def load_and_apply_models(
    path: Path,
    results_df: pd.DataFrame,
    train_df: pd.DataFrame,
    apply_function: callable,
) -> pd.DataFrame:
    # load data and models for all experiments and apply some function to update results_df
    if "condition" not in results_df.columns or results_df["condition"].isnull().all():
        conditions = None
    else:
        conditions = list(map(int, results_df["condition"].unique()))
    target_col = "C2y"
    test_splits = get_test_splits(path, conditions=conditions)
    for n_train_catalysts, group in results_df.groupby("n_train_catalysts"):
        print(f"Processing experiments for n_train_catalysts={n_train_catalysts}...")
        models = load_models(path, n_train_catalysts, conditions=conditions)
        train_splits = get_train_splits(path, n_train_catalysts, conditions=conditions)
        for index in group.index:
            row = group.loc[index]
            # extract settings
            condition = None if conditions is None else row.condition
            seed = row.seed
            feature_set = row.feature_set
            augmentation = True if row.augmentation == "yes" else False
            # get train and test indices
            train_indices = (
                train_splits[seed]
                if conditions is None
                else train_splits[condition][seed]
            )
            test_indices = (
                test_splits[seed]
                if conditions is None
                else test_splits[condition][seed]
            )
            # determine feature columns
            if feature_set == "base+descriptors":
                feature_cols = BASE_PROCESS + DESCRIPTORS
            elif feature_set == "base+atom_numbers+support":
                feature_cols = BASE_PROCESS + ATOM_NUMBERS + SUPPORT
            elif feature_set == "all":  # all features
                feature_cols = BASE_PROCESS + ATOM_NUMBERS + DESCRIPTORS + SUPPORT
            else:
                raise ValueError(f"Unknown feature set: {feature_set}")
            # load, augment, and scale the training and test data
            X_train, y_train, X_test, y_test = load_augment_scale_data(
                train_df,
                feature_cols,
                target_col,
                augmentation,
                train_indices,
                test_indices,
            )
            # get the trained model
            model_id = (
                f"{settings_to_filename_map[(feature_set, augmentation)]}_{seed:04d}"
            )
            model = (
                models[model_id] if conditions is None else models[condition][model_id]
            )
            results_df = apply_function(
                results_df=results_df,
                index=index,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                augmentation=augmentation,
                feature_cols=feature_cols,
            )
    return results_df


def compute_shap_values(
    path: Path,
    results_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    # recompute shap values for all experiments and update the dataframe
    print("Computing SHAP values for all experiments...")

    def _update_shap(index, model, X_test, results_df, feature_cols, **kwargs):
        expl = shap.TreeExplainer(model)
        sv = expl(X_test, check_additivity=False)
        values = sv.values if hasattr(sv, "values") else sv  # compatibility
        mean_abs_shap = np.mean(np.abs(values), axis=0)
        col_names = [f"shap_{col}" for col in feature_cols]
        results_df.loc[index, col_names] = mean_abs_shap
        return results_df

    results_df = load_and_apply_models(
        path=path,
        results_df=results_df,
        train_df=train_df,
        apply_function=_update_shap,
    )
    print("Updated SHAP values in results dataframe.")
    return results_df


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser(
        description="Compute SHAP values for all experiments and update the results dataframe."
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to the experiment directory containing the results and data.",
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        required=True,
        help="Path to the training data CSV file.",
    )
    parser.add_argument(
        "--n_train_catalysts",
        type=int,
        required=True,
        help="Number of training catalysts to use.",
    )
    args = parser.parse_args()
    experiment_path = Path(args.experiment_path)
    train_data_file = Path(args.train_data_file)
    n_train_catalysts = args.n_train_catalysts

    results_df = pd.read_csv(experiment_path / "gathered_results.csv")
    train_df = pd.read_csv(train_data_file)
    results_df = results_df[results_df["n_train_catalysts"] == n_train_catalysts]
    results_df = compute_shap_values(
        path=experiment_path,
        results_df=results_df,
        train_df=train_df,
    )
    results_df.to_csv(
        experiment_path / f"gathered_results_shap_{n_train_catalysts}.csv",
        index=False,
    )
