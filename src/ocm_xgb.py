import random
import argparse
from pathlib import Path
from joblib import dump
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from scipy.stats import spearmanr

from utils import (
    BASE_PROCESS,
    PROCESS_CONDITIONS,
    ELEMENT_RATIOS,
    ATOM_NUMBERS,
    SUPPORT,
    DESCRIPTORS,
    INVERSE_SECOND_IONIZATION,
    MULLIKEN_ELECTRONEGATIVITY,
    SUPPORT_TYPES,
    CONDITIONS_TEMP_SINGLE,
    CONDITIONS_TEMP_PAIRS,
    CONDITIONS_CH4_O2_RATIO,
    get_invariant_embedding,
    get_one_hot_encoding,
    get_composition_descriptors,
    get_xenonpy_descriptors,
    get_cross_validation_param_sets,
    scale_data,
    split_data,
    augment_data,
    scatter_mean,
    save_results_csv,
    get_cross_validation_masks,
    settings_to_filename_map,
)

# additional descriptors that can be appended to the feature set base+descriptors with "+",
# e.g. "base+descriptors+mulliken+band_center"
BASE_DESCRIPTORS = "base+descriptors"
ADDITIONAL_DESCRIPTORS = {
    "second_ionization": INVERSE_SECOND_IONIZATION,
    "mulliken": MULLIKEN_ELECTRONEGATIVITY,
    "band_center": ["support_band_center"],
    "support_types": SUPPORT_TYPES,
}


def is_descriptor_feature_set(feature_set):
    return feature_set == BASE_DESCRIPTORS or feature_set.startswith(BASE_DESCRIPTORS + "+")


def get_additional_descriptors(feature_set):
    """Names of the additional descriptors appended to base+descriptors (e.g. ["mulliken"])."""
    additional = [part for part in feature_set[len(BASE_DESCRIPTORS):].split("+") if part]
    unknown = [part for part in additional if part not in ADDITIONAL_DESCRIPTORS]
    if unknown:
        raise ValueError(
            f"Invalid additional descriptors {unknown} in feature set {feature_set} "
            f"(choose from {list(ADDITIONAL_DESCRIPTORS)})"
        )
    return additional


def get_descriptor_feature_cols(feature_set):
    """Feature columns of base+descriptors and the additional descriptors appended with "+"."""
    feature_cols = BASE_PROCESS + DESCRIPTORS
    for part in get_additional_descriptors(feature_set):
        feature_cols += [col for col in ADDITIONAL_DESCRIPTORS[part] if col not in feature_cols]
    return feature_cols


def get_model_name(feature_set, augmentation):
    """Short name of a feature set and augmentation setting (used for stored models)."""
    if (feature_set, augmentation) in settings_to_filename_map:
        return settings_to_filename_map[(feature_set, augmentation)]
    if is_descriptor_feature_set(feature_set):
        base_name = settings_to_filename_map[(BASE_DESCRIPTORS, augmentation)]
        return "+".join([base_name] + get_additional_descriptors(feature_set))
    raise ValueError(f"No model name defined for feature set {feature_set}")


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
    test_group_ids = None
    if augment_data_flag:
        X_train, y_train, X_val, y_val, X_test, y_test, cross_val_masks, group_ids = (
            augment_data(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                feature_cols,
                other_lists=cross_val_masks,
            )
        )
        test_group_ids = group_ids[2]

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
    if test_group_ids is not None:
        # collapse augmented copies back to one prediction per original row,
        # so downstream evaluation code doesn't need to know about augmentation
        preds = scatter_mean(preds, test_group_ids)
        _, first_idx = np.unique(test_group_ids, return_index=True)
        y_test = y_test[first_idx]

    results = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "history": [],
        "preds_test": preds,
        "y_test": y_test,
        "X_test": X_test,
    }
    return results


def plot_test_results(ax, y_true, y_pred):
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


def compute_ranking_performance(test_df, target_col, y_pred, top_k=3):
    """
    Compute ranking performance metrics: top-k accuracy and Spearman correlation.

    Args:
        test_df: DataFrame containing true target values and catalys IDs
        target_col: Name of the column containing true target values
        y_pred: Predicted target values
        top_k: Number of top catalysts to consider for accuracy

    Returns:
        Dictionary with top-k accuracy and Spearman correlation
    """
    test_df["y_pred"] = y_pred
    # use maximum of y_true and y_pred per catalyst to determine ranking
    df_agg = (
        test_df.groupby("Name").agg({target_col: "max", "y_pred": "max"}).reset_index()
    )

    # check if predicted y is constant
    if df_agg["y_pred"].nunique() == 1:
        print(
            "WARNING: Predicted values are constant. Spearman correlation is undefined."
        )
        spearman_corr = 0.0
        top_k_accuracy = 0.0
    else:
        # Get top-k catalysts based on aggregated true and predicted values
        top_k_true = df_agg.nlargest(top_k, target_col, keep="all")["Name"].values
        # use length of ground_truth top_k_true to determine top_k_pred in case of ties
        top_k_pred = df_agg.nlargest(len(top_k_true), "y_pred")["Name"].values

        # Compute top-k accuracy by taking intersection of sets
        # Use min for cases where there are ties in the top-k selection
        top_k_accuracy = min(1, len(set(top_k_true) & set(top_k_pred)) / top_k)

        # Compute Spearman correlation
        spearman_corr, _ = spearmanr(df_agg[target_col], df_agg["y_pred"])

    return {
        f"top_{top_k}_accuracy": top_k_accuracy,
        "spearman_corr": spearman_corr,
    }


def compute_shap_values(model, X_test, feature_cols):
    """
    Compute mean absolute SHAP values per feature for a trained model.

    Args:
        model: Trained XGBoost model.
        X_test: Test features the model was evaluated on (post augmentation
            and scaling).
        feature_cols: List of feature column names, in the same order as the
            columns of X_test.

    Returns:
        dict: Mapping "shap_<feature_col>" -> mean absolute SHAP value.
    """
    expl = shap.TreeExplainer(model)
    sv = expl(X_test, check_additivity=False)
    values = sv.values if hasattr(sv, "values") else sv  # compatibility
    mean_abs_shap = np.mean(np.abs(values), axis=0)
    return {f"shap_{col}": val for col, val in zip(feature_cols, mean_abs_shap)}


def predict_xgboost(model, scaler, X, feature_cols, augment_data_flag=False):
    """
    Predict with a model from train_xgboost on unscaled and unaugmented data, using
    the same pipeline as for the test set in train_xgboost (augmentation, scaling,
    averaging the predictions of augmented copies). Returns one prediction per row of X.
    """
    group_ids = None
    if augment_data_flag:
        X, _, _, _, _, _, _, group_ids = augment_data(
            X, np.zeros(len(X)), None, None, None, None, feature_cols
        )
        group_ids = group_ids[0]
    preds = model.predict(scaler.transform(X))
    if group_ids is not None:
        preds = scatter_mean(preds, group_ids)
    return preds


def compute_permutation_importance(
    model,
    scaler,
    X_test,
    test_df,
    target_col,
    feature_cols,
    feature_groups,
    augment_data_flag=False,
    n_repeats=10,
    rng=None,
):
    """
    Compute permutation importance on the test set for every single feature and for
    groups of features (all features of a group are permuted jointly).
    Features (or groups) that are constant within each catalyst (e.g. composition
    descriptors) are permuted on the catalyst level, i.e. every test catalyst gets the
    values of another (randomly assigned) test catalyst for all of its rows. All other
    features (e.g. process conditions) are permuted row-wise. The columns M1_mol%,
    M2_mol%, M3_mol% are ignored when checking whether features are constant within
    each catalyst, since the original data contains different values for some
    catalysts at different temperatures (they are permuted on the catalyst level
    together with the other features of the catalyst).
    The importance is the increase in MAE and the decrease in the Spearman correlation
    of the catalyst ranking (by maximum yield) compared to the unpermuted test set,
    so positive values indicate important features.

    Args:
        model, scaler: Trained model and scaler from train_xgboost.
        X_test: Unscaled and unaugmented test features (rows aligned with test_df).
        test_df: Test data with catalyst names ("Name") and target values.
        target_col: Name of the target column.
        feature_cols: List of feature names (columns of X_test).
        feature_groups: Dict of group name -> list of feature names.
        augment_data_flag: Whether the model was trained with data augmentation.
        n_repeats: Number of permutations per feature (group).
        rng: np.random.Generator used for the permutations.

    Returns:
        Dict with mean and std over the repetitions of the importance for every
        feature (perm_{metric}_{feature}_{mean,std}) and group
        (perm_{metric}_group_{group}_{mean,std}), with metric "mae" and "spearman".
    """
    if rng is None:
        rng = np.random.default_rng()
    y_true = test_df[target_col].values
    names = test_df["Name"].values
    catalysts, catalyst_codes = np.unique(names, return_inverse=True)
    # one representative row per catalyst (for catalyst-level permutations)
    _, first_rows = np.unique(catalyst_codes, return_index=True)

    def evaluate(X):
        preds = predict_xgboost(model, scaler, X, feature_cols, augment_data_flag)
        mae = float(np.mean(np.abs(preds - y_true)))
        spearman = compute_ranking_performance(test_df.copy(), target_col, preds)[
            "spearman_corr"
        ]
        return mae, spearman

    def is_catalyst_level(cols):
        # ignore mol% columns (inconsistent within some catalysts in the original data)
        cols = [col for col in cols if feature_cols[col] not in ELEMENT_RATIOS]
        values = pd.DataFrame(X_test[:, cols]).groupby(catalyst_codes).nunique()
        return bool((values <= 1).all().all())

    base_mae, base_spearman = evaluate(X_test)
    to_permute = {feature: [feature] for feature in feature_cols}
    to_permute.update(
        {f"group_{group}": features for group, features in feature_groups.items()}
    )
    importance = {}
    for name, features in to_permute.items():
        cols = [feature_cols.index(feature) for feature in features]
        catalyst_level = is_catalyst_level(cols)
        delta_mae, delta_spearman = [], []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            if catalyst_level:
                donor_rows = first_rows[rng.permutation(len(catalysts))[catalyst_codes]]
            else:
                donor_rows = rng.permutation(len(X_test))
            X_perm[:, cols] = X_test[donor_rows][:, cols]
            mae, spearman = evaluate(X_perm)
            delta_mae.append(mae - base_mae)
            delta_spearman.append(base_spearman - spearman)
        importance[f"perm_mae_{name}_mean"] = float(np.mean(delta_mae))
        importance[f"perm_mae_{name}_std"] = float(np.std(delta_mae))
        importance[f"perm_spearman_{name}_mean"] = float(np.mean(delta_spearman))
        importance[f"perm_spearman_{name}_std"] = float(np.std(delta_spearman))
    return importance


def main(
    data_path,
    seeds,
    n_train_catalysts,
    n_test_catalysts,
    split_strategy,
    n_folds=5,
    cross_val_params=50,
    conditions={},
    folder_name=None,
    feature_sets=["base+atom_numbers+support", "base+descriptors", "all"],
    augmentations=[True, False],
    store_plots=False,
    store_models=False,
    compute_shap=False,
    permutation_importance=False,
    permutation_repeats=10,
):
    # read data
    df = pd.read_csv(data_path)
    # storage path for results
    folder_name = f"xgb_cv_random_{cross_val_params}"
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
                f"random_{cross_val_params}", seed=seed
            )
            for j, feature_set in enumerate(feature_sets):
                _df = df
                # run experiments per feature set
                if is_descriptor_feature_set(feature_set):
                    # base+descriptors, optionally with additional descriptors appended with "+"
                    feature_cols = get_descriptor_feature_cols(feature_set)
                    missing = [col for col in feature_cols if col not in df.columns]
                    if missing:
                        raise ValueError(
                            f"Columns {missing} for feature set {feature_set} not found in the data "
                            "(use a dataset built with src/data_scripts/build_updated_descriptor_data.py)"
                        )
                elif feature_set == "base+atom_numbers+support":
                    feature_cols = BASE_PROCESS + ATOM_NUMBERS + SUPPORT
                elif feature_set == "all":  # all features
                    feature_cols = BASE_PROCESS + ATOM_NUMBERS + DESCRIPTORS + SUPPORT
                elif feature_set in ["one_hot", "invariant"]:
                    feature_cols = BASE_PROCESS + ATOM_NUMBERS + SUPPORT
                    _df = df[feature_cols + ["C2y", "Name"]]
                    if feature_set == "one_hot":
                        _df = get_one_hot_encoding(_df)
                    elif feature_set == "invariant":
                        _df = get_invariant_embedding(_df)
                    feature_cols = [
                        col for col in _df.columns if col not in ["C2y", "Name"]
                    ]
                elif feature_set == "composition_descriptors":
                    feature_cols = BASE_PROCESS + DESCRIPTORS
                    _df = df[feature_cols + ["C2y", "Name"]]
                    _df = get_composition_descriptors(_df)
                    feature_cols = [
                        col for col in _df.columns if col not in ["C2y", "Name"]
                    ]

                elif feature_set in ["xenonpy", "xenonpy_active"]:
                    # xenonpy: XenonPy descriptors of active phase and support,
                    # xenonpy_active: only of the active phase (without the support block)
                    feature_cols = PROCESS_CONDITIONS + ["support_surface_area"]
                    _df = df[feature_cols + ["C2y", "Name"]]
                    _df = get_xenonpy_descriptors(
                        _df, include_support=(feature_set == "xenonpy")
                    )
                    feature_cols = [
                        col for col in _df.columns if col not in ["C2y", "Name"]
                    ]
                else:
                    raise ValueError(f"Invalid feature set: {feature_set}")
                target_col = "C2y"

                # initialize random seed (for data splitting, we want the same splits each run)
                rng = random.Random(seed)
                # also use the same random state for training xgboost with/without augmentation
                xgb_seed = rng.randint(0, 1000000)

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
                    _df,
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
                    _df,
                    train_indices,
                    split_strategy,
                    rng=rng,
                    n_folds=n_folds,
                    conditions=conds,
                )

                for k, augm in enumerate(augmentations):
                    # check if augmentation is applicable
                    if (
                        feature_set
                        in ["invariant", "xenonpy", "xenonpy_active", "composition_descriptors"]
                        and augm
                    ):
                        print(
                            f"WARNING: Data augmentation is not applicable for feature set `{feature_set}`. Ignoring augmentation."
                        )
                        continue

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
                        random_state=random.Random(xgb_seed),
                        augment_data_flag=augm,
                        cross_val_masks=cross_val_masks,
                        cross_val_params=_cross_val_params,
                    )
                    elapsed_time = time() - start_time
                    print(f"Training completed in {elapsed_time:.2f} seconds.")
                    # compute metrics
                    r2, mae, mse, absolute_errors = plot_test_results(
                        ax=axes[i, j * len(augmentations) + k],
                        y_true=xgb_results["y_test"],
                        y_pred=xgb_results["preds_test"],
                    )
                    # compute per-catalyst MAE and max true/predicted yield
                    test_df = _df.iloc[test_indices].reset_index(drop=True)
                    test_df["absolute_error"] = absolute_errors
                    test_df["y_pred"] = xgb_results["preds_test"]
                    catalyst_stats = test_df.groupby("Name").agg(
                        absolute_error=("absolute_error", "mean"),
                        true_yield_max=(target_col, "max"),
                        pred_yield_max=("y_pred", "max"),
                    )
                    mae_by_catalyst = catalyst_stats["absolute_error"]
                    true_yield_max_by_catalyst = catalyst_stats["true_yield_max"]
                    pred_yield_max_by_catalyst = catalyst_stats["pred_yield_max"]
                    # compute ranking performance
                    ranking_performance = compute_ranking_performance(
                        test_df, target_col, xgb_results["preds_test"], top_k=3
                    )
                    # print results for this experiment
                    print(f"Test R2: {r2:.4f}")
                    print(
                        f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {np.sqrt(mse):.4f}"
                    )
                    for k, v in ranking_performance.items():
                        print(f"{k}: {v:.4f}")
                    # compute SHAP values
                    shap_cols = {}
                    if compute_shap:
                        shap_cols = compute_shap_values(
                            xgb_results["model"],
                            xgb_results["X_test"],
                            feature_cols,
                        )
                    # compute permutation importance of single features and of the groups
                    # process conditions and catalyst composition (incl. support)
                    perm_cols = {}
                    if permutation_importance:
                        feature_groups = {
                            "process": [col for col in feature_cols if col in PROCESS_CONDITIONS],
                            "composition": [col for col in feature_cols if col not in PROCESS_CONDITIONS],
                        }
                        perm_cols = compute_permutation_importance(
                            xgb_results["model"],
                            xgb_results["scaler"],
                            X_test,
                            test_df,
                            target_col,
                            feature_cols,
                            {group: cols for group, cols in feature_groups.items() if cols},
                            augment_data_flag=augm,
                            n_repeats=permutation_repeats,
                            rng=np.random.default_rng(xgb_seed),
                        )
                    # store results for this experiment
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
                                for i, name in enumerate(catalyst_stats.index)
                            },
                            **{
                                f"mae_test_catalyst_{i}": mae_val
                                for i, mae_val in enumerate(mae_by_catalyst)
                            },
                            **{
                                f"true_yield_max_test_catalyst_{i}": val
                                for i, val in enumerate(true_yield_max_by_catalyst)
                            },
                            **{
                                f"pred_yield_max_test_catalyst_{i}": val
                                for i, val in enumerate(pred_yield_max_by_catalyst)
                            },
                            "training_time": elapsed_time,
                            **{
                                f"ranking_{k}": v
                                for k, v in ranking_performance.items()
                            },
                            **shap_cols,
                            **perm_cols,
                        }
                    )
                    # collect model and splits
                    if store_models:
                        model_id = f"{get_model_name(feature_set, augm)}_{seed:04d}"
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
                results_path
                / f"xgb_test_results_{seeds[0]}-{seeds[-1]}_{cond_name}.png"
            )
    # save results dataframe to csv
    results_df = pd.DataFrame(results_rows)
    summary_path = results_path / "training_results.csv"
    save_results_csv(results_df, summary_path)
    if store_models:
        # save models to disk
        models_path = results_path / f"xgb_models_{seeds[0]}-{seeds[-1]}.joblib"
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
        required=True,
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
        type=int,
        default=50,
        help="Number of hyper-parameter combinations randomly sampled for cross-validation (default: 50)",
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
        help="Feature sets to run (choose from 'base+atom_numbers+support', 'base+descriptors', 'all', 'one_hot', 'invariant', 'composition_descriptors', 'xenonpy', 'xenonpy_active'; base+descriptors can be extended with additional descriptors appended with '+', e.g. 'base+descriptors+mulliken+band_center', choose from 'second_ionization', 'mulliken', 'band_center', 'support_types') (default: ['base+atom_numbers+support', 'base+descriptors', 'all']) ",
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
    parser.add_argument(
        "--compute_shap",
        action="store_true",
        help="Whether to compute SHAP values for each trained model directly "
        "after training (default: False)",
    )
    parser.add_argument(
        "--compute_permutation_importance",
        action="store_true",
        help="Whether to compute the permutation importance of single features and of the "
        "groups process conditions and catalyst composition on the test set (default: False)",
    )
    parser.add_argument(
        "--permutation_repeats",
        type=int,
        default=10,
        help="Number of permutations per feature (group) for the permutation importance (default: 10)",
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

    assert (
        args.n_folds > 1
    ), "Number of folds for cross-validation must be greater than 1."
    assert (
        args.cross_val_params > 0
    ), "Number of cross-validation parameter sets must be greater than 0."

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
        compute_shap=args.compute_shap,
        permutation_importance=args.compute_permutation_importance,
        permutation_repeats=args.permutation_repeats,
    )
