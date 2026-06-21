import random
import os
import time
import operator
from pathlib import Path
from itertools import permutations, product
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_PROCESS = [
    "Temp",
    "CT",
    "CH4/O2",
    "Ar_flow",
    "O2_flow",
    "CH4_flow",
    "Total_flow",
    "M1_mol%",
    "M2_mol%",
    "M3_mol%",
]
ATOM_NUMBERS = ["M1_atom_number", "M2_atom_number", "M3_atom_number"]
SUPPORT = ["Support_ID"]
DESCRIPTORS = [
    "M1_electronegativity",
    "M1_inverse_ionization",
    "M1_d_electron_count",
    "M1_ionic_radius",
    "M1_oxidation_state",
    "M2_electronegativity",
    "M2_inverse_ionization",
    "M2_d_electron_count",
    "M2_ionic_radius",
    "M2_oxidation_state",
    "M3_electronegativity",
    "M3_inverse_ionization",
    "M3_d_electron_count",
    "M3_ionic_radius",
    "M3_oxidation_state",
    "support_surface_area",
]

CONDITIONS_TEMP_SINGLE = {
    "700": {"Temp": {"==": 700}},
    "750": {"Temp": {"==": 750}},
    "775": {"Temp": {"==": 775}},
    "800": {"Temp": {"==": 800}},
    "850": {"Temp": {"==": 850}},
    "900": {"Temp": {"==": 900}},
}
CONDITIONS_TEMP_PAIRS = {
    "700&750": {"Temp": {">=": 700, "<=": 750}},
    "775&800": {"Temp": {">=": 775, "<=": 800}},
    "850&900": {"Temp": {">=": 850, "<=": 900}},
}
CONDITIONS_CH4_O2_RATIO = {
    "2": {"CH4/O2": {"==": 2}},
    "3": {"CH4/O2": {"==": 3}},
    "4": {"CH4/O2": {"==": 4}},
    "6": {"CH4/O2": {"==": 6}},
}

xgb_grid_params = {
    "n_estimators": [100, 350, 700],
    "learning_rate": [0.02, 0.05, 0.1, 0.3],
    "max_depth": [6, 10],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.6, 0.8],
    "reg_alpha": [0.0, 0.5],
    "reg_lambda": [0.5, 1.0, 2.0],
    "gamma": [0.0, 10],
    "min_child_weight": [1, 2, 3],
}

rf_grid_params = {
    "n_estimators": [100, 350, 700],
    "max_depth": [None, 6, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [0.5, 0.7, 1.0],
    "max_samples": [0.7, 0.9, None],
}

nn_grid_params = {
    "epochs": [
        20000,
    ],  # maximum number of epochs to train
    "batch_size": [512, 1024],
    "hidden_dims": [(128,), (256,), (256, 128), (256, 128, 64)],
    "lr": [1e-3, 5e-3, 1e-4],  # initial learning rate (lr)
    "min_lr": [
        1e-5,
    ],  # minimum lr at which training is stopped
    "weight_decay": [
        1e-5,
    ],  # weight decay for AdamW optimizer
    "patience": [1, 25, 50],  # reduce on plateau scheduler patience
    "factor": [0.5, 0.9],  # reduce on plateau lr reduction factor
    "dropout": [0, 0.2, 0.5],  # dropout rate for hidden layers
    "ema_decay": [
        0.99,
    ],  # decay for exponential moving average of model weights
    "ema_warmup": [
        1,
        5,
    ],  # number of epochs to wait before starting to apply EMA updates (allows model to warm up first)
    "early_stopping": [
        1500,
    ],  # number of epochs with no improvement after which training will be stopped
}

settings_to_filename_map = {
    # (feature_set, use_augmentation): filename
    ("base+descriptors", False): "b",
    ("base+descriptors", True): "ba",
    ("base+atom_numbers+support", False): "n",
    ("base+atom_numbers+support", True): "na",
    ("all", False): "a",
    ("all", True): "aa",
    ("unique", True): "aa+u",
    ("non_unique", True): "aa+nu",
    ("threshold", True): "aa+thresh",
    ("top_6", True): "aa+top6",
    ("weighted_mean", True): "aa+wmean",
    ("mean", True): "aa+mean",
}


def get_cross_validation_param_sets(param_set_name: str, seed: int = 42):
    if param_set_name.startswith("random_"):
        num_samples = int(param_set_name.split("_")[1])
        rng = random.Random(seed)
        param_values = [
            [rng.choice(xgb_grid_params[key]) for key in xgb_grid_params.keys()]
            for _ in range(num_samples)
        ]
        return [dict(zip(xgb_grid_params.keys(), vals)) for vals in param_values]
    elif param_set_name.startswith("rf_"):
        num_samples = int(param_set_name.split("_")[1])
        # compute all possible combinations of the params available in rf_grid_params
        _all_param_values = list(product(*rf_grid_params.values()))
        rng = np.random.default_rng(seed)
        param_values = rng.choice(
            _all_param_values, size=num_samples, replace=False
        ).tolist()
        return [dict(zip(rf_grid_params.keys(), vals)) for vals in param_values]
    elif param_set_name.startswith("nn_"):
        num_samples = int(param_set_name.split("_")[1])
        rng = random.Random(seed)
        param_values = [
            [rng.choice(nn_grid_params[key]) for key in nn_grid_params.keys()]
            for _ in range(num_samples)
        ]
        return [dict(zip(nn_grid_params.keys(), vals)) for vals in param_values]
    else:
        raise ValueError(
            f"Invalid cross-validation parameter set name: {param_set_name}"
        )


def get_results_csv(experiment_path: Path) -> pd.DataFrame:
    if experiment_path.exists() and experiment_path.is_dir():
        result_file = experiment_path / "results_summary.csv"
        if result_file.exists():
            df_res = pd.read_csv(result_file)
            return [df_res]
        else:
            results = []
            for folder in experiment_path.iterdir():
                results.extend(get_results_csv(folder))
            return results
    else:
        return []


def acquire_lock(lock_path: Path, timeout: float = 30.0, poll: float = 0.1):
    start = time.time()
    lock_path = Path(lock_path)
    while True:
        try:
            # try to create lock file atomically
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"{os.getpid()}\n{time.time()}\n")
            return
        except FileExistsError:
            # lock file exists, wait and retry
            if (time.time() - start) >= timeout:
                raise TimeoutError(f"Timeout waiting for lock {lock_path}")
            time.sleep(poll)


def release_lock(lock_path: Path):
    try:
        os.remove(str(lock_path))
    except FileNotFoundError:
        pass


def get_cross_validation_masks(
    df,
    train_indices,
    split_strategy,
    rng,
    n_folds=5,
    conditions={},
    catalyst_name_column="Name",
):
    """Generate cross-validation masks for catalyst-based splitting.

    Args:
        df (pd.DataFrame): The full dataset containing a 'Name' column for catalysts.
        train_indices (list): List of indices corresponding to training data in df.
        split_strategy (str): The data splitting strategy used ('catalyst', 'catalyst_random', or 'catalyst').
        rng (random.Random): Random state for shuffling the training data.
        n_folds (int): Number of cross-validation folds.
        conditions (dict): Conditions for filtering the data. Only used with 'catalyst' split strategy.
          If conditions are provided, only parts of the training data fulfilling the conditions
          will be used in the validation partitions of cross-validation (i.e. data points
          not fulfilling the conditions will remain in the training set all the time).
        catalyst_name_column (str): The name of the column containing catalyst names in df.

    Returns:
        list: A list of boolean masks for each fold (entries belonging to the validation split are marked as True).
    """
    if split_strategy == "catalyst":
        train_df = df.iloc[train_indices].reset_index(drop=True)
        if len(conditions) > 0:
            # If conditions are provided, only use data points fulfilling the conditions
            # in the validation splits and always keep the others in the training set
            cond_mask = get_conditions_mask(train_df, conditions)
            train_df = train_df[cond_mask]
        train_catalysts = train_df[catalyst_name_column].unique()
        n_catalysts = len(train_catalysts)
        # if there is only one or no catalyst, we cannot do catalyst-based CV
        # do random CV instead
        if n_catalysts <= 1:
            print(
                f"Warning: Only {n_catalysts} training catalyst(s) specified for cross-validation. "
                f"Will run cross validation based on random folds instead of using catalyst-based folds."
            )
            _val_masks = get_cross_validation_masks(
                train_df,
                train_indices,
                "random",
                rng,
                n_folds=n_folds,
            )
            # shuffle masks since indices are not shuffled with split_strategy 'catalyst'
            rng = np.random.default_rng(rng.randint(0, 1000000))
            perm = rng.permutation(np.arange(len(_val_masks[0])))
            val_masks = [mask[perm] for mask in _val_masks]
            return val_masks
        # adjust n_folds if there are less catalysts than folds
        if n_folds > n_catalysts:
            n_folds = n_catalysts
            print(
                f"Warning: Reducing n_folds to {n_folds} since there are only {n_catalysts} training catalysts."
            )
        # shuffle train_catalysts
        rng = np.random.default_rng(rng.randint(0, 1000000))
        shuffled_catalysts = rng.permutation(train_catalysts)
        # split into n_folds parts
        val_catalysts = np.array_split(
            shuffled_catalysts[: n_catalysts * (n_catalysts // n_folds)], n_folds
        )
        # compute catalyst to indices mapping
        catalyst_to_indices = train_df.index.groupby(train_df[catalyst_name_column])
        # Create a mask for each fold
        val_masks = []
        for catalysts in val_catalysts:
            mask = np.zeros(len(train_indices), dtype=bool)
            idx = np.concatenate([catalyst_to_indices[c] for c in catalysts])
            mask[idx] = True
            val_masks.append(mask)
        return val_masks
    elif split_strategy in ["random", "catalyst_random"]:
        # For random splits, the train_indices are already shuffled, we just need to
        # partition them into n_folds folds
        n_train = len(train_indices)
        fold_sizes = (n_train // n_folds) * np.ones(n_folds, dtype=int)
        fold_sizes[: n_train % n_folds] += 1
        val_masks = []
        current = 0
        for fold_size in fold_sizes:
            mask = np.zeros(n_train, dtype=bool)
            mask[current : current + fold_size] = True
            val_masks.append(mask)
            current += fold_size
        return val_masks
    else:
        raise NotImplementedError(
            "Cross-validation masks are only implemented for 'catalyst', 'catalyst_random', and 'random' split strategy."
        )


def split_data(
    df,
    feature_cols,
    target_col,
    split_strategy,
    rng,
    test_size=0.2,
    val_size=0.0,
    n_train_catalysts=None,
    n_val_catalysts=None,
    n_test_catalysts=None,
    return_indices=False,
    conditions={},
    catalyst_name_column="Name",
):
    """
    Splits the data into training, validation, and test sets based on the specified strategy.

    Args:
        df (pd.DataFrame): The full dataset containing a 'Name' column for catalysts.
        feature_cols (list): List of feature column names.
        target_col (str): The name of the target column.
        split_strategy (str): The data splitting strategy used ('random', 'catalyst', 'catalyst_random').
            'random': Randomly splits the whole data into train/val/test sets.
            'catalyst': Splits the data based on catalysts, ensuring no catalyst overlap between sets.
            'catalyst_random': Randomly selects a subset of catalysts for training, then splits randomly.
        rng (random.Random): Random state for shuffling the data.
        test_size (float): Proportion of the data to include in the test split. Ignored with split_strategy 'catalyst' if n_test_catalysts is given.
        val_size (float): Proportion of the data to include in the validation split. Ignored with split_strategy 'catalyst' if n_val_catalysts is given.
        n_train_catalysts (int): Number of training catalysts to use in split_strategy 'catalyst' and 'catalyst_random' (must be > 0). If None, all catalysts in df that are not used for val or test splits.
        n_val_catalysts (int): Number of validation catalysts to use in split_strategy 'catalyst'. Inferred from val_size if None.
        n_test_catalysts (int): Number of test catalysts to use in split_strategy 'catalyst'. Inferred from test_size if None.
        return_indices (bool): Whether to return the indices of the splits.
        conditions (dict): Conditions for filtering the data before splitting. Only used with 'catalyst' split strategy.
        catalyst_name_column (str): The name of the column containing catalyst names in df.

    Returns:
        tuple: A tuple containing the training, validation, and test sets (X_train, y_train, X_val, y_val, X_test, y_test).
            If return_indices is True, also returns the indices of the splits (X_train, y_train, X_val, y_val, X_test, y_test, train_indices, val_indices, test_indices).
    """
    assert split_strategy in [
        "random",
        "catalyst",
        "catalyst_random",
    ], f"Invalid split strategy: {split_strategy}"

    X = df[feature_cols].values
    y = df[target_col].values

    if split_strategy == "catalyst":
        # Do catalyst-based splittingi (strategy 'catalyst')
        all_catalysts = df[catalyst_name_column].unique()
        # infer number of catalysts for each split if not provided
        if n_val_catalysts is None:
            if val_size == 0:
                n_val_catalysts = 0
            else:
                n_val_catalysts = max(int(len(all_catalysts) * val_size), 1)
        if n_test_catalysts is None:
            n_test_catalysts = max(int(len(all_catalysts) * test_size), 1)
        if n_train_catalysts is None:
            n_train_catalysts = len(all_catalysts) - n_val_catalysts - n_test_catalysts
            assert n_train_catalysts >= 1, "At least one training catalyst is required."
        assert n_train_catalysts + n_val_catalysts + n_test_catalysts <= len(
            all_catalysts
        ), "Not enough catalysts for the requested split sizes."

        # sample test catalysts and train catalysts (train includes val for now)
        if len(conditions) == 0:
            # no conditions provided, split according to catalyst names only
            test_catalysts = rng.sample(list(all_catalysts), n_test_catalysts)
            remaining_catalysts = [c for c in all_catalysts if c not in test_catalysts]
            train_catalysts = rng.sample(
                remaining_catalysts, n_train_catalysts + n_val_catalysts  # val included
            )
            train_indices, test_indices = generate_splits(
                df,
                train_catalysts,
                test_catalysts,
                conditions={},
                catalyst_name_column=catalyst_name_column,
            )
        else:
            # use all catalysts for training but remove all data points at the target
            # condition except for n_training_catalysts+n_val_catalysts catalysts
            # the test set will only consist of filtered data points from
            # n_test_catalysts catalysts in the train set at the target condition
            all_shuffled = rng.sample(list(all_catalysts), len(all_catalysts))
            train_catalysts = all_shuffled[:n_train_catalysts + n_val_catalysts]
            train_indices, _ = generate_splits(
                df,
                train_catalyst_names=all_shuffled,
                test_catalyst_names=[],
                conditions=conditions,
                full_catalyst_names=train_catalysts,
                catalyst_name_column=catalyst_name_column,
            )
            if n_test_catalysts > 0:
                test_catalysts = all_shuffled[-n_test_catalysts :]
                _, test_indices = generate_splits(
                    df,
                    train_catalyst_names=test_catalysts,
                    test_catalyst_names=[],
                    conditions=conditions,
                    full_catalyst_names=[],
                    catalyst_name_column=catalyst_name_column,
                )
            else:
                test_catalysts = []
                test_indices = train_indices[0:0]  # empty array of indices
        X_test = X[test_indices]
        y_test = y[test_indices]

        # if val catalysts are requested, sample them from the train catalysts
        if n_val_catalysts > 0:
            val_catalysts = rng.sample(train_catalysts, n_val_catalysts)
            train_catalysts = [c for c in train_catalysts if c not in val_catalysts]
            if len(conditions) == 0:
                # no conditions provided, split according to catalyst names only
                train_indices, val_indices = generate_splits(
                    df,
                    train_catalysts,
                    val_catalysts,
                    conditions={},
                    catalyst_name_column=catalyst_name_column,
                )
            else:
                # use conditions to filter a few data points that fulfill the conditions
                # for the validation set
                # we cannot directly get train_indices here because it actually consists
                # of data from more catalysts than those in train_catalysts+val_catalysts
                _, val_indices = generate_splits(
                    df,
                    train_catalyst_names=train_catalysts + val_catalysts,
                    test_catalyst_names=[],
                    conditions=conditions,
                    full_catalyst_names=train_catalysts,
                )
                # instead, we explicitly remove the val_indices from the train_indices
                train_indices = train_indices.difference(val_indices)
            X_val = X[val_indices]
            y_val = y[val_indices]
        else:
            X_val = None
            y_val = None
        X_train = X[train_indices]
        y_train = y[train_indices]
    else:
        # Do random splitting (strategies 'random' and 'catalyst_random')
        if split_strategy == "catalyst_random":
            # Reduce data set to only training catalysts
            if n_train_catalysts is None or n_train_catalysts <= 0:
                raise ValueError(
                    "n_train_catalysts must be specified and > 0 for catalyst_random split."
                )
            all_catalysts = df[catalyst_name_column].unique()
            train_catalysts = rng.sample(list(all_catalysts), n_train_catalysts)
            train_indices, _ = generate_splits(
                df,
                train_catalysts,
                test_catalyst_names=[],
                conditions={},
                catalyst_name_column=catalyst_name_column,
            )
            all_indices = train_indices
        else:
            all_indices = np.arange(X.shape[0])
        # Sample test and train splits first (train includes val for now)
        trainval_indices, test_indices = train_test_split(
            all_indices, test_size=test_size, random_state=rng.randint(0, 1000000)
        )
        X_test = X[test_indices]
        y_test = y[test_indices]
        if val_size > 0:
            # If validation size is specified, split trainval into train and val
            val_relative = val_size / (1.0 - test_size)
            train_indices, val_indices = train_test_split(
                trainval_indices,
                test_size=val_relative,
                random_state=rng.randint(0, 1000000),
            )
            X_val = X[val_indices]
            y_val = y[val_indices]
        else:
            train_indices = trainval_indices
            X_val = None
            y_val = None
        X_train = X[train_indices]
        y_train = y[train_indices]

    # return splits
    if return_indices:
        if X_val is None:
            val_indices = None
        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            train_indices,
            val_indices,
            test_indices,
        )
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test


def augment_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_cols,
    permutation_cols=(
        ("M1_atom_number", "M2_atom_number", "M3_atom_number"),
        ("M1_mol%", "M2_mol%", "M3_mol%"),
        ("M1_electronegativity", "M2_electronegativity", "M3_electronegativity"),
        ("M1_inverse_ionization", "M2_inverse_ionization", "M3_inverse_ionization"),
        ("M1_d_electron_count", "M2_d_electron_count", "M3_d_electron_count"),
        ("M1_ionic_radius", "M2_ionic_radius", "M3_ionic_radius"),
        ("M1_oxidation_state", "M2_oxidation_state", "M3_oxidation_state"),
    ),
    other_lists=None,
):
    """
    Parameters:
    - X_train: Training data features
    - y_train: Training data targets
    - X_val: Validation data features (can be None)
    - y_val: Validation data targets (can be None only if X_val is None)
    - X_test: Test data features (can be None)
    - y_test: Test data targets (can be None only if X_test is None)
    - feature_cols: List of feature column names
    - permutation_cols: Tuple of tuples of three column names to permute in all
        possible combinations (i.e. each input will have 6 permutations added).
        The entries in the tuples must be ordered identically in all tuples (e.g.
        all features belonging to M1 at the first position, M2 at the second and
        M3 at the third, like:
        (("M1_atom_number", "M2_atom_number", "M3_atom_number"),
         ("M1_electronegativity", "M2_electronegativity", "M3_electronegativity")).
    - other_lists: Other arrays with shape like y_train that need to be repeated
        (e.g. masks for indexing the training set).
    """
    # Augment data by adding all permutations of the specified columns
    assert (
        len(feature_cols) == X_train.shape[1]
    ), "Number of feature_cols must match number of columns in X_train"
    n_features = len(feature_cols)
    all_permutations = np.array(list(permutations([0, 1, 2])))
    all_indices = np.arange(n_features)[None, :].repeat(6, 0)
    for permutation_triple in permutation_cols:
        f1, f2, f3 = permutation_triple
        if f1 in feature_cols and f2 in feature_cols and f3 in feature_cols:
            idx1 = feature_cols.index(f1)
            idx2 = feature_cols.index(f2)
            idx3 = feature_cols.index(f3)
            all_indices[:, [idx1, idx2, idx3]] = all_indices[0, [idx1, idx2, idx3]][
                all_permutations
            ]

    X_train = X_train[:, all_indices].reshape(-1, n_features)
    y_train = y_train.repeat(6)
    if X_val is not None:
        X_val = X_val[:, all_indices].reshape(-1, n_features)
        y_val = y_val.repeat(6)
    if X_test is not None:
        X_test = X_test[:, all_indices].reshape(-1, n_features)
        y_test = y_test.repeat(6)
    if other_lists is not None:
        repeated_lists = []
        for arr in other_lists:
            repeated_lists.append(arr.repeat(6))
    else:
        repeated_lists = None
    return X_train, y_train, X_val, y_val, X_test, y_test, repeated_lists


def scale_data(
    X_train,
    X_val,
    X_test,
    feature_cols,
    no_mean_cols=[],
    passthrough_cols=[],
):
    # Build a ColumnTransformer that preserves original column order:
    # we create one transformer per original column index, using StandardScaler for cont cols
    # and 'passthrough' for atom id cols. Listing transformers in index order keeps output order.
    ct_transformers = []
    for i in range(X_train.shape[1]):
        if feature_cols[i] in no_mean_cols:
            ct_transformers.append(
                (f"scaler_no_mean_{i}", StandardScaler(with_mean=False), [i])
            )
        elif feature_cols[i] in passthrough_cols:
            ct_transformers.append((f"pass_{i}", "passthrough", [i]))
        else:
            ct_transformers.append((f"scaler_{i}", StandardScaler(), [i]))

    col_transformer = ColumnTransformer(transformers=ct_transformers, remainder="drop")

    # use the ColumnTransformer as the scaler object to return
    scaler = col_transformer

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if X_val is not None:
        X_val = scaler.transform(X_val)

    return X_train, X_val, X_test, scaler


def get_conditions_mask(df, conditions):
    """Generate a boolean mask for the DataFrame based on the given conditions.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        conditions (dict): Dictionary specifying column-based conditions.
            All data points that fulfill these conditions will marked True.
            The format is:
                {
                    "column_name": {
                        "operator": value,
                        ...
                    },
                    ...
            Supported operators are: '<', '<=', '>', '>=', '==', '=', '!='.
            Example:
                {
                    "Temp": {">=": 750, "<=": 800},
                    "Ar_flow": {"==": 10.5}
            This means: Select rows where Temp >= 750 && Temp <= 800 && Ar_flow == 10.5
            and mark these True.
            If conditions is an empty dict, all rows will be marked True.

    Returns:
        pd.Series: A boolean mask where True indicates rows that meet the conditions.
    """
    # Map string operators to functions
    ops = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "=": operator.eq,  # Handle both '==' and '='
        "!=": operator.ne,
    }
    cond_mask = pd.Series([True] * len(df), index=df.index)
    for col, conds in conditions.items():
        for op_str, val in conds.items():
            cond_mask &= ops[op_str](df[col], val)
    return cond_mask


def generate_splits(
    df,
    train_catalyst_names,
    test_catalyst_names,
    conditions,
    full_catalyst_names=None,
    catalyst_name_column="Name",
):
    """
    Splits a DataFrame into training and test indices based on catalyst names and
    additional column conditions.

    Args:
        df (pd.DataFrame): The input DataFrame containing at least a 'Name' column and
            any columns referenced in `conditions`.
        train_catalyst_names (Iterable[str]): List or set of catalyst names to be
            considered for the training set.
        test_catalyst_names (Iterable[str]): List or set of catalyst names to be
            included in the test set.
        conditions (dict): Dictionary specifying column-based conditions for splitting.
            All data points that fulfill these conditions will be added to the test set.
            The format is:
                {
                    "column_name": {
                        "operator": value,
                        ...
                    },
                    ...
            Supported operators are: '<', '<=', '>', '>=', '==', '=', '!='.
            Example:
                {
                    "Temp": {">=": 750, "<=": 800},
                    "Ar_flow": {"==": 10.5}
            This means: Select rows where Temp >= 750 && Temp <= 800 && Ar_flow == 10.5
            and add these data points to the test set.
            If conditions is an empty dict, all train_catalyst_names will be assigned to
            the training set.
        full_catalyst_names (Iterable[str], optional): List or set of catalyst names
            that has to be a subset of train_catalyst_names.
            All experiments of these catalysts will be included in the training set,
            regardless of conditions. In this way, it can be simulated that experiments
            with the target conditions exist for training for some of the catalysts.
            Defaults to None.
        catalyst_name_column (str): The name of the column in df that contains the catalyst names. Defaults to "Name".

    Returns:
        tuple:
            train_indices (pd.Index): Indices of rows assigned to the training set
                (train_catalyst_names that do NOT meet all conditions).
            test_indices (pd.Index): Indices of rows assigned to the test set
                (all test_catalyst_names, plus train_catalyst_names that meet all
                conditions).
    """
    # Get indices for test set: all rows with catalyst in test_catalysts
    test_mask = df[catalyst_name_column].isin(test_catalyst_names)
    test_indices = df.index[test_mask]

    # For catalysts in train_catalysts, check conditions
    train_mask = df[catalyst_name_column].isin(train_catalyst_names)
    if len(conditions) == 0:
        # If no conditions specified, all train_catalysts go to train set
        train_indices = df.index[train_mask]
        return train_indices, test_indices
    cond_mask = get_conditions_mask(df, conditions)
    # Ensure catalysts in full_catalyst_names are always in training set
    if full_catalyst_names:
        assert set(full_catalyst_names).issubset(
            set(train_catalyst_names)
        ), "full_catalyst_names must be a subset of train_catalyst_names"
        full_mask = df[catalyst_name_column].isin(full_catalyst_names)
        cond_mask &= ~full_mask
    # Indices of train_catalysts that meet conditions: add to test set
    train_cond_indices = df.index[train_mask & cond_mask]
    test_indices = test_indices.union(train_cond_indices)

    # Indices of train_catalysts that do NOT meet conditions: assign to train set
    train_indices = df.index[train_mask & ~cond_mask]

    return train_indices, test_indices
