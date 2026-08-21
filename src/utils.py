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
    ("one_hot", False): "oh",
    ("one_hot", True): "oha",
    ("invariant", False): "inv",
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
        result_file = experiment_path / "training_results.csv"
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


def get_one_hot_encoding(df, column_names=ATOM_NUMBERS + SUPPORT):
    """Get one-hot encoding for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_names (list): List of column names to one-hot encode (default: ATOM_NUMBERS + SUPPORT).

    Returns:
        pd.DataFrame: A new DataFrame with one-hot encoded columns that also contains
            all columns not in column_names in the same format as before.
    """
    return pd.get_dummies(df, columns=column_names, drop_first=False)


def get_invariant_embedding(df):
    """Get invariant embedding for the elements and support columns in a DataFrame.
        The invariant embedding is obtained by having one feature per element and support.
        The support is one-hot encoded, whereas the elements are indicated with their 
        corresponding concentration.
        For example, if the input DataFrame has elements 1,...,5 and supports A, B, C,
        and a data point has support B and elements M1=3, M2=5, and M3=1 with 
        concentrations 0.2, 0.5, and 0.3, respectively, the invariant embedding for this
        data point will be [0.3, 0, 0.2, 0, 0.5, 0, 1, 0].


    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with invariant embeddings for elements and support.
    """
    # Get unique elements and supports
    unique_elements = sorted(set(df[ATOM_NUMBERS[0]].unique()).union(
        df[ATOM_NUMBERS[1]].unique()).union(df[ATOM_NUMBERS[2]].unique())
    )
    unique_supports = sorted(df[SUPPORT[0]].unique())
    element_cols = [f"element_{str(e)}" for e in unique_elements]
    support_cols = [f"support_{str(s)}" for s in unique_supports]
    # Initialize the invariant embedding DataFrame
    invariant_df = pd.DataFrame(
        0,
        index=df.index,
        columns=element_cols + support_cols,
        dtype=float,
    )

    # Fill in the element concentrations
    for i, atom_col in enumerate(ATOM_NUMBERS):
        for element, name in zip(unique_elements, element_cols):
            mask = df[atom_col] == element
            invariant_df.loc[mask, name] = df.loc[mask, f"M{i+1}_mol%"]/100  # Assuming M1_mol%, M2_mol%, M3_mol% are in percentage

    # Fill in the one-hot encoded supports
    for support, name in zip(unique_supports, support_cols):
        invariant_df.loc[df[SUPPORT[0]] == support, name] = 1

    # Cast one-hot support columns to integer dtype (keeps element columns as float)
    invariant_df[support_cols] = invariant_df[support_cols].astype("int64")

    # Prepend all columns not in ATOM_NUMBERS + SUPPORT and ["M1_mol%", "M2_mol%", "M3_mol%"] to the invariant_df
    other_cols = [col for col in df.columns if col not in ATOM_NUMBERS + SUPPORT + ["M1_mol%", "M2_mol%", "M3_mol%"]]
    invariant_df = pd.concat([df[other_cols], invariant_df], axis=1)

    return invariant_df

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
    train_pool=None,
    test_pool=None,
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
        train_pool (list): Optional list of catalyst names to use as a pool for drawing training catalysts. Only implemented for split_strategy='catalyst' with conditions={}.
        test_pool (list): Optional list of catalyst names to use as a pool for drawing test catalysts. Only implemented for split_strategy='catalyst' with conditions={}.

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
            if train_pool is None:
                train_pool = all_catalysts
            if test_pool is None:
                test_pool = all_catalysts
            assert len(test_pool) >= n_test_catalysts, \
                f"Not enough catalysts in test_pool ({len(test_pool)}) for the " \
                f"requested split size ({n_test_catalysts})."
            test_catalysts = rng.sample(list(test_pool), n_test_catalysts)
            remaining_catalysts = [c for c in train_pool if c not in test_catalysts]
            assert len(remaining_catalysts) >= n_train_catalysts + n_val_catalysts, \
                f"Not enough catalysts in train_pool ({len(remaining_catalysts)}) for "\
                f"the requested split size ({n_train_catalysts + n_val_catalysts})."
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
            train_catalysts = all_shuffled[: n_train_catalysts + n_val_catalysts]
            train_indices, _ = generate_splits(
                df,
                train_catalyst_names=all_shuffled,
                test_catalyst_names=[],
                conditions=conditions,
                full_catalyst_names=train_catalysts,
                catalyst_name_column=catalyst_name_column,
            )
            if n_test_catalysts > 0:
                test_catalysts = all_shuffled[-n_test_catalysts:]
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


def find_site_permutation_groups(feature_cols, site_prefixes=("M1_", "M2_", "M3_")):
    """Auto-detect column groups that belong to the same metal site and thus need
    to be permuted together when relabeling the sites (e.g. M1 <-> M2).

    A group is any set of columns that share a common suffix after one of the
    site prefixes, e.g. "M1_mol%", "M2_mol%", "M3_mol%" share the suffix "mol%".
    Only suffixes for which *all* site prefixes are present in feature_cols are
    returned.

    Args:
        feature_cols (list): List of feature column names.
        site_prefixes (tuple): Column name prefixes identifying each site, in
            the order sites should be permuted (default: M1_, M2_, M3_).

    Returns:
        tuple: Tuple of tuples of column names, one tuple per detected group,
            ordered like site_prefixes (e.g. (M1_x, M2_x, M3_x)).
    """
    feature_set = set(feature_cols)
    groups = []
    seen_suffixes = set()
    for col in feature_cols:
        if not col.startswith(site_prefixes[0]):
            continue
        suffix = col[len(site_prefixes[0]):]
        if suffix in seen_suffixes:
            continue
        seen_suffixes.add(suffix)
        group = tuple(f"{prefix}{suffix}" for prefix in site_prefixes)
        if all(c in feature_set for c in group):
            groups.append(group)
    return tuple(groups)


def augment_data(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_cols,
    permutation_cols=None,
    other_lists=None,
):
    """
    Augment data by relabeling the metal sites (M1, M2, M3) in all possible ways.
    A data point can have one or more empty metal sites (e.g. a two-metal or
    single-metal catalyst, encoded as all-zero values across the permuted
    columns for that site). Relabelings that only swap sites that are identical
    for a given row (most commonly two or three empty sites) would reproduce a
    row that is already present, so those are not added -- each row gets one
    augmented copy per *distinct* site assignment (1, 3, or 6 depending on how
    many sites are actually distinguishable for that row).

    Parameters:
    - X_train: Training data features
    - y_train: Training data targets
    - X_val: Validation data features (can be None)
    - y_val: Validation data targets (can be None only if X_val is None)
    - X_test: Test data features (can be None)
    - y_test: Test data targets (can be None only if X_test is None)
    - feature_cols: List of feature column names
    - permutation_cols: Tuple of tuples of three column names to permute in all
        possible combinations. The entries in the tuples must be ordered
        identically in all tuples (e.g. all features belonging to M1 at the
        first position, M2 at the second and M3 at the third, like:
        (("M1_atom_number", "M2_atom_number", "M3_atom_number"),
         ("M1_electronegativity", "M2_electronegativity", "M3_electronegativity")).
        If None (default), these groups are auto-detected from feature_cols via
        find_site_permutation_groups.
    - other_lists: Other arrays with shape like y_train that need to be repeated
        (e.g. masks for indexing the training set).

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, repeated_lists,
            group_ids). Since each row can now expand into a different number
            of augmented copies (1, 3, or 6), group_ids is a 3-tuple
            (train_group_ids, val_group_ids, test_group_ids) -- one int array
            per split (None if the corresponding X was None), each the same
            length as the augmented split and giving the index of the original
            (pre-augmentation) row each augmented row came from. Use these
            together with scatter_mean to aggregate e.g. predictions made on
            the augmented data back to one value per original row.
    """
    assert (
        len(feature_cols) == X_train.shape[1]
    ), "Number of feature_cols must match number of columns in X_train"
    n_features = len(feature_cols)

    if permutation_cols is None:
        permutation_cols = find_site_permutation_groups(feature_cols)

    all_permutations = np.array(list(permutations([0, 1, 2])))
    n_perms = len(all_permutations)
    all_indices = np.arange(n_features)[None, :].repeat(n_perms, 0)
    group_col_indices = [
        [feature_cols.index(c) for c in group] for group in permutation_cols
    ]
    for idx1, idx2, idx3 in group_col_indices:
        all_indices[:, [idx1, idx2, idx3]] = all_indices[0, [idx1, idx2, idx3]][
            all_permutations
        ]
    # columns whose values determine whether two site-relabelings of a row are
    # actually distinct (i.e. all M1/M2/M3-site columns being permuted)
    relevant_cols = np.array(group_col_indices).reshape(-1)

    def _augment(X, y, extra=None):
        if X is None:
            return None, None, None, None
        n = X.shape[0]
        variants = X[:, all_indices]  # shape (n, n_perms, n_features)
        if len(relevant_cols) > 0:
            relevant = variants[:, :, relevant_cols]  # (n, n_perms, n_relevant)
            # mark, for each row, permutations that reproduce an earlier one
            duplicate = np.zeros((n, n_perms), dtype=bool)
            for a in range(n_perms):
                for b in range(a + 1, n_perms):
                    duplicate[:, b] |= np.all(
                        relevant[:, a, :] == relevant[:, b, :], axis=1
                    )
            keep_mask = ~duplicate
        else:
            # nothing to permute: only the (unmodified) identity variant is kept
            keep_mask = np.zeros((n, n_perms), dtype=bool)
            keep_mask[:, 0] = True
        flat_keep = keep_mask.reshape(-1)
        X_aug = variants.reshape(-1, n_features)[flat_keep]
        y_aug = np.repeat(y, n_perms)[flat_keep]
        # index of the original (pre-augmentation) row each augmented row came
        # from -- use with scatter_mean to aggregate back per original row
        group_ids = np.repeat(np.arange(n), n_perms)[flat_keep]
        extra_aug = None
        if extra is not None:
            extra_aug = [np.repeat(arr, n_perms)[flat_keep] for arr in extra]
        return X_aug, y_aug, extra_aug, group_ids

    X_train, y_train, repeated_lists, train_group_ids = _augment(
        X_train, y_train, other_lists
    )
    X_val, y_val, _, val_group_ids = _augment(X_val, y_val)
    X_test, y_test, _, test_group_ids = _augment(X_test, y_test)
    group_ids = (train_group_ids, val_group_ids, test_group_ids)
    return X_train, y_train, X_val, y_val, X_test, y_test, repeated_lists, group_ids


def scatter_mean(values, group_ids, n_groups=None):
    """Aggregate values by averaging entries that share the same group id.

    Intended to collapse predictions (or other per-row arrays) computed on
    data produced by augment_data back to one value per original row, since
    the number of augmented copies can now differ per row (see augment_data).

    Args:
        values (np.ndarray): 1D array of per-augmented-row values.
        group_ids (np.ndarray): 1D int array, same length as `values`, giving
            the original row index each entry belongs to (as returned by
            augment_data).
        n_groups (int, optional): Number of groups. Defaults to
            group_ids.max() + 1.

    Returns:
        np.ndarray: 1D array of length n_groups with the mean of `values` per
            group, ordered by group id (i.e. in the original row order).
    """
    values = np.asarray(values)
    group_ids = np.asarray(group_ids)
    if n_groups is None:
        n_groups = int(group_ids.max()) + 1
    sums = np.bincount(group_ids, weights=values, minlength=n_groups)
    counts = np.bincount(group_ids, minlength=n_groups)
    return sums / counts


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
