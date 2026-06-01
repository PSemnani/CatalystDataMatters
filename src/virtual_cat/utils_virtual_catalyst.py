import random
import itertools
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# ==================== FEATURE DEFINITIONS ====================
# Process conditions (features that vary per experiment)
PROCESS_CONDITIONS = [
    "T",
    "Q", 
    "CH4_O2",
    "InertFraction",
]

# Catalyst descriptors (features unique per catalyst)
DESCRIPTORS = [
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
    "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16"
]

# All features
ALL_FEATURES = PROCESS_CONDITIONS + DESCRIPTORS

# Target variable
TARGET = "C2y"

# Catalyst identifier column
CATALYST_ID = "cat_ID"


# ==================== TEMPERATURE CONDITIONS ====================
# For conditional testing in stage 2
CONDITIONS_TEMP_SINGLE = {
    "650": {"T": {"==": 650}},
}

# Can expand this later based on your temperature ranges
CONDITIONS_TEMP_PAIRS = {
    # Add your temperature pairs here if needed
}


# ==================== XGB HYPERPARAMETER GRID ====================
# Grid for hyperparameter search
xgb_grid_params = {
    "n_estimators": [100, 350, 700],
    "learning_rate": [0.02, 0.05, 0.1, 0.3],
    "max_depth": [6, 10],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.5],
    "reg_lambda": [0.5, 1.0, 2.0],
    "gamma": [0.0, 1.0, 10.0],
    "min_child_weight": [1, 2, 3],
}


# ==================== MODEL CONFIGURATIONS ====================
def optimized_xgb(seed: int) -> xgb.XGBRegressor:
    """Optimized XGBoost configuration"""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=seed,
        n_estimators=350,
        learning_rate=0.1,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=2,
    )


def default_xgb(seed: int) -> xgb.XGBRegressor:
    """Default XGBoost configuration"""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=seed,
        n_estimators=700,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1,
    )


def get_random_xgb_params(n_samples: int = 50, seed: int = 42):
    """Generate random hyperparameter combinations for grid search"""
    rng = random.Random(seed)
    param_combinations = []
    
    for _ in range(n_samples):
        params = {
            key: rng.choice(values) 
            for key, values in xgb_grid_params.items()
        }
        param_combinations.append(params)
    
    return param_combinations


# ==================== DATA SPLITTING ====================
def split_data_catalyst(
    df,
    feature_cols,
    target_col,
    rng,
    n_train_catalysts,
    n_val_catalysts=0,
    n_test_catalysts=10,
    conditions=None,
):
    """
    Split data by catalyst (entire catalysts go to train/val/test).
    
    Args:
        df: DataFrame with data
        feature_cols: List of feature column names
        target_col: Target column name
        rng: Random number generator
        n_train_catalysts: Number of catalysts for training
        n_val_catalysts: Number of catalysts for validation (0 = no validation set)
        n_test_catalysts: Number of catalysts for testing
        conditions: Optional dict of conditions for filtering (for stage 2)
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, train_indices, val_indices, test_indices
    """
    # Get unique catalysts
    unique_catalysts = df[CATALYST_ID].unique()
    n_catalysts = len(unique_catalysts)
    
    # Shuffle catalysts
    catalysts_shuffled = list(unique_catalysts)
    rng.shuffle(catalysts_shuffled)
    
    # Split catalysts
    test_catalysts = catalysts_shuffled[:n_test_catalysts]
    remaining = catalysts_shuffled[n_test_catalysts:]
    
    if n_val_catalysts > 0:
        val_catalysts = remaining[:n_val_catalysts]
        train_catalysts = remaining[n_val_catalysts:n_val_catalysts + n_train_catalysts]
    else:
        val_catalysts = []
        train_catalysts = remaining[:n_train_catalysts]
    
    # Get indices for each split
    train_mask = df[CATALYST_ID].isin(train_catalysts)
    test_mask = df[CATALYST_ID].isin(test_catalysts)
    
    if n_val_catalysts > 0:
        val_mask = df[CATALYST_ID].isin(val_catalysts)
        val_indices = df.index[val_mask]
    else:
        val_indices = None
    
    train_indices = df.index[train_mask]
    test_indices = df.index[test_mask]
    
    # Extract features and targets
    X_train = df.loc[train_indices, feature_cols].values
    y_train = df.loc[train_indices, target_col].values
    
    X_test = df.loc[test_indices, feature_cols].values
    y_test = df.loc[test_indices, target_col].values
    
    if n_val_catalysts > 0:
        X_val = df.loc[val_indices, feature_cols].values
        y_val = df.loc[val_indices, target_col].values
    else:
        X_val = None
        y_val = None
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_indices, val_indices, test_indices


def split_data_random(
    df,
    feature_cols,
    target_col,
    rng,
    train_fraction=0.8,
):
    """
    Random 80-20 split of all data points (baseline comparison).
    
    Args:
        df: DataFrame with data
        feature_cols: List of feature column names
        target_col: Target column name
        rng: Random number generator
        train_fraction: Fraction of data for training (default 0.8)
    
    Returns:
        X_train, y_train, X_test, y_test, train_indices, test_indices
    """
    # Shuffle all indices
    all_indices = df.index.tolist()
    rng.shuffle(all_indices)
    
    # Split by fraction
    n_train = int(len(all_indices) * train_fraction)
    train_indices = all_indices[:n_train]
    test_indices = all_indices[n_train:]
    
    # Extract features and targets
    X_train = df.loc[train_indices, feature_cols].values
    y_train = df.loc[train_indices, target_col].values
    
    X_test = df.loc[test_indices, feature_cols].values
    y_test = df.loc[test_indices, target_col].values
    
    return X_train, y_train, X_test, y_test, train_indices, test_indices


# ==================== SCALING ====================
def scale_data(X_train, X_val, X_test, feature_cols):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train, X_val, X_test: Feature arrays
        feature_cols: List of feature column names
    
    Returns:
        Scaled X_train, X_val, X_test, and the scaler object
    """
    # Build column transformer that scales each column
    ct_transformers = []
    for i in range(X_train.shape[1]):
        ct_transformers.append((f"scaler_{i}", StandardScaler(), [i]))
    
    col_transformer = ColumnTransformer(
        transformers=ct_transformers, 
        remainder="drop"
    )
    
    scaler = col_transformer
    
    # Fit on training data and transform all sets
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if X_val is not None:
        X_val = scaler.transform(X_val)
    
    return X_train, X_val, X_test, scaler


# ==================== HELPER FUNCTIONS ====================
def compute_metrics(y_true, y_pred):
    """Compute regression metrics"""
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))
    
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
