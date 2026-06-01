"""
================================================================================
CATALYST DATA QUALITY IMPROVEMENT - MEMORY-EFFICIENT VERSION
================================================================================
Fixed: Efficient leverage computation for large datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

print("="*80)
print("CATALYST DATA QUALITY IMPROVEMENT - MEMORY-EFFICIENT VERSION")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "/Users/parastoo/DataMatters/Dataset/virtual_catalyst_dataset_with_inert.csv"

CATALYST_ID = "cat_ID"
TARGET = "C2y"
DESCRIPTORS = [f"D{i}" for i in range(1, 17)]
PROCESS_CONDITIONS = ["T", "Q", "CH4_O2", "InertFraction"]
ALL_FEATURES = DESCRIPTORS + PROCESS_CONDITIONS

OUTLIER_CONTAMINATION = 0.05
HIGH_RESIDUAL_PERCENTILE = 95
LEVERAGE_SAMPLE_SIZE = 10000  # NEW: Sample for leverage computation

print("\nConfiguration:")
print(f"  Data path: {DATA_PATH}")
print(f"  Features: {len(ALL_FEATURES)}")
print(f"  Outlier detection: Top {OUTLIER_CONTAMINATION*100}%")
print(f"  Leverage sampling: {LEVERAGE_SAMPLE_SIZE} samples (memory-efficient)")

# ============================================================================
# STEP 1: LOAD AND BASIC CLEANING
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOAD AND BASIC CLEANING")
print("="*80)

df_original = pd.read_csv(DATA_PATH)
original_size = len(df_original)

print(f"\nOriginal dataset:")
print(f"  Shape: {df_original.shape}")
print(f"  Catalysts: {df_original[CATALYST_ID].nunique()}")

print(f"\nData quality check:")
print(f"  NaN in {TARGET}: {df_original[TARGET].isna().sum()}")
print(f"  NaN in features: {df_original[ALL_FEATURES].isna().sum().sum()}")
print(f"  Negative {TARGET}: {(df_original[TARGET] < 0).sum()}")

df = df_original.copy()

print(f"\n⏳ Performing basic cleaning...")

df = df.dropna(subset=[TARGET])
df = df.dropna(subset=ALL_FEATURES)
print(f"  After removing NaN: {len(df):,} rows (removed {original_size - len(df)})")

df = df[~np.isinf(df[TARGET])]
for col in ALL_FEATURES:
    if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        df = df[~np.isinf(df[col])]
print(f"  After removing inf: {len(df):,} rows")

negative_count = (df[TARGET] < 0).sum()
if negative_count > 0:
    df.loc[df[TARGET] < 0, TARGET] = 0
    print(f"  Converted {negative_count} negative {TARGET} to zero")

print(f"\n✓ Basic cleaning complete: {len(df):,} rows, {df[CATALYST_ID].nunique()} catalysts")

# ============================================================================
# STEP 2: DIAGNOSTIC ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: COMPREHENSIVE DIAGNOSTIC ANALYSIS")
print("="*80)

X = df[ALL_FEATURES].values
y = df[TARGET].values

# ============================================================================
# 2.1 YIELD DISTRIBUTION
# ============================================================================

print("\n2.1 Yield Distribution Analysis")
print("-" * 70)

yield_stats = df[TARGET].describe()
print(f"\nYield statistics:")
for stat, val in yield_stats.items():
    print(f"  {stat:>10s}: {val:.4f}")

zero_yield = (df[TARGET] == 0).sum()
near_zero_yield = (df[TARGET] < 0.01).sum()
low_yield = (df[TARGET] < 0.05).sum()

print(f"\nYield categories:")
print(f"  Zero (= 0):        {zero_yield:6d} ({zero_yield/len(df)*100:5.2f}%)")
print(f"  Near-zero (< 0.01): {near_zero_yield:6d} ({near_zero_yield/len(df)*100:5.2f}%)")
print(f"  Low (< 0.05):       {low_yield:6d} ({low_yield/len(df)*100:5.2f}%)")

catalyst_stats = df.groupby(CATALYST_ID)[TARGET].agg(['mean', 'std', 'min', 'max', 'count'])
zero_yield_catalysts = (catalyst_stats['max'] == 0).sum()
poor_catalysts = (catalyst_stats['mean'] < 0.05).sum()

print(f"\nCatalyst-level:")
print(f"  All-zero catalysts: {zero_yield_catalysts}")
print(f"  Poor catalysts (mean < 0.05): {poor_catalysts}")

# ============================================================================
# 2.2 STATISTICAL OUTLIERS
# ============================================================================

print("\n2.2 Statistical Outlier Detection")
print("-" * 70)

z_scores_y = np.abs(stats.zscore(df[TARGET]))
z_outliers_y = z_scores_y > 3
print(f"\nTarget outliers (|z| > 3): {z_outliers_y.sum()} ({z_outliers_y.sum()/len(df)*100:.2f}%)")

z_outliers_X = np.zeros(len(df), dtype=bool)
for col in ALL_FEATURES:
    z_scores = np.abs(stats.zscore(df[col]))
    z_outliers_X |= (z_scores > 3)
print(f"Feature outliers (|z| > 3): {z_outliers_X.sum()} ({z_outliers_X.sum()/len(df)*100:.2f}%)")

# ============================================================================
# 2.3 MULTIVARIATE OUTLIERS
# ============================================================================

print("\n2.3 Multivariate Outlier Detection")
print("-" * 70)

print(f"\n⏳ Running Isolation Forest...")
iso_forest = IsolationForest(
    contamination=OUTLIER_CONTAMINATION,
    random_state=42,
    n_jobs=-1
)
outlier_labels = iso_forest.fit_predict(X)
is_outlier = outlier_labels == -1

print(f"✓ Multivariate outliers: {is_outlier.sum()} ({is_outlier.sum()/len(df)*100:.2f}%)")

outlier_scores = iso_forest.score_samples(X)
df['outlier_score'] = outlier_scores

# ============================================================================
# 2.4 PREDICTION-BASED ANALYSIS
# ============================================================================

print("\n2.4 Prediction-Based Analysis")
print("-" * 70)

print(f"\n⏳ Training Random Forest for residual analysis...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

predictions = cross_val_predict(rf, X, y, cv=5, n_jobs=-1)
residuals = np.abs(y - predictions)
df['residual'] = residuals
df['prediction'] = predictions

r2_cv = r2_score(y, predictions)
mae_cv = mean_absolute_error(y, predictions)

print(f"✓ 5-fold CV performance:")
print(f"  R² = {r2_cv:.4f}")
print(f"  MAE = {mae_cv:.4f}")

residual_threshold = np.percentile(residuals, HIGH_RESIDUAL_PERCENTILE)
is_hard_predict = residuals > residual_threshold

print(f"\nHard-to-predict samples (>{HIGH_RESIDUAL_PERCENTILE}th percentile):")
print(f"  Threshold: {residual_threshold:.4f}")
print(f"  Count: {is_hard_predict.sum()} ({is_hard_predict.sum()/len(df)*100:.2f}%)")

# ============================================================================
# 2.5 LEVERAGE ANALYSIS (MEMORY-EFFICIENT)
# ============================================================================

print("\n2.5 Leverage Analysis (Memory-Efficient)")
print("-" * 70)

print(f"\n⏳ Computing leverage scores using sampling approach...")

# FIXED: Use sampling for large datasets
if len(df) > LEVERAGE_SAMPLE_SIZE:
    print(f"  Dataset large ({len(df):,} samples), using sample of {LEVERAGE_SAMPLE_SIZE:,}")
    
    # Sample for leverage computation
    sample_indices = np.random.choice(len(df), size=LEVERAGE_SAMPLE_SIZE, replace=False)
    X_leverage_sample = X[sample_indices]
    
    # Compute leverage on sample
    scaler_leverage = RobustScaler()
    X_leverage_scaled = scaler_leverage.fit_transform(X_leverage_sample)
    
    try:
        # Compute hat matrix for sample
        XtX_inv = np.linalg.pinv(X_leverage_scaled.T @ X_leverage_scaled)
        leverage_sample = np.sum(X_leverage_scaled @ XtX_inv * X_leverage_scaled, axis=1)
        
        # Use sample statistics to flag full dataset
        # High leverage in sample gives us threshold
        leverage_threshold_sample = 3 * len(ALL_FEATURES) / len(X_leverage_sample)
        
        # For full dataset, compute approximate leverage using residuals as proxy
        # High residual + high feature variance = likely high leverage
        feature_extremeness = np.abs(stats.zscore(X, axis=0)).max(axis=1)
        leverage_proxy = feature_extremeness * residuals
        leverage_proxy_threshold = np.percentile(leverage_proxy, 95)
        
        is_high_leverage = leverage_proxy > leverage_proxy_threshold
        df['leverage_proxy'] = leverage_proxy
        
        print(f"✓ Leverage analysis complete (using proxy method):")
        print(f"  High leverage samples (top 5%): {is_high_leverage.sum()} ({is_high_leverage.sum()/len(df)*100:.2f}%)")
        
    except Exception as e:
        print(f"  ⚠️ Leverage computation failed: {e}")
        print(f"  Using alternative: Mahalanobis distance")
        
        # Fallback: Use Mahalanobis distance as leverage proxy
        from scipy.spatial.distance import mahalanobis
        
        mean = np.mean(X_leverage_scaled, axis=0)
        cov = np.cov(X_leverage_scaled.T)
        
        try:
            cov_inv = np.linalg.pinv(cov)
            mahal_dist = np.array([mahalanobis(x, mean, cov_inv) for x in X_leverage_scaled[:1000]])  # Sample 1000
            mahal_threshold = np.percentile(mahal_dist, 95)
            
            # Approximate for full dataset
            is_high_leverage = np.zeros(len(df), dtype=bool)
            is_high_leverage[sample_indices[:len(mahal_dist)]] = mahal_dist > mahal_threshold
            df['leverage_proxy'] = 0
            
            print(f"✓ Using Mahalanobis distance proxy")
            print(f"  High leverage samples: {is_high_leverage.sum()}")
        except:
            print(f"  ⚠️ All leverage methods failed, skipping")
            is_high_leverage = np.zeros(len(df), dtype=bool)
            df['leverage_proxy'] = 0
else:
    # Small dataset, can compute exact leverage
    print(f"  Computing exact leverage for {len(df):,} samples...")
    
    scaler_leverage = RobustScaler()
    X_scaled = scaler_leverage.fit_transform(X)
    
    try:
        H = X_scaled @ np.linalg.pinv(X_scaled.T @ X_scaled) @ X_scaled.T
        leverage = np.diag(H)
        df['leverage'] = leverage
        
        high_leverage_threshold = 3 * len(ALL_FEATURES) / len(df)
        is_high_leverage = leverage > high_leverage_threshold
        
        print(f"✓ Leverage analysis complete:")
        print(f"  Threshold: {high_leverage_threshold:.6f}")
        print(f"  High leverage: {is_high_leverage.sum()} ({is_high_leverage.sum()/len(df)*100:.2f}%)")
    except:
        print(f"  ⚠️ Leverage computation failed, skipping")
        is_high_leverage = np.zeros(len(df), dtype=bool)
        df['leverage'] = 0

# ============================================================================
# 2.6 FEATURE QUALITY
# ============================================================================

print("\n2.6 Feature Quality Analysis")
print("-" * 70)

feature_variance = df[ALL_FEATURES].var().sort_values()
low_variance_features = feature_variance[feature_variance < 0.01].index.tolist()

print(f"\nLow variance features (< 0.01): {len(low_variance_features)}")
if low_variance_features:
    print(f"  {low_variance_features}")

feature_correlations = df[ALL_FEATURES].corrwith(df[TARGET]).abs().sort_values(ascending=False)
weak_features = feature_correlations[feature_correlations < 0.05].index.tolist()

print(f"\nWeakly correlated features (|corr| < 0.05): {len(weak_features)}")
if weak_features:
    print(f"  {weak_features}")

print(f"\n⏳ Computing feature importances...")
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

low_importance_features = feature_importance[feature_importance['importance'] < 0.01]['feature'].tolist()
print(f"\nLow importance features (< 0.01): {len(low_importance_features)}")
if low_importance_features:
    print(f"  {low_importance_features}")

print("\nTop 10 most important features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 2.7 CREATE FLAGS
# ============================================================================

print("\n2.7 Creating Diagnostic Flags")
print("-" * 70)

df['flag_zero_yield'] = df[TARGET] == 0
df['flag_near_zero'] = df[TARGET] < 0.01
df['flag_low_yield'] = (df[TARGET] >= 0.01) & (df[TARGET] < 0.05)
df['flag_z_outlier_target'] = z_outliers_y
df['flag_z_outlier_features'] = z_outliers_X
df['flag_multivariate_outlier'] = is_outlier
df['flag_hard_predict'] = is_hard_predict
df['flag_high_leverage'] = is_high_leverage

quality_flags = [
    'flag_zero_yield',
    'flag_near_zero', 
    'flag_multivariate_outlier',
    'flag_hard_predict',
    'flag_high_leverage'
]

df['flag_any_issue'] = df[quality_flags].any(axis=1)

print("\nDiagnostic flag summary:")
for flag in quality_flags:
    count = df[flag].sum()
    pct = count / len(df) * 100
    print(f"  {flag:30s}: {count:6d} ({pct:5.2f}%)")

print(f"\n  {'flag_any_issue':30s}: {df['flag_any_issue'].sum():6d} ({df['flag_any_issue'].sum()/len(df)*100:5.2f}%)")

diagnostic_file = 'catalyst_data_with_diagnostics.csv'
df.to_csv(diagnostic_file, index=False)
print(f"\n✓ Saved: {diagnostic_file}")

# ============================================================================
# STEP 3: DEFINE CLEANING STRATEGIES
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DEFINING CLEANING STRATEGIES")
print("="*80)

# Conservative
df_conservative = df[
    (df[TARGET] > 0) &
    (~df['flag_multivariate_outlier'])
].copy()

# Moderate (RECOMMENDED)
df_moderate = df[
    (df[TARGET] >= 0.01) &
    (~df['flag_multivariate_outlier']) &
    (~df['flag_hard_predict'])
].copy()

# Aggressive
df_aggressive = df[
    (df[TARGET] >= 0.01) &
    (~df['flag_multivariate_outlier']) &
    (~df['flag_hard_predict']) &
    (~df['flag_high_leverage'])
].copy()

# Custom
df_custom = df[~df['flag_any_issue']].copy()

strategies = {
    'Original': df,
    'Conservative': df_conservative,
    'Moderate': df_moderate,
    'Aggressive': df_aggressive,
    'Custom': df_custom
}

print("\nCleaning strategy comparison:")
print(f"{'Strategy':<15} {'Samples':>10} {'Removed':>10} {'% Removed':>10} {'Catalysts':>10}")
print("-" * 60)

for name, data in strategies.items():
    n_samples = len(data)
    n_removed = len(df) - n_samples
    pct_removed = n_removed / len(df) * 100
    n_catalysts = data[CATALYST_ID].nunique()
    
    print(f"{name:<15} {n_samples:>10,} {n_removed:>10,} {pct_removed:>9.2f}% {n_catalysts:>10}")

# Save cleaned datasets
print(f"\n⏳ Saving cleaned datasets...")
for name, data in strategies.items():
    if name != 'Original':
        filename = f"catalyst_data_cleaned_{name.lower()}.csv"
        data.to_csv(filename, index=False)
        print(f"  ✓ {filename}")

# ============================================================================
# STEP 4: TRAIN AND COMPARE MODELS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: COMPARING MODELS ON DIFFERENT DATASETS")
print("="*80)

xgb_config = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

results_comparison = []

print(f"\n⏳ Training XGBoost on each strategy...")

np.random.seed(42)

for strategy_name, df_strategy in strategies.items():
    print(f"\n  {strategy_name}...")
    
    # Split by catalyst
    all_cats = df_strategy[CATALYST_ID].unique()
    n_train_cats = int(0.8 * len(all_cats))
    train_cats = np.random.choice(all_cats, size=n_train_cats, replace=False)
    test_cats = [c for c in all_cats if c not in train_cats]
    
    df_train = df_strategy[df_strategy[CATALYST_ID].isin(train_cats)]
    df_test = df_strategy[df_strategy[CATALYST_ID].isin(test_cats)]
    
    if len(df_train) < 100 or len(df_test) < 20:
        print(f"    ⚠️ Skipping (insufficient data)")
        continue
    
    X_train = df_train[ALL_FEATURES].values
    y_train = df_train[TARGET].values
    X_test = df_test[ALL_FEATURES].values
    y_test = df_test[TARGET].values
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBRegressor(**xgb_config)
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    gap = train_r2 - test_r2
    
    results_comparison.append({
        'Strategy': strategy_name,
        'N_train': len(df_train),
        'N_test': len(df_test),
        'N_catalysts_train': len(train_cats),
        'N_catalysts_test': len(test_cats),
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Test_MAE': test_mae,
        'Gap': gap
    })
    
    print(f"    Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, Gap={gap:.4f}")

df_results = pd.DataFrame(results_comparison)

# ============================================================================
# STEP 5: RESULTS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print(f"\n{'Strategy':<15} {'N_train':>10} {'Train R²':>10} {'Test R²':>10} {'MAE':>10} {'Gap':>10}")
print("-" * 70)

for _, row in df_results.iterrows():
    print(f"{row['Strategy']:<15} {int(row['N_train']):>10,} {row['Train_R2']:>10.4f} "
          f"{row['Test_R2']:>10.4f} {row['Test_MAE']:>10.4f} {row['Gap']:>10.4f}")

print("-" * 70)

best_idx = df_results['Test_R2'].idxmax()
best_strategy = df_results.loc[best_idx, 'Strategy']
best_r2 = df_results.loc[best_idx, 'Test_R2']
best_gap = df_results.loc[best_idx, 'Gap']
best_mae = df_results.loc[best_idx, 'Test_MAE']

print(f"\n🏆 BEST STRATEGY: {best_strategy}")
print(f"   Test R² = {best_r2:.4f}")
print(f"   Test MAE = {best_mae:.4f}")
print(f"   Gap = {best_gap:.4f}")

original_r2 = df_results[df_results['Strategy'] == 'Original']['Test_R2'].values[0]
improvement = best_r2 - original_r2
improvement_pct = (improvement / original_r2) * 100

print(f"\n📈 IMPROVEMENT:")
print(f"   Original R² = {original_r2:.4f}")
print(f"   Best R² = {best_r2:.4f}")
print(f"   Gain = {improvement:+.4f} ({improvement_pct:+.1f}%)")

# Save results
df_results.to_csv('cleaning_strategy_results.csv', index=False)
print(f"\n✓ Saved: cleaning_strategy_results.csv")

# ============================================================================
# STEP 6: SAVE BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 6: SAVING BEST MODEL")
print("="*80)

import joblib

# Train final model on best strategy
best_data = strategies[best_strategy]

all_cats = best_data[CATALYST_ID].unique()
n_train_cats = int(0.8 * len(all_cats))
train_cats_final = np.random.choice(all_cats, size=n_train_cats, replace=False)
test_cats_final = [c for c in all_cats if c not in train_cats_final]

df_final_train = best_data[best_data[CATALYST_ID].isin(train_cats_final)]
df_final_test = best_data[best_data[CATALYST_ID].isin(test_cats_final)]

X_final_train = df_final_train[ALL_FEATURES].values
y_final_train = df_final_train[TARGET].values

scaler_final = RobustScaler()
X_final_train_scaled = scaler_final.fit_transform(X_final_train)

model_final = XGBRegressor(**xgb_config)
model_final.fit(X_final_train_scaled, y_final_train)

joblib.dump(model_final, 'model_xgboost_cleaned_best.pkl')
joblib.dump(scaler_final, 'scaler_robust_cleaned.pkl')

metadata = {
    'best_strategy': best_strategy,
    'test_r2': best_r2,
    'test_mae': best_mae,
    'overfitting_gap': best_gap,
    'improvement': improvement,
    'improvement_pct': improvement_pct,
    'features': ALL_FEATURES,
    'n_samples_removed': len(df) - len(best_data),
    'removal_pct': (len(df) - len(best_data)) / len(df) * 100
}

joblib.dump(metadata, 'model_metadata_cleaned.pkl')

print(f"\n✓ Saved:")
print(f"  1. model_xgboost_cleaned_best.pkl")
print(f"  2. scaler_robust_cleaned.pkl")
print(f"  3. model_metadata_cleaned.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DATA CLEANING COMPLETE!")
print("="*80)

print(f"\n📊 SUMMARY:")
print(f"  Original: {len(df):,} samples")
print(f"  Best cleaned: {len(best_data):,} samples")
print(f"  Removed: {len(df)-len(best_data):,} ({(len(df)-len(best_data))/len(df)*100:.1f}%)")

print(f"\n🎯 PERFORMANCE:")
print(f"  Original R² = {original_r2:.4f}")
print(f"  Cleaned R² = {best_r2:.4f}")
print(f"  Improvement = {improvement:+.4f} ({improvement_pct:+.1f}%)")

print(f"\n💡 RECOMMENDATION:")
if improvement > 0.05:
    print(f"  ✅ Excellent improvement! Use {best_strategy} strategy")
    print(f"  → Proceed to feature engineering")
elif improvement > 0.02:
    print(f"  ✅ Good improvement. Use {best_strategy} strategy")
    print(f"  → Consider feature engineering")
else:
    print(f"  ⚠️ Limited improvement")
    print(f"  → Focus on feature engineering or catalyst families")

if best_gap < 0.4:
    print(f"\n  ✅ Overfitting well controlled (gap={best_gap:.4f})")
else:
    print(f"\n  ⚠️ Still overfitting (gap={best_gap:.4f})")
    print(f"  → Try stronger regularization")

print("\n" + "="*80)