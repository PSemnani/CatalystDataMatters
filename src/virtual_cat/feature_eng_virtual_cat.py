"""
TIER 2: FEATURE ENGINEERING ON CLEANED DATA
Goal: Push from R² = 0.70 to R² = 0.75-0.80

Based on your feature importance:
- D11 (19.7%) and D1 (17.5%) are critical
- Temperature T (12.5%) is important
- D10, D8, Q, InertFraction are weak (<1%)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("="*80)
print("TIER 2: FEATURE ENGINEERING ON CLEANED DATA")
print("="*80)

# ============================================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================================

print("\nSTEP 1: Loading best cleaned dataset...")

# Load the Conservative cleaned data (your best performer)
df = pd.read_csv('catalyst_data_cleaned_conservative.csv')

CATALYST_ID = "cat_ID"
TARGET = "C2y"
DESCRIPTORS = [f"D{i}" for i in range(1, 17)]
PROCESS_CONDITIONS = ["T", "Q", "CH4_O2", "InertFraction"]

print(f"Loaded: {len(df):,} samples, {df[CATALYST_ID].nunique()} catalysts")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

print("\nSTEP 2: Creating engineered features...")

df_eng = df.copy()

# Top features from your analysis
TOP_DESCRIPTORS = ['D11', 'D1', 'D16', 'D15', 'D2', 'D7']
WEAK_FEATURES = ['D10', 'D8', 'Q', 'InertFraction']

print(f"\nTop descriptors: {TOP_DESCRIPTORS}")
print(f"Weak features: {WEAK_FEATURES}")

# ----------------------------------------------------------------------
# 2.1 INTERACTIONS between top descriptors
# ----------------------------------------------------------------------

print("\n2.1 Creating pairwise interactions...")

interactions = [
    ('D11', 'D1'),   # Top 2 features
    ('D11', 'D16'),
    ('D11', 'D15'),
    ('D1', 'D16'),
    ('D1', 'D15'),
    ('D16', 'D15'),
    ('D11', 'T'),    # Descriptor × Temperature
    ('D1', 'T'),
    ('D7', 'T'),
]

interaction_features = []
for d1, d2 in interactions:
    # Multiplication
    feat_name = f'{d1}×{d2}'
    df_eng[feat_name] = df[d1] * df[d2]
    interaction_features.append(feat_name)
    
    # Division (with safe denominator)
    feat_name = f'{d1}÷{d2}'
    df_eng[feat_name] = df[d1] / (df[d2] + 1e-6)
    interaction_features.append(feat_name)

print(f"Created {len(interaction_features)} interaction features")

# ----------------------------------------------------------------------
# 2.2 POLYNOMIAL features from top descriptors
# ----------------------------------------------------------------------

print("\n2.2 Creating polynomial features...")

polynomial_features = []
for desc in ['D11', 'D1', 'D16', 'D15']:
    # Square
    feat_name = f'{desc}²'
    df_eng[feat_name] = df[desc] ** 2
    polynomial_features.append(feat_name)
    
    # Cube
    feat_name = f'{desc}³'
    df_eng[feat_name] = df[desc] ** 3
    polynomial_features.append(feat_name)
    
    # Square root (absolute value to handle negatives)
    feat_name = f'√{desc}'
    df_eng[feat_name] = np.sqrt(np.abs(df[desc])) * np.sign(df[desc])
    polynomial_features.append(feat_name)

print(f"Created {len(polynomial_features)} polynomial features")

# ----------------------------------------------------------------------
# 2.3 AGGREGATE statistics over all descriptors
# ----------------------------------------------------------------------

print("\n2.3 Creating aggregate statistics...")

desc_matrix = df[DESCRIPTORS].values
aggregate_features = []

features_to_add = {
    'DESC_mean': desc_matrix.mean(axis=1),
    'DESC_std': desc_matrix.std(axis=1),
    'DESC_max': desc_matrix.max(axis=1),
    'DESC_min': desc_matrix.min(axis=1),
    'DESC_range': desc_matrix.max(axis=1) - desc_matrix.min(axis=1),
    'DESC_median': np.median(desc_matrix, axis=1),
}

for feat_name, values in features_to_add.items():
    df_eng[feat_name] = values
    aggregate_features.append(feat_name)

print(f"Created {len(aggregate_features)} aggregate features")

# ----------------------------------------------------------------------
# 2.4 TEMPERATURE-based features
# ----------------------------------------------------------------------

print("\n2.4 Creating temperature features...")

temperature_features = []

# Temperature zones (low/medium/high)
df_eng['T_zone'] = pd.cut(df['T'], bins=3, labels=[0, 1, 2]).astype(float)
temperature_features.append('T_zone')

# Temperature squared (often important in chemical kinetics)
df_eng['T²'] = df['T'] ** 2
temperature_features.append('T²')

# Temperature × key descriptors (already in interactions)
# Arrhenius-like feature: 1/T
df_eng['1/T'] = 1 / (df['T'] + 1e-6)
temperature_features.append('1/T')

print(f"Created {len(temperature_features)} temperature features")

# ----------------------------------------------------------------------
# 2.5 RATIO features
# ----------------------------------------------------------------------

print("\n2.5 Creating ratio features...")

ratio_features = []

# Process condition ratios
df_eng['T/CH4_O2'] = df['T'] / (df['CH4_O2'] + 1e-6)
ratio_features.append('T/CH4_O2')

df_eng['CH4_O2×T'] = df['CH4_O2'] * df['T']
ratio_features.append('CH4_O2×T')

# Descriptor dominance (top descriptor / mean of all)
df_eng['D11_dominance'] = df['D11'] / (df_eng['DESC_mean'] + 1e-6)
ratio_features.append('D11_dominance')

df_eng['D1_dominance'] = df['D1'] / (df_eng['DESC_mean'] + 1e-6)
ratio_features.append('D1_dominance')

print(f"Created {len(ratio_features)} ratio features")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

all_new_features = (interaction_features + polynomial_features + 
                    aggregate_features + temperature_features + ratio_features)

print(f"\n✓ Total new features created: {len(all_new_features)}")
print(f"  - Interactions: {len(interaction_features)}")
print(f"  - Polynomials: {len(polynomial_features)}")
print(f"  - Aggregates: {len(aggregate_features)}")
print(f"  - Temperature: {len(temperature_features)}")
print(f"  - Ratios: {len(ratio_features)}")

# ============================================================================
# STEP 3: FEATURE SELECTION
# ============================================================================

print("\n" + "="*80)
print("STEP 3: TESTING DIFFERENT FEATURE SETS")
print("="*80)

# Define feature sets to test
feature_sets = {
    'Original': DESCRIPTORS + PROCESS_CONDITIONS,
    'Without_Weak': [f for f in DESCRIPTORS + PROCESS_CONDITIONS if f not in WEAK_FEATURES],
    'Top_Only': TOP_DESCRIPTORS + ['T', 'CH4_O2'],
    'Original+Interactions': DESCRIPTORS + PROCESS_CONDITIONS + interaction_features,
    'Original+Polynomials': DESCRIPTORS + PROCESS_CONDITIONS + polynomial_features,
    'Original+All_Engineered': DESCRIPTORS + PROCESS_CONDITIONS + all_new_features,
    'Top+All_Engineered': TOP_DESCRIPTORS + ['T', 'CH4_O2'] + all_new_features,
}

# Train-test split
np.random.seed(42)
all_cats = df[CATALYST_ID].unique()
n_train_cats = int(0.8 * len(all_cats))
train_cats = np.random.choice(all_cats, size=n_train_cats, replace=False)
test_cats = [c for c in all_cats if c not in train_cats]

df_train = df_eng[df_eng[CATALYST_ID].isin(train_cats)]
df_test = df_eng[df_eng[CATALYST_ID].isin(test_cats)]

y_train = df_train[TARGET].values
y_test = df_test[TARGET].values

# XGBoost configuration
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

results = []

print(f"\n⏳ Training models on {len(feature_sets)} feature sets...")

for set_name, features in feature_sets.items():
    print(f"\n  {set_name} ({len(features)} features)...")
    
    # Extract features
    X_train = df_train[features].values
    X_test = df_test[features].values
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = XGBRegressor(**xgb_config)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    gap = train_r2 - test_r2
    
    results.append({
        'Feature_Set': set_name,
        'N_Features': len(features),
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Test_MAE': test_mae,
        'Gap': gap
    })
    
    print(f"    Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, Gap={gap:.4f}")

df_results = pd.DataFrame(results)

# ============================================================================
# STEP 4: RESULTS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING RESULTS")
print("="*80)

print(f"\n{'Feature Set':<25} {'N_Features':>12} {'Train R²':>10} {'Test R²':>10} {'MAE':>10} {'Gap':>10}")
print("-" * 85)

for _, row in df_results.iterrows():
    print(f"{row['Feature_Set']:<25} {int(row['N_Features']):>12} {row['Train_R2']:>10.4f} "
          f"{row['Test_R2']:>10.4f} {row['Test_MAE']:>10.4f} {row['Gap']:>10.4f}")

print("-" * 85)

# Find best
best_idx = df_results['Test_R2'].idxmax()
best_set = df_results.loc[best_idx, 'Feature_Set']
best_r2 = df_results.loc[best_idx, 'Test_R2']
best_mae = df_results.loc[best_idx, 'Test_MAE']
best_gap = df_results.loc[best_idx, 'Gap']

print(f"\n🏆 BEST FEATURE SET: {best_set}")
print(f"   Test R² = {best_r2:.4f}")
print(f"   Test MAE = {best_mae:.4f}")
print(f"   Gap = {best_gap:.4f}")

# Compare to baseline
baseline_r2 = df_results[df_results['Feature_Set'] == 'Original']['Test_R2'].values[0]
improvement = best_r2 - baseline_r2
improvement_pct = (improvement / baseline_r2) * 100

print(f"\n📈 IMPROVEMENT OVER BASELINE:")
print(f"   Baseline (Original features) = {baseline_r2:.4f}")
print(f"   Best (with engineering) = {best_r2:.4f}")
print(f"   Gain = {improvement:+.4f} ({improvement_pct:+.1f}%)")

# ============================================================================
# STEP 5: SAVE BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SAVING BEST MODEL")
print("="*80)

# Train final model with best features
best_features = feature_sets[best_set]

X_train_best = df_train[best_features].values
X_test_best = df_test[best_features].values

scaler_best = RobustScaler()
X_train_best_scaled = scaler_best.fit_transform(X_train_best)
X_test_best_scaled = scaler_best.transform(X_test_best)

model_best = XGBRegressor(**xgb_config)
model_best.fit(X_train_best_scaled, y_train)

# Save
joblib.dump(model_best, 'model_xgboost_final_engineered.pkl')
joblib.dump(scaler_best, 'scaler_final_engineered.pkl')

# Save feature list
metadata_final = {
    'best_feature_set': best_set,
    'features': best_features,
    'n_features': len(best_features),
    'test_r2': best_r2,
    'test_mae': best_mae,
    'gap': best_gap,
    'improvement_over_original': improvement,
    'improvement_pct': improvement_pct
}

joblib.dump(metadata_final, 'metadata_final_engineered.pkl')

# Save engineered dataset
df_eng.to_csv('catalyst_data_fully_engineered.csv', index=False)

print(f"\n✓ Saved:")
print(f"  1. model_xgboost_final_engineered.pkl")
print(f"  2. scaler_final_engineered.pkl")
print(f"  3. metadata_final_engineered.pkl")
print(f"  4. catalyst_data_fully_engineered.csv")

# Save results
df_results.to_csv('feature_engineering_results.csv', index=False)
print(f"  5. feature_engineering_results.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TIER 1 + TIER 2 COMPLETE!")
print("="*80)

print(f"\n🎯 COMPLETE JOURNEY:")
print(f"  Original (uncleaned):          R² = 0.39")
print(f"  After cleaning (Conservative): R² = 0.70 (+79%)")
print(f"  After feature engineering:     R² = {best_r2:.4f} ({improvement_pct:+.1f}%)")

total_improvement = best_r2 - 0.39
total_pct = (total_improvement / 0.39) * 100

print(f"\n  TOTAL IMPROVEMENT: {total_improvement:+.4f} ({total_pct:+.1f}%)")

print(f"\n💡 NEXT STEPS:")
if best_r2 > 0.75:
    print("  🎉 Excellent! R² > 0.75")
    print("  → Try ensemble methods for final boost")
    print("  → Consider catalyst family modeling")
elif best_r2 > 0.72:
    print("  ✅ Very good! R² > 0.72")
    print("  → Try ensemble (XGBoost + RF + LightGBM)")
    print("  → Fine-tune hyperparameters")
else:
    print("  ✅ Good progress from data cleaning")
    print("  → Feature engineering gave modest gains")
    print("  → Try ensemble methods")
    print("  → Consider catalyst clustering")

print("\n" + "="*80)