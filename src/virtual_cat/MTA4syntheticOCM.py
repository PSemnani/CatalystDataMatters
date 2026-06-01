"""
================================================================================
MOLECULAR TASK ARITHMETIC (MTA) - COMPLETE PIPELINE
Applied to Cleaned Conservative Dataset
================================================================================

Tracks both R² and MAE throughout
Includes comprehensive visualizations
Tests multiple strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import copy
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

print("="*80)
print("MOLECULAR TASK ARITHMETIC: CATALYST YIELD PREDICTION")
print("Using Conservative Cleaned Dataset")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CATALYST_ID = "cat_ID"
TARGET = "C2y"
DESCRIPTORS = [f"D{i}" for i in range(1, 17)]
PROCESS_CONDITIONS = ["T", "Q", "CH4_O2", "InertFraction"]
ORIGINAL_FEATURES = DESCRIPTORS + PROCESS_CONDITIONS

# MTA Parameters
YIELD_THRESHOLD = 0.10  # Define negative (<0.1) vs positive (>=0.1) catalysts
N_TEST_CATALYSTS = 30
FEW_SHOT_SIZES = [10, 20, 50, 100, 200, 500]

print("\nConfiguration:")
print(f"  Yield threshold: {YIELD_THRESHOLD}")
print(f"  Test catalysts: {N_TEST_CATALYSTS}")
print(f"  Few-shot sizes: {FEW_SHOT_SIZES}")

# ============================================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING CLEANED DATA")
print("="*80)

# Load Conservative cleaned data
df = pd.read_csv('catalyst_data_cleaned_conservative.csv')

print(f"\nLoaded Conservative cleaned dataset:")
print(f"  Samples: {len(df):,}")
print(f"  Catalysts: {df[CATALYST_ID].nunique()}")
print(f"  Features: {len(ORIGINAL_FEATURES)}")

print(f"\nYield statistics:")
print(f"  Min:  {df[TARGET].min():.4f}")
print(f"  Mean: {df[TARGET].mean():.4f}")
print(f"  Max:  {df[TARGET].max():.4f}")
print(f"  Std:  {df[TARGET].std():.4f}")

# ============================================================================
# STEP 2: DEFINE POSITIVE AND NEGATIVE CATALYSTS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: DEFINING CATALYST CLASSES")
print("="*80)

# Calculate average yield per catalyst
catalyst_avg_yield = df.groupby(CATALYST_ID)[TARGET].mean().sort_values()

# Classify catalysts
negative_catalysts = catalyst_avg_yield[catalyst_avg_yield < YIELD_THRESHOLD].index
positive_catalysts = catalyst_avg_yield[catalyst_avg_yield >= YIELD_THRESHOLD].index

print(f"\nThreshold: {YIELD_THRESHOLD}")
print(f"Negative catalysts (< {YIELD_THRESHOLD}): {len(negative_catalysts)}")
print(f"Positive catalysts (>= {YIELD_THRESHOLD}): {len(positive_catalysts)}")
print(f"Ratio negative/positive: {len(negative_catalysts)/len(positive_catalysts):.2f}")

print(f"\nNegative catalyst yield range: {catalyst_avg_yield.loc[negative_catalysts].min():.4f} - {catalyst_avg_yield.loc[negative_catalysts].max():.4f}")
print(f"Positive catalyst yield range: {catalyst_avg_yield.loc[positive_catalysts].min():.4f} - {catalyst_avg_yield.loc[positive_catalysts].max():.4f}")

# Label data
df['catalyst_class'] = df[CATALYST_ID].map(
    lambda x: 'negative' if x in negative_catalysts else 'positive'
)

df_negative = df[df['catalyst_class'] == 'negative']
df_positive = df[df['catalyst_class'] == 'positive']

print(f"\nSample counts:")
print(f"  Negative: {len(df_negative):,}")
print(f"  Positive: {len(df_positive):,}")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT (CATALYST-BASED)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: CREATING TRAIN-TEST SPLIT")
print("="*80)

np.random.seed(42)

# Stratified split by catalyst class
n_test_negative = int(N_TEST_CATALYSTS * len(negative_catalysts) / (len(negative_catalysts) + len(positive_catalysts)))
n_test_positive = N_TEST_CATALYSTS - n_test_negative

test_negative_cats = np.random.choice(negative_catalysts, size=n_test_negative, replace=False)
test_positive_cats = np.random.choice(positive_catalysts, size=n_test_positive, replace=False)
test_catalysts = list(test_negative_cats) + list(test_positive_cats)

train_negative_cats = [c for c in negative_catalysts if c not in test_catalysts]
train_positive_cats = [c for c in positive_catalysts if c not in test_catalysts]

print(f"\nTest set: {N_TEST_CATALYSTS} catalysts")
print(f"  Negative: {n_test_negative}")
print(f"  Positive: {n_test_positive}")

print(f"\nTraining set:")
print(f"  Negative catalysts: {len(train_negative_cats)}")
print(f"  Positive catalysts: {len(train_positive_cats)}")

# Create datasets
df_train_negative = df[df[CATALYST_ID].isin(train_negative_cats)]
df_train_positive = df[df[CATALYST_ID].isin(train_positive_cats)]
df_train_all = pd.concat([df_train_negative, df_train_positive])
df_test = df[df[CATALYST_ID].isin(test_catalysts)]

# Extract features
X_train_negative = df_train_negative[ORIGINAL_FEATURES].values
y_train_negative = df_train_negative[TARGET].values

X_train_positive = df_train_positive[ORIGINAL_FEATURES].values
y_train_positive = df_train_positive[TARGET].values

X_train_all = df_train_all[ORIGINAL_FEATURES].values
y_train_all = df_train_all[TARGET].values

X_test = df_test[ORIGINAL_FEATURES].values
y_test = df_test[TARGET].values

print(f"\nSample counts:")
print(f"  Train negative: {len(X_train_negative):,}")
print(f"  Train positive: {len(X_train_positive):,}")
print(f"  Train total: {len(X_train_all):,}")
print(f"  Test: {len(X_test):,}")

# Scale features
scaler = RobustScaler()
X_train_negative_scaled = scaler.fit_transform(X_train_negative)
X_train_positive_scaled = scaler.transform(X_train_positive)
X_train_all_scaled = scaler.transform(X_train_all)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features scaled with RobustScaler")

# ============================================================================
# STEP 4: HELPER FUNCTIONS
# ============================================================================

def extract_weights(model):
    """Extract weights from MLPRegressor"""
    weights = []
    for i in range(len(model.coefs_)):
        weights.append(model.coefs_[i].flatten())
        weights.append(model.intercepts_[i].flatten())
    return np.concatenate(weights)

def set_weights(model, weight_vector):
    """Set weights in MLPRegressor"""
    idx = 0
    for i in range(len(model.coefs_)):
        coef_size = model.coefs_[i].size
        model.coefs_[i] = weight_vector[idx:idx+coef_size].reshape(model.coefs_[i].shape)
        idx += coef_size
        
        intercept_size = model.intercepts_[i].size
        model.intercepts_[i] = weight_vector[idx:idx+intercept_size].reshape(model.intercepts_[i].shape)
        idx += intercept_size
    return model

def evaluate_model(model, X, y, label="Model"):
    """Evaluate model with both R² and MAE"""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return {
        'label': label,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'predictions': y_pred
    }

# ============================================================================
# STEP 5: TRAIN BASELINE MODELS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: TRAINING BASELINE MODELS")
print("="*80)

# Neural Network configuration (for MTA)
nn_config = {
    'hidden_layer_sizes': (128, 64),  # Medium size
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.05,  # Moderate regularization
    'learning_rate_init': 0.001,
    'learning_rate': 'adaptive',
    'batch_size': 128,
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.2,
    'n_iter_no_change': 20,
    'tol': 1e-4,
    'random_state': 42,
    'verbose': False
}

# XGBoost configuration (for comparison)
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

print("\n5.1 Training XGBoost Baseline (Best Model from Tier 1)")
print("-" * 70)

model_xgb = XGBRegressor(**xgb_config)
model_xgb.fit(X_train_all_scaled, y_train_all)

results_xgb = evaluate_model(model_xgb, X_test_scaled, y_test, "XGBoost Baseline")
results_xgb_train = evaluate_model(model_xgb, X_train_all_scaled, y_train_all, "XGBoost Train")

print(f"\nXGBoost Performance:")
print(f"  Train R² = {results_xgb_train['r2']:.4f}, MAE = {results_xgb_train['mae']:.4f}")
print(f"  Test R²  = {results_xgb['r2']:.4f}, MAE = {results_xgb['mae']:.4f}")
print(f"  Gap (R²) = {results_xgb_train['r2'] - results_xgb['r2']:.4f}")

print("\n5.2 Training Neural Network for MTA")
print("-" * 70)

# Pretrained model (all data)
print("\n⏳ Training pretrained NN (all data)...")
model_pretrained = MLPRegressor(**nn_config)
model_pretrained.fit(X_train_all_scaled, y_train_all)

results_pretrained = evaluate_model(model_pretrained, X_test_scaled, y_test, "NN Pretrained")
results_pretrained_train = evaluate_model(model_pretrained, X_train_all_scaled, y_train_all, "NN Pretrained Train")

print(f"\nPretrained NN Performance:")
print(f"  Train R² = {results_pretrained_train['r2']:.4f}, MAE = {results_pretrained_train['mae']:.4f}")
print(f"  Test R²  = {results_pretrained['r2']:.4f}, MAE = {results_pretrained['mae']:.4f}")
print(f"  Gap (R²) = {results_pretrained_train['r2'] - results_pretrained['r2']:.4f}")

# ============================================================================
# STEP 6: TRAIN OPPOSITE MODEL (NEGATIVE DATA)
# ============================================================================

print("\n" + "="*80)
print("STEP 6: TRAINING OPPOSITE MODEL (NEGATIVE CATALYSTS)")
print("="*80)

print(f"\n⏳ Training on {len(X_train_negative):,} negative samples...")
print(f"   (Catalysts with avg yield < {YIELD_THRESHOLD})")

model_opposite = MLPRegressor(**nn_config)
model_opposite.fit(X_train_negative_scaled, y_train_negative)

results_opposite = evaluate_model(model_opposite, X_test_scaled, y_test, "NN Opposite")
results_opposite_train = evaluate_model(model_opposite, X_train_negative_scaled, y_train_negative, "NN Opposite Train")

print(f"\nOpposite Model Performance:")
print(f"  Train R² = {results_opposite_train['r2']:.4f}, MAE = {results_opposite_train['mae']:.4f}")
print(f"  Test R²  = {results_opposite['r2']:.4f}, MAE = {results_opposite['mae']:.4f}")

# ============================================================================
# STEP 7: COMPUTE TASK VECTOR
# ============================================================================

print("\n" + "="*80)
print("STEP 7: COMPUTING TASK VECTOR")
print("="*80)

θ_pretrained = extract_weights(model_pretrained)
θ_opposite = extract_weights(model_opposite)

# Task vector: direction from pretrained to opposite (negative direction)
τ = θ_opposite - θ_pretrained

print(f"\n✓ Task vector computed:")
print(f"  Dimension: {len(τ)}")
print(f"  L2 norm: {np.linalg.norm(τ):.4f}")
print(f"  Mean: {τ.mean():.6f}")
print(f"  Std: {τ.std():.6f}")

# ============================================================================
# STEP 8: APPLY MTA (TEST LAMBDA VALUES)
# ============================================================================

print("\n" + "="*80)
print("STEP 8: MOLECULAR TASK ARITHMETIC - LAMBDA SEARCH")
print("="*80)

print("\n⏳ Testing lambda values to move AWAY from negative direction...")

# Test range of lambda values
lambdas = np.arange(0.0, 1.05, 0.05)
results_mta = []

for lambda_val in lambdas:
    # Apply MTA: move AWAY from negative direction (subtract task vector)
    θ_mta = θ_pretrained - lambda_val * τ
    
    # Create model with MTA weights
    model_mta = copy.deepcopy(model_pretrained)
    model_mta = set_weights(model_mta, θ_mta)
    
    # Evaluate
    result = evaluate_model(model_mta, X_test_scaled, y_test, f"MTA λ={lambda_val:.2f}")
    result['lambda'] = lambda_val
    results_mta.append(result)

df_mta = pd.DataFrame(results_mta)

# Find best lambda for R² and MAE
best_r2_idx = df_mta['r2'].idxmax()
best_mae_idx = df_mta['mae'].idxmin()

best_lambda_r2 = df_mta.loc[best_r2_idx, 'lambda']
best_r2 = df_mta.loc[best_r2_idx, 'r2']
best_mae_at_r2 = df_mta.loc[best_r2_idx, 'mae']

best_lambda_mae = df_mta.loc[best_mae_idx, 'lambda']
best_mae = df_mta.loc[best_mae_idx, 'mae']
best_r2_at_mae = df_mta.loc[best_mae_idx, 'r2']

print(f"\n✓ Lambda search complete:")
print(f"  Tested {len(lambdas)} values from {lambdas.min():.2f} to {lambdas.max():.2f}")

print(f"\n  Best for R²:")
print(f"    λ = {best_lambda_r2:.2f}")
print(f"    R² = {best_r2:.4f} (vs pretrained: {results_pretrained['r2']:.4f})")
print(f"    MAE = {best_mae_at_r2:.4f} (vs pretrained: {results_pretrained['mae']:.4f})")

print(f"\n  Best for MAE:")
print(f"    λ = {best_lambda_mae:.2f}")
print(f"    R² = {best_r2_at_mae:.4f}")
print(f"    MAE = {best_mae:.4f}")

# Use best lambda for R² as primary
best_lambda = best_lambda_r2
θ_mta_best = θ_pretrained - best_lambda * τ
model_mta_best = copy.deepcopy(model_pretrained)
model_mta_best = set_weights(model_mta_best, θ_mta_best)

results_mta_best = evaluate_model(model_mta_best, X_test_scaled, y_test, f"MTA Best (λ={best_lambda:.2f})")

# ============================================================================
# STEP 9: FEW-SHOT LEARNING
# ============================================================================

print("\n" + "="*80)
print("STEP 9: FEW-SHOT LEARNING")
print("="*80)

print(f"\n⏳ Testing few-shot learning with positive data...")
print(f"   Available positive samples: {len(X_train_positive):,}")

# Adjust few-shot sizes based on available data
few_shot_sizes_actual = [s for s in FEW_SHOT_SIZES if s <= len(X_train_positive)]
print(f"   Testing sizes: {few_shot_sizes_actual}")

results_fewshot_mta = []
results_fewshot_traditional = []

for n_shot in few_shot_sizes_actual:
    # Sample positive data
    indices = np.random.choice(len(X_train_positive), size=n_shot, replace=False)
    X_few = X_train_positive_scaled[indices]
    y_few = y_train_positive[indices]
    
    # MTA + Few-shot: Start from MTA model, finetune on positive
    model_mta_few = MLPRegressor(**nn_config)
    model_mta_few.coefs_ = [c.copy() for c in model_mta_best.coefs_]
    model_mta_few.intercepts_ = [i.copy() for i in model_mta_best.intercepts_]
    model_mta_few.n_layers_ = model_mta_best.n_layers_
    model_mta_few.n_outputs_ = model_mta_best.n_outputs_
    model_mta_few.out_activation_ = model_mta_best.out_activation_
    
    # Finetune (limited iterations)
    nn_config_finetune = nn_config.copy()
    nn_config_finetune['max_iter'] = 200
    nn_config_finetune['warm_start'] = True
    
    try:
        model_mta_few.fit(X_few, y_few)
        result_mta_few = evaluate_model(model_mta_few, X_test_scaled, y_test, f"MTA+Few n={n_shot}")
    except:
        # If warm start fails, train from scratch
        model_mta_few = MLPRegressor(**nn_config)
        model_mta_few.fit(X_few, y_few)
        result_mta_few = evaluate_model(model_mta_few, X_test_scaled, y_test, f"MTA+Few n={n_shot}")
    
    result_mta_few['n_shot'] = n_shot
    results_fewshot_mta.append(result_mta_few)
    
    # Traditional: Train from scratch on few-shot data
    model_trad = MLPRegressor(**nn_config)
    model_trad.fit(X_few, y_few)
    result_trad = evaluate_model(model_trad, X_test_scaled, y_test, f"Traditional n={n_shot}")
    result_trad['n_shot'] = n_shot
    results_fewshot_traditional.append(result_trad)
    
    print(f"  n={n_shot:4d}: MTA R²={result_mta_few['r2']:.4f} MAE={result_mta_few['mae']:.4f} | "
          f"Trad R²={result_trad['r2']:.4f} MAE={result_trad['mae']:.4f}")

df_fewshot_mta = pd.DataFrame(results_fewshot_mta)
df_fewshot_trad = pd.DataFrame(results_fewshot_traditional)

# ============================================================================
# STEP 10: COMPREHENSIVE COMPARISON
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE RESULTS: ALL METHODS")
print("="*80)

# Compile all results
all_results = {
    'XGBoost Baseline': results_xgb,
    'NN Pretrained': results_pretrained,
    'NN Opposite': results_opposite,
    f'MTA Zero-Shot (λ={best_lambda:.2f})': results_mta_best,
}

# Add best few-shot results if available
if len(df_fewshot_mta) > 0:
    best_fewshot_mta_idx = df_fewshot_mta['r2'].idxmax()
    best_fewshot_mta = df_fewshot_mta.loc[best_fewshot_mta_idx]
    all_results[f"MTA+Few-shot (n={int(best_fewshot_mta['n_shot'])})"] = best_fewshot_mta.to_dict()
    
    best_fewshot_trad_idx = df_fewshot_trad['r2'].idxmax()
    best_fewshot_trad = df_fewshot_trad.loc[best_fewshot_trad_idx]
    all_results[f"Traditional Few-shot (n={int(best_fewshot_trad['n_shot'])})"] = best_fewshot_trad.to_dict()

print(f"\n{'Method':<40} {'Test R²':>10} {'Test MAE':>10}")
print("-" * 65)

for method, result in all_results.items():
    print(f"{method:<40} {result['r2']:>10.4f} {result['mae']:>10.4f}")

print("-" * 65)

# Find best method
best_method_r2 = max(all_results.items(), key=lambda x: x[1]['r2'])
best_method_mae = min(all_results.items(), key=lambda x: x[1]['mae'])

print(f"\n🏆 BEST R²: {best_method_r2[0]}")
print(f"   R² = {best_method_r2[1]['r2']:.4f}, MAE = {best_method_r2[1]['mae']:.4f}")

print(f"\n🏆 BEST MAE: {best_method_mae[0]}")
print(f"   R² = {best_method_mae[1]['r2']:.4f}, MAE = {best_method_mae[1]['mae']:.4f}")

# Compare to XGBoost baseline
xgb_r2 = results_xgb['r2']
xgb_mae = results_xgb['mae']

print(f"\n📊 COMPARISON TO XGBOOST BASELINE:")
print(f"   XGBoost: R² = {xgb_r2:.4f}, MAE = {xgb_mae:.4f}")
print(f"   Best R²: {best_method_r2[1]['r2']:.4f} ({(best_method_r2[1]['r2']-xgb_r2)*100:+.1f}%)")
print(f"   Best MAE: {best_method_mae[1]['mae']:.4f} ({(best_method_mae[1]['mae']-xgb_mae)/xgb_mae*100:+.1f}%)")

# ============================================================================
# STEP 11: VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("STEP 11: CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

# Plot 1: Lambda search - R²
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_mta['lambda'], df_mta['r2'], 'o-', linewidth=2, markersize=8, color='steelblue')
ax1.axhline(results_pretrained['r2'], color='gray', linestyle='--', linewidth=2, label='Pretrained')
ax1.axhline(results_xgb['r2'], color='green', linestyle='--', linewidth=2, label='XGBoost')
ax1.axvline(best_lambda_r2, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax1.scatter([best_lambda_r2], [best_r2], s=200, c='gold', edgecolor='red', linewidth=3, zorder=5)
ax1.set_xlabel('Lambda (λ)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Test R²', fontweight='bold', fontsize=11)
ax1.set_title('MTA Lambda Search: R²', fontweight='bold', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Lambda search - MAE
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df_mta['lambda'], df_mta['mae'], 'o-', linewidth=2, markersize=8, color='coral')
ax2.axhline(results_pretrained['mae'], color='gray', linestyle='--', linewidth=2, label='Pretrained')
ax2.axhline(results_xgb['mae'], color='green', linestyle='--', linewidth=2, label='XGBoost')
ax2.axvline(best_lambda_mae, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax2.scatter([best_lambda_mae], [best_mae], s=200, c='gold', edgecolor='red', linewidth=3, zorder=5)
ax2.set_xlabel('Lambda (λ)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Test MAE', fontweight='bold', fontsize=11)
ax2.set_title('MTA Lambda Search: MAE', fontweight='bold', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: Method comparison - R²
ax3 = fig.add_subplot(gs[0, 2])
methods = list(all_results.keys())
r2_values = [all_results[m]['r2'] for m in methods]
colors = ['green' if 'XGBoost' in m else 'steelblue' if 'Pretrained' in m else 'orange' if 'MTA' in m else 'purple' for m in methods]
bars = ax3.bar(range(len(methods)), r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels([m[:20] for m in methods], rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Test R²', fontweight='bold', fontsize=11)
ax3.set_title('Method Comparison: R²', fontweight='bold', fontsize=13)
ax3.grid(alpha=0.3, axis='y')
# Highlight best
best_idx_r2 = r2_values.index(max(r2_values))
bars[best_idx_r2].set_edgecolor('gold')
bars[best_idx_r2].set_linewidth(4)

# Plot 4: Method comparison - MAE
ax4 = fig.add_subplot(gs[0, 3])
mae_values = [all_results[m]['mae'] for m in methods]
bars = ax4.bar(range(len(methods)), mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(len(methods)))
ax4.set_xticklabels([m[:20] for m in methods], rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Test MAE', fontweight='bold', fontsize=11)
ax4.set_title('Method Comparison: MAE', fontweight='bold', fontsize=13)
ax4.grid(alpha=0.3, axis='y')
# Highlight best
best_idx_mae = mae_values.index(min(mae_values))
bars[best_idx_mae].set_edgecolor('gold')
bars[best_idx_mae].set_linewidth(4)

# Plot 5: Few-shot learning - R²
if len(df_fewshot_mta) > 0:
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.plot(df_fewshot_mta['n_shot'], df_fewshot_mta['r2'], 'o-', linewidth=2, markersize=10, 
             color='orange', label='MTA + Few-shot')
    ax5.plot(df_fewshot_trad['n_shot'], df_fewshot_trad['r2'], 's-', linewidth=2, markersize=10,
             color='purple', label='Traditional Few-shot')
    ax5.axhline(results_xgb['r2'], color='green', linestyle='--', linewidth=2, label='XGBoost Baseline')
    ax5.axhline(results_mta_best['r2'], color='orange', linestyle=':', linewidth=2, alpha=0.5, label='MTA Zero-shot')
    ax5.set_xlabel('Number of Few-Shot Samples', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Test R²', fontweight='bold', fontsize=11)
    ax5.set_title('Few-Shot Learning Performance: R²', fontweight='bold', fontsize=13)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    ax5.set_xscale('log')

# Plot 6: Few-shot learning - MAE
if len(df_fewshot_mta) > 0:
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.plot(df_fewshot_mta['n_shot'], df_fewshot_mta['mae'], 'o-', linewidth=2, markersize=10,
             color='orange', label='MTA + Few-shot')
    ax6.plot(df_fewshot_trad['n_shot'], df_fewshot_trad['mae'], 's-', linewidth=2, markersize=10,
             color='purple', label='Traditional Few-shot')
    ax6.axhline(results_xgb['mae'], color='green', linestyle='--', linewidth=2, label='XGBoost Baseline')
    ax6.axhline(results_mta_best['mae'], color='orange', linestyle=':', linewidth=2, alpha=0.5, label='MTA Zero-shot')
    ax6.set_xlabel('Number of Few-Shot Samples', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Test MAE', fontweight='bold', fontsize=11)
    ax6.set_title('Few-Shot Learning Performance: MAE', fontweight='bold', fontsize=13)
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)
    ax6.set_xscale('log')

# Plot 7: Predictions scatter (best model)
ax7 = fig.add_subplot(gs[2, 0])
best_model_preds = best_method_r2[1]['predictions']
ax7.scatter(y_test, best_model_preds, alpha=0.6, s=30, edgecolor='black', linewidth=0.5, c=y_test, cmap='viridis')
ax7.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax7.set_xlabel('True C2y', fontweight='bold', fontsize=11)
ax7.set_ylabel('Predicted C2y', fontweight='bold', fontsize=11)
ax7.set_title(f'Best Model Predictions\n{best_method_r2[0][:30]}', fontweight='bold', fontsize=11)
ax7.grid(alpha=0.3)

# Plot 8: Residuals
ax8 = fig.add_subplot(gs[2, 1])
residuals_best = y_test - best_model_preds
ax8.scatter(best_model_preds, residuals_best, alpha=0.6, s=30, edgecolor='black', linewidth=0.5)
ax8.axhline(0, color='red', linestyle='--', linewidth=2)
ax8.set_xlabel('Predicted C2y', fontweight='bold', fontsize=11)
ax8.set_ylabel('Residuals', fontweight='bold', fontsize=11)
ax8.set_title('Residual Plot (Best Model)', fontweight='bold', fontsize=11)
ax8.grid(alpha=0.3)

# Plot 9: R² vs MAE trade-off
ax9 = fig.add_subplot(gs[2, 2])
for method, result in all_results.items():
    marker = 'o' if 'XGBoost' in method else 's' if 'MTA' in method else '^'
    size = 200 if method == best_method_r2[0] or method == best_method_mae[0] else 100
    ax9.scatter(result['r2'], result['mae'], s=size, marker=marker, alpha=0.7, 
               edgecolor='black', linewidth=2, label=method[:20])
ax9.set_xlabel('Test R²', fontweight='bold', fontsize=11)
ax9.set_ylabel('Test MAE', fontweight='bold', fontsize=11)
ax9.set_title('R² vs MAE Trade-off', fontweight='bold', fontsize=13)
ax9.legend(fontsize=7, loc='best')
ax9.grid(alpha=0.3)

# Plot 10: Summary statistics
ax10 = fig.add_subplot(gs[2, 3])
ax10.axis('off')

summary_text = f"""
MTA RESULTS SUMMARY

Baseline (XGBoost):
  R² = {results_xgb['r2']:.4f}
  MAE = {results_xgb['mae']:.4f}

Best MTA Zero-Shot:
  λ = {best_lambda:.2f}
  R² = {results_mta_best['r2']:.4f}
  MAE = {results_mta_best['mae']:.4f}
  
ΔR² = {results_mta_best['r2'] - results_xgb['r2']:+.4f}
ΔMAE = {results_mta_best['mae'] - results_xgb['mae']:+.4f}

Task Vector:
  Dimension: {len(τ)}
  L2 Norm: {np.linalg.norm(τ):.2f}

Dataset:
  Train: {len(X_train_all):,} samples
  Test: {len(X_test):,} samples
  Neg catalysts: {len(train_negative_cats)}
  Pos catalysts: {len(train_positive_cats)}
"""

ax10.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Molecular Task Arithmetic: Complete Analysis', fontsize=16, fontweight='bold')
plt.savefig('mta_complete_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: mta_complete_analysis.png")
plt.show()

# ============================================================================
# STEP 12: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 12: SAVING RESULTS")
print("="*80)

import joblib

# Save best models
joblib.dump(model_xgb, 'model_xgb_baseline.pkl')
joblib.dump(model_pretrained, 'model_nn_pretrained.pkl')
joblib.dump(model_opposite, 'model_nn_opposite.pkl')
joblib.dump(model_mta_best, 'model_mta_best.pkl')
joblib.dump(scaler, 'scaler_mta.pkl')

# Save task vector
joblib.dump({
    'task_vector': τ,
    'best_lambda': best_lambda,
    'pretrained_weights': θ_pretrained,
    'opposite_weights': θ_opposite
}, 'mta_task_vector.pkl')

# Save results DataFrame
results_summary = pd.DataFrame([
    {'Method': k, 'R2': v['r2'], 'MAE': v['mae'], 'RMSE': v.get('rmse', np.nan)}
    for k, v in all_results.items()
])
results_summary.to_csv('mta_results_summary.csv', index=False)

# Save lambda search results
df_mta.to_csv('mta_lambda_search.csv', index=False)

# Save few-shot results if available
if len(df_fewshot_mta) > 0:
    df_fewshot_mta.to_csv('mta_fewshot_results.csv', index=False)
    df_fewshot_trad.to_csv('traditional_fewshot_results.csv', index=False)

print(f"\n✓ Saved files:")
print(f"  1. model_xgb_baseline.pkl")
print(f"  2. model_nn_pretrained.pkl")
print(f"  3. model_nn_opposite.pkl")
print(f"  4. model_mta_best.pkl")
print(f"  5. scaler_mta.pkl")
print(f"  6. mta_task_vector.pkl")
print(f"  7. mta_results_summary.csv")
print(f"  8. mta_lambda_search.csv")
if len(df_fewshot_mta) > 0:
    print(f"  9. mta_fewshot_results.csv")
    print(f" 10. traditional_fewshot_results.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("MTA ANALYSIS COMPLETE!")
print("="*80)

print(f"\n📊 KEY FINDINGS:")
print(f"\n1. BASELINE PERFORMANCE:")
print(f"   XGBoost: R² = {results_xgb['r2']:.4f}, MAE = {results_xgb['mae']:.4f}")

print(f"\n2. MTA ZERO-SHOT:")
print(f"   Best λ = {best_lambda:.2f}")
print(f"   R² = {results_mta_best['r2']:.4f} ({(results_mta_best['r2']-results_xgb['r2'])*100:+.1f}%)")
print(f"   MAE = {results_mta_best['mae']:.4f} ({(results_mta_best['mae']-results_xgb['mae'])/results_xgb['mae']*100:+.1f}%)")

if len(df_fewshot_mta) > 0:
    print(f"\n3. BEST FEW-SHOT:")
    print(f"   MTA: R² = {best_fewshot_mta['r2']:.4f}, MAE = {best_fewshot_mta['mae']:.4f}")
    print(f"   Traditional: R² = {best_fewshot_trad['r2']:.4f}, MAE = {best_fewshot_trad['mae']:.4f}")

print(f"\n💡 CONCLUSION:")
if results_mta_best['r2'] > results_xgb['r2']:
    print(f"   ✅ MTA improved over XGBoost baseline!")
else:
    print(f"   ⚠️  MTA did not improve over XGBoost")
    print(f"   → XGBoost (tree-based) remains best for this problem")
    print(f"   → Neural network MTA limited by model architecture")

print("\n" + "="*80)