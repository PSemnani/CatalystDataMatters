"""
================================================================================
FILTERING Virtual CATALYST
4 Datasets with Temperature Filtering
================================================================================
"""

import pandas as pd
import numpy as np
import os

print("="*80)
print("CATALYST FILTERING: CREATE 4 DATASETS")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CATALYST_ID = "cat_ID"
TARGET = "C2y"
LOW_YIELD_THRESHOLD = 0.06
TEMPERATURES_TO_REMOVE = [650, 950]

# Output folder
OUTPUT_FOLDER = '/Users/parastoo/DataMatters/Dataset/virtual_catalyst_datasets'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Yield threshold: {LOW_YIELD_THRESHOLD}")
print(f"  Temperatures to remove: {TEMPERATURES_TO_REMOVE}")
print(f"  Output folder: {OUTPUT_FOLDER}")

# ============================================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOAD DATA AND REMOVE TEMPERATURE CONDITIONS")
print("="*80)

# Load original data
df_original = pd.read_csv('/Users/parastoo/DataMatters/Dataset/virtual_catalyst_dataset_with_inert.csv')

print(f"\nOriginal dataset:")
print(f"  Samples: {len(df_original):,}")
print(f"  Catalysts: {df_original[CATALYST_ID].nunique()}")

# Basic cleaning (remove NaN and inf)
df_original = df_original.dropna(subset=[TARGET])
df_original = df_original[~np.isinf(df_original[TARGET])]

# CRITICAL: Remove temperature conditions 650 and 950
df = df_original[~df_original['T'].isin(TEMPERATURES_TO_REMOVE)].copy()

print(f"\nAfter removing T={TEMPERATURES_TO_REMOVE}:")
print(f"  Samples: {len(df):,} (removed {len(df_original) - len(df):,})")
print(f"  Catalysts: {df[CATALYST_ID].nunique()}")
print(f"  Remaining temperatures: {sorted(df['T'].unique())}")

# ============================================================================
# STEP 1B: CHECK AND FIX NEGATIVE YIELDS
# ============================================================================

print("\n" + "="*80)
print("STEP 1B: CHECK FOR NEGATIVE YIELDS")
print("="*80)

# Check for negative yields BEFORE cleaning
negative_mask = df[TARGET] < 0
n_negative = negative_mask.sum()

if n_negative > 0:
    negative_values = df.loc[negative_mask, TARGET]
    print(f"\n⚠️  Found {n_negative:,} negative yield values:")
    print(f"  Most negative: {negative_values.min():.6f}")
    print(f"  Least negative: {negative_values.max():.6f}")
    print(f"  Mean of negatives: {negative_values.mean():.6f}")
    print(f"  Median of negatives: {negative_values.median():.6f}")
    print(f"  % of total data: {n_negative / len(df) * 100:.4f}%")
    
    # Set negative values to 0
    df.loc[negative_mask, TARGET] = 0.0
    print(f"\n✓ Set all {n_negative:,} negative values to 0.0")
else:
    print(f"\n✓ No negative yield values found")

print(f"\nData range after cleaning:")
print(f"  Min yield: {df[TARGET].min():.6f}")
print(f"  Max yield: {df[TARGET].max():.6f}")

# ============================================================================
# STEP 2: ANALYZE EACH CATALYST
# ============================================================================

print("\n" + "="*80)
print("STEP 2: ANALYZE CATALYST QUALITY")
print("="*80)

catalyst_stats = []

for cat in df[CATALYST_ID].unique():
    cat_data = df[df[CATALYST_ID] == cat]
    
    n_total = len(cat_data)
    n_above_threshold = (cat_data[TARGET] >= LOW_YIELD_THRESHOLD).sum()
    n_below_threshold = (cat_data[TARGET] < LOW_YIELD_THRESHOLD).sum()
    pct_below = (n_below_threshold / n_total) * 100
    
    catalyst_stats.append({
        'catalyst_id': cat,
        'total_samples': n_total,
        'samples_above_0.06': n_above_threshold,
        'samples_below_0.06': n_below_threshold,
        'pct_below_0.06': pct_below,
        'mean_yield': cat_data[TARGET].mean()
    })

df_catalyst_stats = pd.DataFrame(catalyst_stats)

print(f"\nCatalyst quality summary:")
print(f"  Total catalysts: {len(df_catalyst_stats)}")
print(f"  Catalysts with 0% below 0.06: {(df_catalyst_stats['pct_below_0.06'] == 0).sum()}")
print(f"  Catalysts with <50% below 0.06: {(df_catalyst_stats['pct_below_0.06'] < 50).sum()}")
print(f"  Catalysts with 100% below 0.06: {(df_catalyst_stats['pct_below_0.06'] == 100).sum()}")

# ============================================================================
# DATASET 1: TOP 59 CATALYSTS BY COUNT OF best SAMPLES
# ============================================================================

print("\n" + "="*80)
print("DATASET 1: TOP 59 CATALYSTS (MOST SAMPLES > 0.06)")
print("="*80)

# Sort by number of samples ABOVE threshold (descending) and take top 59
top_59_catalysts = df_catalyst_stats.nlargest(59, 'samples_above_0.06')['catalyst_id'].values

df_top59 = df[df[CATALYST_ID].isin(top_59_catalysts)].copy()

print(f"\nTop 59 catalysts selected based on COUNT of samples ≥ 0.06")
print(f"  Catalysts: {len(top_59_catalysts)}")
print(f"  Total samples: {len(df_top59):,}")
print(f"  Samples ≥ 0.06: {(df_top59[TARGET] >= LOW_YIELD_THRESHOLD).sum():,}")
print(f"  Samples < 0.06: {(df_top59[TARGET] < LOW_YIELD_THRESHOLD).sum():,}")
print(f"  % samples < 0.06: {(df_top59[TARGET] < LOW_YIELD_THRESHOLD).sum() / len(df_top59) * 100:.2f}%")

# Save
output_path_1 = os.path.join(OUTPUT_FOLDER, 'top59catalysts.csv')
df_top59.to_csv(output_path_1, index=False)
print(f"\n✓ Saved: {output_path_1}")

# Show top 10 catalysts
top_10_info = df_catalyst_stats.nlargest(10, 'samples_above_0.06')[
    ['catalyst_id', 'samples_above_0.06', 'total_samples', 'pct_below_0.06']
]
print(f"\nTop 10 catalysts by count of good samples:")
print(top_10_info.to_string(index=False))

# ============================================================================
# DATASET 2: REMOVE CATALYSTS WITH ≥50% BELOW THRESHOLD
# ============================================================================

print("\n" + "="*80)
print("DATASET 2: REMOVE CATALYSTS WITH ≥50% SAMPLES < 0.06")
print("="*80)

# Keep catalysts where LESS than 50% are below threshold
cats_50percent = df_catalyst_stats[df_catalyst_stats['pct_below_0.06'] < 50]['catalyst_id'].values

df_50percent = df[df[CATALYST_ID].isin(cats_50percent)].copy()

print(f"\nKeep catalysts with <50% samples below 0.06")
print(f"  Catalysts kept: {len(cats_50percent)}")
print(f"  Catalysts removed: {len(df_catalyst_stats) - len(cats_50percent)}")
print(f"  Total samples: {len(df_50percent):,}")
print(f"  Samples ≥ 0.06: {(df_50percent[TARGET] >= LOW_YIELD_THRESHOLD).sum():,}")
print(f"  Samples < 0.06: {(df_50percent[TARGET] < LOW_YIELD_THRESHOLD).sum():,}")
print(f"  % samples < 0.06: {(df_50percent[TARGET] < LOW_YIELD_THRESHOLD).sum() / len(df_50percent) * 100:.2f}%")

# Save
output_path_2 = os.path.join(OUTPUT_FOLDER, 'catalyst50percent.csv')
df_50percent.to_csv(output_path_2, index=False)
print(f"\n✓ Saved: {output_path_2}")

# ============================================================================
# DATASET 3: REMOVE ONLY 100% BAD CATALYSTS
# ============================================================================

print("\n" + "="*80)
print("DATASET 3: REMOVE ONLY CATALYSTS WITH 100% SAMPLES < 0.06")
print("="*80)

# Keep catalysts where NOT all samples are below threshold
cats_not_100percent_bad = df_catalyst_stats[df_catalyst_stats['pct_below_0.06'] < 100]['catalyst_id'].values

df_not_100percent = df[df[CATALYST_ID].isin(cats_not_100percent_bad)].copy()

print(f"\nRemove only catalysts with 100% samples below 0.06")
print(f"  Catalysts kept: {len(cats_not_100percent_bad)}")
print(f"  Catalysts removed: {len(df_catalyst_stats) - len(cats_not_100percent_bad)}")
print(f"  Total samples: {len(df_not_100percent):,}")
print(f"  Samples ≥ 0.06: {(df_not_100percent[TARGET] >= LOW_YIELD_THRESHOLD).sum():,}")
print(f"  Samples < 0.06: {(df_not_100percent[TARGET] < LOW_YIELD_THRESHOLD).sum():,}")
print(f"  % samples < 0.06: {(df_not_100percent[TARGET] < LOW_YIELD_THRESHOLD).sum() / len(df_not_100percent) * 100:.2f}%")

# Save
output_path_3 = os.path.join(OUTPUT_FOLDER, 'catalyst100percent.csv')
df_not_100percent.to_csv(output_path_3, index=False)
print(f"\n✓ Saved: {output_path_3}")

# ============================================================================
# DATASET 4: ORIGINAL DATA (TEMPERATURE FILTERED ONLY)
# ============================================================================

print("\n" + "="*80)
print("DATASET 4: ORIGINAL (TEMPERATURE FILTERED ONLY)")
print("="*80)

print(f"\nAll catalysts, only T={TEMPERATURES_TO_REMOVE} removed")
print(f"  Catalysts: {df[CATALYST_ID].nunique()}")
print(f"  Total samples: {len(df):,}")
print(f"  Samples ≥ 0.06: {(df[TARGET] >= LOW_YIELD_THRESHOLD).sum():,}")
print(f"  Samples < 0.06: {(df[TARGET] < LOW_YIELD_THRESHOLD).sum():,}")
print(f"  % samples < 0.06: {(df[TARGET] < LOW_YIELD_THRESHOLD).sum() / len(df) * 100:.2f}%")
print(f"  Min yield: {df[TARGET].min():.6f}")
print(f"  Max yield: {df[TARGET].max():.6f}")

# Save
output_path_4 = os.path.join(OUTPUT_FOLDER, 'original_temp_filtered.csv')
df.to_csv(output_path_4, index=False)
print(f"\n✓ Saved: {output_path_4}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY: 4 DATASETS CREATED")
print("="*80)

print(f"\n{'Dataset':<35} {'Catalysts':>12} {'Samples':>12} {'% Data':>10} {'Samples < 0.06':>15} {'% < 0.06':>10}")
print("-" * 100)

# Dataset 4 - Original (temp filtered)
orig_below = (df[TARGET] < LOW_YIELD_THRESHOLD).sum()
print(f"{'Original (T filtered)':<35} {df[CATALYST_ID].nunique():>12} {len(df):>12,} {100.0:>9.1f}% {orig_below:>15,} {orig_below/len(df)*100:>9.1f}%")

# Dataset 1 - Top 59
top59_below = (df_top59[TARGET] < LOW_YIELD_THRESHOLD).sum()
print(f"{'Top 59 Catalysts':<35} {len(top_59_catalysts):>12} {len(df_top59):>12,} {len(df_top59)/len(df)*100:>9.1f}% {top59_below:>15,} {top59_below/len(df_top59)*100:>9.1f}%")

# Dataset 2 - 50%
pct50_below = (df_50percent[TARGET] < LOW_YIELD_THRESHOLD).sum()
print(f"{'50% Threshold':<35} {len(cats_50percent):>12} {len(df_50percent):>12,} {len(df_50percent)/len(df)*100:>9.1f}% {pct50_below:>15,} {pct50_below/len(df_50percent)*100:>9.1f}%")

# Dataset 3 - 100%
pct100_below = (df_not_100percent[TARGET] < LOW_YIELD_THRESHOLD).sum()
print(f"{'100% Threshold':<35} {len(cats_not_100percent_bad):>12} {len(df_not_100percent):>12,} {len(df_not_100percent)/len(df)*100:>9.1f}% {pct100_below:>15,} {pct100_below/len(df_not_100percent)*100:>9.1f}%")

print("-" * 100)

print(f"\n✅ ALL 4 DATASETS SAVED IN:")
print(f"   {OUTPUT_FOLDER}/")
print(f"\n   Files created:")
print(f"   1. original_temp_filtered.csv  (baseline - all catalysts)")
print(f"   2. top59catalysts.csv          (highest quality)")
print(f"   3. catalyst50percent.csv       (medium quality)")
print(f"   4. catalyst100percent.csv      (low quality threshold)")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\n" + "="*80)
print("VERIFICATION: CHECK FILE CONTENTS")
print("="*80)

for filename in ['original_temp_filtered.csv', 'top59catalysts.csv', 'catalyst50percent.csv', 'catalyst100percent.csv']:
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    df_verify = pd.read_csv(filepath)
    print(f"\n{filename}:")
    print(f"  Rows: {len(df_verify):,}")
    print(f"  Columns: {len(df_verify.columns)}")
    print(f"  Catalysts: {df_verify[CATALYST_ID].nunique()}")
    print(f"  Column names: {list(df_verify.columns)}")
    print(f"  Temperature values: {sorted(df_verify['T'].unique())}")
    print(f"  Yield range: [{df_verify[TARGET].min():.6f}, {df_verify[TARGET].max():.6f}]")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
