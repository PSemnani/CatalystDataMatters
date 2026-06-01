"""
Data Validation Script
=======================
Checks that the virtual catalyst dataset has the correct format and structure.
Run this first to ensure your data is ready for the volume study.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def validate_data(data_path):
    """
    Validate the dataset format and structure.
    
    Args:
        data_path: Path to the CSV file
    
    Returns:
        bool: True if validation passed, False otherwise
    """
    print("="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    try:
        # Load data
        print(f"\n1. Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"   ✓ Data loaded successfully")
        print(f"   Shape: {df.shape}")
        
        # Check required columns
        print("\n2. Checking required columns...")
        required_cols = {
            "cat_ID": "Catalyst identifier",
            "T": "Temperature",
            "Q": "Flow rate", 
            "CH4_O2": "CH4/O2 ratio",
            "InertFraction": "Inert fraction",
            "C2y": "Target variable (C2 yield)",
        }
        
        descriptor_cols = [f"D{i}" for i in range(1, 17)]
        
        missing_cols = []
        for col, desc in required_cols.items():
            if col in df.columns:
                print(f"   ✓ {col}: {desc}")
            else:
                print(f"   ✗ MISSING: {col}: {desc}")
                missing_cols.append(col)
        
        # Check descriptors
        missing_descriptors = []
        for col in descriptor_cols:
            if col not in df.columns:
                missing_descriptors.append(col)
        
        if len(missing_descriptors) == 0:
            print(f"   ✓ All 16 descriptors (D1-D16) present")
        else:
            print(f"   ✗ MISSING descriptors: {missing_descriptors}")
            missing_cols.extend(missing_descriptors)
        
        if missing_cols:
            print(f"\n   ERROR: Missing required columns: {missing_cols}")
            return False
        
        # Check data types
        print("\n3. Checking data types...")
        numeric_cols = ["T", "Q", "CH4_O2", "InertFraction", "C2y"] + descriptor_cols
        
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"   ✓ {col}: numeric")
            else:
                print(f"   ⚠ {col}: {df[col].dtype} (expected numeric)")
        
        # Check catalyst structure
        print("\n4. Checking catalyst structure...")
        n_catalysts = df["cat_ID"].nunique()
        n_total = len(df)
        avg_per_catalyst = n_total / n_catalysts
        
        print(f"   Total rows: {n_total:,}")
        print(f"   Unique catalysts: {n_catalysts}")
        print(f"   Average experiments per catalyst: {avg_per_catalyst:.1f}")
        
        # Check for missing values
        print("\n5. Checking for missing values...")
        missing_counts = df[numeric_cols + ["cat_ID"]].isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            print(f"   ✓ No missing values found")
        else:
            print(f"   ⚠ Found {total_missing} missing values:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"      {col}: {count} ({count/len(df)*100:.2f}%)")
        
        # Check target variable distribution
        print("\n6. Analyzing target variable (C2y)...")
        print(f"   Min:    {df['C2y'].min():.4f}")
        print(f"   Max:    {df['C2y'].max():.4f}")
        print(f"   Mean:   {df['C2y'].mean():.4f}")
        print(f"   Median: {df['C2y'].median():.4f}")
        print(f"   Std:    {df['C2y'].std():.4f}")
        
        # Check process conditions ranges
        print("\n7. Process condition ranges...")
        for col in ["T", "Q", "CH4_O2", "InertFraction"]:
            unique_vals = df[col].nunique()
            print(f"   {col}:")
            print(f"      Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
            print(f"      Unique values: {unique_vals}")
        
        # Sample catalysts
        print("\n8. Sample catalyst IDs (first 10):")
        sample_cats = df["cat_ID"].unique()[:10]
        for cat in sample_cats:
            n_exp = len(df[df["cat_ID"] == cat])
            print(f"   {cat}: {n_exp} experiments")
        
        # Final verdict
        print("\n" + "="*60)
        if total_missing > 0:
            print("⚠ VALIDATION PASSED WITH WARNINGS")
            print("  Data structure is correct but contains missing values")
        else:
            print("✓ VALIDATION PASSED")
            print("  Data is ready for volume study!")
        print("="*60)
        
        return True
        
    except FileNotFoundError:
        print(f"\n✗ ERROR: File not found: {data_path}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate virtual catalyst dataset format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV data file",
    )
    
    args = parser.parse_args()
    
    success = validate_data(args.data_path)
    
    if success:
        print("\nYou can now run the volume study with:")
        print(f"  python volume_study.py --data_path {args.data_path} --fast_test")
    else:
        print("\nPlease fix the data issues before proceeding.")
