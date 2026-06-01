#!/usr/bin/env python3
"""
Master Script: Run All Experiments
===================================
Runs validation, fast tests, and optionally full experiments in sequence.

Usage:
    python run_all_experiments.py --data_path your_data.csv --mode fast_test
    python run_all_experiments.py --data_path your_data.csv --mode full
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Could not find script: {cmd[1]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Master script to run all experiments"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV data file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["validate_only", "fast_test", "full"],
        default="fast_test",
        help="Experiment mode: validate_only, fast_test, or full (default: fast_test)"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip data validation step"
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"ERROR: Data file not found: {args.data_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("MASTER EXPERIMENT RUNNER")
    print("="*70)
    print(f"Data file: {args.data_path}")
    print(f"Mode: {args.mode}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    # Step 1: Validate data
    if not args.skip_validation:
        success = run_command(
            ["python", "validate_data.py", "--data_path", args.data_path],
            "Data Validation"
        )
        results["validation"] = success
        
        if not success:
            print("\n" + "="*70)
            print("⚠ VALIDATION FAILED")
            print("Please fix data issues before proceeding.")
            print("="*70)
            sys.exit(1)
    else:
        print("\nSkipping validation (--skip_validation flag used)")
    
    if args.mode == "validate_only":
        print("\n" + "="*70)
        print("✓ VALIDATION COMPLETE")
        print("Run with --mode fast_test to test experiments")
        print("="*70)
        sys.exit(0)
    
    # Step 2: Run experiments based on mode
    if args.mode == "fast_test":
        print("\n" + "="*70)
        print("RUNNING FAST TESTS")
        print("This will take approximately 10-15 minutes")
        print("="*70)
        
        # Fast test: volume study
        success = run_command(
            ["python", "volume_study.py", 
             "--data_path", args.data_path,
             "--fast_test"],
            "Volume Study (Fast Test)"
        )
        results["volume_study_fast"] = success
        
        # Fast test: baseline
        success = run_command(
            ["python", "baseline_random_split.py",
             "--data_path", args.data_path,
             "--seeds", "1", "2"],
            "Baseline Random Split (Fast Test)"
        )
        results["baseline_fast"] = success
        
        # Fast test: hyperparameter search
        success = run_command(
            ["python", "hyperparameter_search.py",
             "--data_path", args.data_path,
             "--fast_test"],
            "Hyperparameter Search (Fast Test)"
        )
        results["hyperparam_fast"] = success
        
    elif args.mode == "full":
        print("\n" + "="*70)
        print("RUNNING FULL EXPERIMENTS")
        print("This will take approximately 2-4 hours")
        print("="*70)
        
        response = input("\nThis will take a long time. Continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted.")
            sys.exit(0)
        
        # Full: volume study
        success = run_command(
            ["python", "volume_study.py",
             "--data_path", args.data_path,
             "--seeds", "1", "2", "3", "4", "5",
             "--train_sizes", "1", "3", "5", "10", "20", "30", "40", "50", 
                             "70", "100", "150", "200", "300"],
            "Volume Study (Full)"
        )
        results["volume_study_full"] = success
        
        # Full: baseline
        success = run_command(
            ["python", "baseline_random_split.py",
             "--data_path", args.data_path,
             "--seeds", "1", "2", "3", "4", "5"],
            "Baseline Random Split (Full)"
        )
        results["baseline_full"] = success
        
        # Full: hyperparameter search
        success = run_command(
            ["python", "hyperparameter_search.py",
             "--data_path", args.data_path,
             "--seeds", "1", "2", "3", "4", "5",
             "--n_param_samples", "50"],
            "Hyperparameter Search (Full)"
        )
        results["hyperparam_full"] = success
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_success = True
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
        if not success:
            all_success = False
    
    print("="*70)
    
    if all_success:
        print("\n✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("  1. Review the output folders for results")
        print("  2. Analyze results_summary.csv files")
        print("  3. Check prediction plots")
        print("  4. If fast tests look good, run full experiments")
    else:
        print("\n⚠ SOME EXPERIMENTS FAILED")
        print("Check the error messages above for details")
    
    print()


if __name__ == "__main__":
    main()
