"""
================================================================================
COMPARE RESULTS ACROSS 3 VIRTUAL CATALYST DATASETS
High-quality individual plots for publication
================================================================================
Author: Parastoo
Date: 2026-02-02
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# Publication settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Color scheme
COLORS = {
    'Top59': '#4472C4',      # Blue
    '50pct': '#ED7D31',      # Orange  
    '100pct': '#70AD47',     # Green
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure(fig, output_dir, filename):
    """Save figure in both PNG and PDF formats."""
    output_dir = Path(output_dir)
    
    # Save PNG (high resolution)
    fig.savefig(output_dir / f'{filename}.png', dpi=400, bbox_inches='tight', facecolor='white')
    
    # Save PDF (vector, best for LaTeX)
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    print(f"  ✓ Saved: {filename}.png and {filename}.pdf")


def load_results(path1, path2, path3, names):
    """Load results from 3 datasets."""
    results = {}
    
    for path, name in zip([path1, path2, path3], names):
        df = pd.read_csv(path)
        df['dataset'] = name
        results[name] = df
        print(f"Loaded {name}: {len(df)} experiments")
    
    return results


# ============================================================================
# INDIVIDUAL PLOT FUNCTIONS
# ============================================================================

def plot_r2_boxplot(results_dict, output_dir):
    """Test R² comparison - Box plot."""
    datasets = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    box_data = [results_dict[name]['test_r2'].values for name in datasets]
    positions = np.arange(len(datasets))
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                    medianprops=dict(linewidth=2, color='darkred'),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    for patch, name in zip(bp['boxes'], datasets):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('Test R² Comparison', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add sample size annotations
    for i, name in enumerate(datasets):
        n = len(results_dict[name])
        ax.text(i, ax.get_ylim()[0], f'n={n}', ha='center', va='top', fontsize=10)
    
    save_figure(fig, output_dir, 'fig1_r2_comparison')


def plot_mae_boxplot(results_dict, output_dir):
    """Test MAE comparison - Box plot."""
    datasets = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    box_data = [results_dict[name]['test_mae'].values for name in datasets]
    positions = np.arange(len(datasets))
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                    medianprops=dict(linewidth=2, color='darkred'),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    for patch, name in zip(bp['boxes'], datasets):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Test MAE', fontweight='bold')
    ax.set_title('Test MAE Comparison', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add sample size annotations
    for i, name in enumerate(datasets):
        n = len(results_dict[name])
        ax.text(i, ax.get_ylim()[1], f'n={n}', ha='center', va='bottom', fontsize=10)
    
    save_figure(fig, output_dir, 'fig2_mae_comparison')


def plot_rmse_boxplot(results_dict, output_dir):
    """Test RMSE comparison - Box plot."""
    datasets = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    box_data = [results_dict[name]['test_rmse'].values for name in datasets]
    positions = np.arange(len(datasets))
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                    medianprops=dict(linewidth=2, color='darkred'),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    for patch, name in zip(bp['boxes'], datasets):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Test RMSE', fontweight='bold')
    ax.set_title('Test RMSE Comparison', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    save_figure(fig, output_dir, 'fig3_rmse_comparison')


def plot_r2_violin(results_dict, output_dir):
    """Test R² distribution - Violin plot."""
    datasets = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    box_data = [results_dict[name]['test_r2'].values for name in datasets]
    positions = np.arange(len(datasets))
    
    parts = ax.violinplot(box_data, positions=positions, widths=0.7,
                         showmeans=True, showmedians=True)
    
    # Color the violin plots
    for pc, name in zip(parts['bodies'], datasets):
        pc.set_facecolor(COLORS[name])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Style other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('Test R² Distribution', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    save_figure(fig, output_dir, 'fig4_r2_distribution')


def plot_mae_vs_r2_scatter(results_dict, output_dir):
    """MAE vs R² trade-off scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    for name in results_dict.keys():
        df = results_dict[name]
        ax.scatter(df['test_mae'], df['test_r2'],
                  s=150, alpha=0.6, color=COLORS[name],
                  label=name, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Test MAE', fontweight='bold')
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('MAE vs R² Trade-off', fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(alpha=0.3)
    
    save_figure(fig, output_dir, 'fig5_mae_vs_r2_tradeoff')


def plot_dataset_sizes(results_dict, output_dir):
    """Dataset sizes comparison."""
    datasets = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    train_samples = [results_dict[name]['n_train_samples'].iloc[0] for name in datasets]
    test_samples = [results_dict[name]['n_test_samples'].iloc[0] for name in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_samples, width, label='Train Samples',
                  color='#4472C4', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_samples, width, label='Test Samples',
                  color='#ED7D31', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Dataset Sizes', fontweight='bold')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(alpha=0.3, axis='y')
    
    save_figure(fig, output_dir, 'fig6_dataset_sizes')


def plot_mean_r2_barplot(results_dict, output_dir):
    """Mean Test R² with error bars."""
    datasets = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mean_r2 = [results_dict[name]['test_r2'].mean() for name in datasets]
    std_r2 = [results_dict[name]['test_r2'].std() for name in datasets]
    
    x = np.arange(len(datasets))
    bars = ax.bar(x, mean_r2, yerr=std_r2, capsize=10,
                  color=[COLORS[name] for name in datasets],
                  alpha=0.8, edgecolor='black', linewidth=2,
                  error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars, mean_r2, std_r2)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + err + 0.02,
               f'{val:.4f}\n±{err:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Mean Test R²', fontweight='bold')
    ax.set_title('Average Test R² with Standard Deviation', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, max(mean_r2) + max(std_r2) + 0.15)
    
    save_figure(fig, output_dir, 'fig7_mean_r2_comparison')


def plot_mean_mae_barplot(results_dict, output_dir):
    """Mean Test MAE with error bars."""
    datasets = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mean_mae = [results_dict[name]['test_mae'].mean() for name in datasets]
    std_mae = [results_dict[name]['test_mae'].std() for name in datasets]
    
    x = np.arange(len(datasets))
    bars = ax.bar(x, mean_mae, yerr=std_mae, capsize=10,
                  color=[COLORS[name] for name in datasets],
                  alpha=0.8, edgecolor='black', linewidth=2,
                  error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars, mean_mae, std_mae)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + err + 0.003,
               f'{val:.4f}\n±{err:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Mean Test MAE', fontweight='bold')
    ax.set_title('Average Test MAE with Standard Deviation', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, max(mean_mae) + max(std_mae) + 0.03)
    
    save_figure(fig, output_dir, 'fig8_mean_mae_comparison')


def plot_performance_summary(results_dict, output_dir):
    """Combined performance metrics table as figure."""
    datasets = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create summary table
    summary_data = []
    for name in datasets:
        df = results_dict[name]
        summary_data.append([
            name,
            f"{df['n_train_samples'].iloc[0]:,}",
            f"{df['n_test_samples'].iloc[0]:,}",
            f"{df['test_r2'].mean():.4f} ± {df['test_r2'].std():.4f}",
            f"{df['test_mae'].mean():.4f} ± {df['test_mae'].std():.4f}",
            f"{df['test_rmse'].mean():.4f} ± {df['test_rmse'].std():.4f}",
            f"{df['cv_mae_mean'].mean():.4f} ± {df['cv_mae_mean'].std():.4f}",
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Dataset', 'Train\nSamples', 'Test\nSamples',
                              'Test R²', 'Test MAE', 'Test RMSE', 'CV MAE'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(7):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by dataset
    for i, name in enumerate(datasets):
        table[(i+1, 0)].set_facecolor(COLORS[name])
        table[(i+1, 0)].set_text_props(weight='bold')
        for j in range(1, 7):
            table[(i+1, j)].set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
    
    ax.set_title('Performance Summary', fontweight='bold', fontsize=18, pad=20)
    
    save_figure(fig, output_dir, 'fig9_performance_summary_table')


def plot_cv_vs_test_mae(results_dict, output_dir):
    """CV MAE vs Test MAE comparison."""
    datasets = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    cv_mae = [results_dict[name]['cv_mae_mean'].mean() for name in datasets]
    test_mae = [results_dict[name]['test_mae'].mean() for name in datasets]
    
    bars1 = ax.bar(x - width/2, cv_mae, width, label='CV MAE',
                  color='#70AD47', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_mae, width, label='Test MAE',
                  color='#ED7D31', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('MAE', fontweight='bold')
    ax.set_title('Cross-Validation vs Test MAE', fontweight='bold')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(alpha=0.3, axis='y')
    
    save_figure(fig, output_dir, 'fig10_cv_vs_test_mae')


# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def plot_all_comparisons(results_dict, output_dir):
    """Generate all individual publication-quality plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating publication-quality plots...")
    print("="*80)
    
    plot_r2_boxplot(results_dict, output_dir)
    plot_mae_boxplot(results_dict, output_dir)
    plot_rmse_boxplot(results_dict, output_dir)
    plot_r2_violin(results_dict, output_dir)
    plot_mae_vs_r2_scatter(results_dict, output_dir)
    plot_dataset_sizes(results_dict, output_dir)
    plot_mean_r2_barplot(results_dict, output_dir)
    plot_mean_mae_barplot(results_dict, output_dir)
    plot_cv_vs_test_mae(results_dict, output_dir)
    plot_performance_summary(results_dict, output_dir)
    
    print("="*80)
    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nFiles created (PNG + PDF for each):")
    print("  fig1_r2_comparison")
    print("  fig2_mae_comparison")
    print("  fig3_rmse_comparison")
    print("  fig4_r2_distribution")
    print("  fig5_mae_vs_r2_tradeoff")
    print("  fig6_dataset_sizes")
    print("  fig7_mean_r2_comparison")
    print("  fig8_mean_mae_comparison")
    print("  fig9_performance_summary_table")
    print("  fig10_cv_vs_test_mae")


def create_summary_csv(results_dict, output_dir):
    """Create summary CSV."""
    summary_data = []
    
    for name in results_dict.keys():
        df = results_dict[name]
        summary_data.append({
            'dataset': name,
            'n_train_samples': df['n_train_samples'].iloc[0],
            'n_test_samples': df['n_test_samples'].iloc[0],
            'n_train_catalysts': df['n_train_catalysts'].iloc[0],
            'test_r2_mean': df['test_r2'].mean(),
            'test_r2_std': df['test_r2'].std(),
            'test_r2_min': df['test_r2'].min(),
            'test_r2_max': df['test_r2'].max(),
            'test_mae_mean': df['test_mae'].mean(),
            'test_mae_std': df['test_mae'].std(),
            'test_mae_min': df['test_mae'].min(),
            'test_mae_max': df['test_mae'].max(),
            'test_rmse_mean': df['test_rmse'].mean(),
            'test_rmse_std': df['test_rmse'].std(),
            'cv_mae_mean': df['cv_mae_mean'].mean(),
            'cv_mae_std': df['cv_mae_mean'].std(),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(output_dir) / 'comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary CSV saved: {summary_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(result_path1, result_path2, result_path3, dataset_names, output_dir):
    """Compare results from 3 datasets."""
    print("="*80)
    print("COMPARING 3 VIRTUAL CATALYST DATASETS")
    print("PUBLICATION-QUALITY INDIVIDUAL PLOTS")
    print("="*80)
    
    print(f"\nDatasets:")
    print(f"  1. {dataset_names[0]}: {result_path1}")
    print(f"  2. {dataset_names[1]}: {result_path2}")
    print(f"  3. {dataset_names[2]}: {result_path3}")
    print(f"\nOutput: {output_dir}")
    
    # Load results
    print("\n" + "="*80)
    print("LOADING RESULTS")
    print("="*80 + "\n")
    
    results_dict = load_results(
        result_path1, result_path2, result_path3,
        dataset_names
    )
    
    # Create all plots
    plot_all_comparisons(results_dict, output_dir)
    
    # Create summary CSV
    create_summary_csv(results_dict, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    print("\n📊 QUICK SUMMARY:")
    for name in dataset_names:
        df = results_dict[name]
        print(f"\n{name}:")
        print(f"  Test R²:  {df['test_r2'].mean():.4f} ± {df['test_r2'].std():.4f}")
        print(f"  Test MAE: {df['test_mae'].mean():.4f} ± {df['test_mae'].std():.4f}")
    
    print(f"\n✓ All outputs saved to: {output_dir}")
    print("\nFor LaTeX/Overleaf: Use the PDF files for vector graphics!")
    print("="*80)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare results across 3 virtual catalyst datasets - Publication quality"
    )
    
    parser.add_argument(
        "--result_path1",
        type=str,
        required=True,
        help="Path to training_results.csv for dataset 1"
    )
    
    parser.add_argument(
        "--result_path2",
        type=str,
        required=True,
        help="Path to training_results.csv for dataset 2"
    )
    
    parser.add_argument(
        "--result_path3",
        type=str,
        required=True,
        help="Path to training_results.csv for dataset 3"
    )
    
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs=3,
        default=["Top59", "50pct", "100pct"],
        help="Names for the 3 datasets (default: ['Top59', '50pct', '100pct'])"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for comparison plots"
    )
    
    args = parser.parse_args()
    
    main(
        result_path1=args.result_path1,
        result_path2=args.result_path2,
        result_path3=args.result_path3,
        dataset_names=args.dataset_names,
        output_dir=args.output_dir,
    )