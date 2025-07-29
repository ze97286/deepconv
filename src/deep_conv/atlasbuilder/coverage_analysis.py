import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_coverage_distributions(cov_df, control_cov_df, mv_df, control_mv_df):
    """
    Comprehensive coverage analysis for methylation data
    
    Parameters:
    -----------
    cov_df : DataFrame
        Coverage data for cell types/tumours (regions x samples)
    control_cov_df : DataFrame
        Coverage data for controls (regions x samples)
    mv_df : DataFrame
        Marker values for cell types/tumours (regions x samples)
    control_mv_df : DataFrame
        Marker values for controls (regions x samples)
    
    Returns:
    --------
    dict : Dictionary containing all analysis results
    """
    results = {}
    # Get sample columns (excluding metadata)
    metadata_cols = ['name', 'direction']
    sample_cols = [col for col in cov_df.columns if col not in metadata_cols]
    control_sample_cols = [col for col in control_cov_df.columns if col not in metadata_cols]
    # 1. Basic coverage statistics per sample
    print("=== Per-Sample Coverage Statistics ===")
    sample_stats = {}
    for col in sample_cols:
        coverage = cov_df[col].values
        coverage_nonzero = coverage[coverage > 0]
        stats = {
            'mean': np.mean(coverage),
            'median': np.median(coverage),
            'mean_nonzero': np.mean(coverage_nonzero) if len(coverage_nonzero) > 0 else 0,
            'median_nonzero': np.median(coverage_nonzero) if len(coverage_nonzero) > 0 else 0,
            'pct_10': np.percentile(coverage, 10),
            'pct_25': np.percentile(coverage, 25),
            'pct_75': np.percentile(coverage, 75),
            'pct_90': np.percentile(coverage, 90),
            'regions_with_0': np.sum(coverage == 0),
            'regions_with_0_pct': 100 * np.sum(coverage == 0) / len(coverage),
            'regions_under_10': np.sum(coverage < 10),
            'regions_under_10_pct': 100 * np.sum(coverage < 10) / len(coverage),
            'regions_under_50': np.sum(coverage < 50),
            'regions_under_50_pct': 100 * np.sum(coverage < 50) / len(coverage),
            'regions_under_100': np.sum(coverage < 100),
            'regions_under_100_pct': 100 * np.sum(coverage < 100) / len(coverage),
            'total_regions': len(coverage)
        }
        sample_stats[col] = stats
    results['sample_stats'] = pd.DataFrame(sample_stats).T
    # 2. Cell type grouped analysis
    print("\n=== Cell Type Coverage Analysis ===")
    cell_type_groups = defaultdict(list)
    for col in sample_cols:
        cell_type = col.split('_')[0]
        cell_type_groups[cell_type].append(col)
    cell_type_stats = {}
    for cell_type, samples in cell_type_groups.items():
        if len(samples) == 0:
            continue
        # Get coverage for all samples of this cell type
        cell_type_coverage = cov_df[samples].values
        # Calculate statistics across samples
        mean_coverage_per_region = np.mean(cell_type_coverage, axis=1)
        std_coverage_per_region = np.std(cell_type_coverage, axis=1)
        cv_per_region = np.divide(std_coverage_per_region, mean_coverage_per_region, 
                                  out=np.zeros_like(std_coverage_per_region), 
                                  where=mean_coverage_per_region!=0)
        # How many samples cover each region
        regions_covered_by_n_samples = {}
        for n in range(len(samples) + 1):
            count = np.sum(np.sum(cell_type_coverage > 0, axis=1) == n)
            regions_covered_by_n_samples[f'covered_by_{n}_samples'] = count
        stats = {
            'num_samples': len(samples),
            'mean_coverage_across_samples': np.mean(mean_coverage_per_region),
            'median_coverage_across_samples': np.median(mean_coverage_per_region),
            'mean_cv': np.mean(cv_per_region[mean_coverage_per_region > 0]),
            'median_cv': np.median(cv_per_region[mean_coverage_per_region > 0]),
            'regions_with_any_coverage': np.sum(mean_coverage_per_region > 0),
            'regions_with_any_coverage_pct': 100 * np.sum(mean_coverage_per_region > 0) / len(mean_coverage_per_region),
            **regions_covered_by_n_samples
        }
        cell_type_stats[cell_type] = stats
    results['cell_type_stats'] = pd.DataFrame(cell_type_stats).T
    # 3. tumour-specific analysis
    print("\n=== tumour Coverage Analysis ===")
    tumour_samples = [col for col in sample_cols if 'oac' in col.lower()]
    if tumour_samples:
        tumour_coverage = cov_df[tumour_samples].values
        # Total coverage across all tumour samples
        total_tumour_coverage = np.sum(tumour_coverage, axis=1)
        # Number of tumour samples covering each region
        num_tumour_samples_per_region = np.sum(tumour_coverage > 0, axis=1)
        tumour_stats = {
            'num_tumour_samples': len(tumour_samples),
            'mean_total_coverage': np.mean(total_tumour_coverage),
            'median_total_coverage': np.median(total_tumour_coverage),
            'regions_covered_by_all_tumours': np.sum(num_tumour_samples_per_region == len(tumour_samples)),
            'regions_covered_by_all_tumours_pct': 100 * np.sum(num_tumour_samples_per_region == len(tumour_samples)) / len(num_tumour_samples_per_region),
            'regions_covered_by_none': np.sum(num_tumour_samples_per_region == 0),
            'regions_covered_by_at_least_3': np.sum(num_tumour_samples_per_region >= 3),
            'regions_covered_by_at_least_3_pct': 100 * np.sum(num_tumour_samples_per_region >= 3) / len(num_tumour_samples_per_region)
        }
        # Distribution of total tumour coverage
        coverage_thresholds = [50, 100, 200, 500, 1000]
        for thresh in coverage_thresholds:
            tumour_stats[f'regions_total_cov_over_{thresh}'] = np.sum(total_tumour_coverage >= thresh)
            tumour_stats[f'regions_total_cov_over_{thresh}_pct'] = 100 * np.sum(total_tumour_coverage >= thresh) / len(total_tumour_coverage)
        results['tumour_stats'] = tumour_stats
        results['tumour_coverage_per_region'] = total_tumour_coverage
        results['num_tumour_samples_per_region'] = num_tumour_samples_per_region
    # 4. Control cohort analysis
    print("\n=== Control Cohort Analysis ===")
    gi_samples = [col for col in control_sample_cols if col.startswith('GI')]
    other_control_samples = [col for col in control_sample_cols if not col.startswith('GI') and col not in metadata_cols]
    control_cohort_stats = {}
    # Low coverage cohort (GI)
    if gi_samples:
        gi_coverage = control_cov_df[gi_samples].values
        gi_mean = np.mean(gi_coverage, axis=1)
        control_cohort_stats['low_coverage_cohort'] = {
            'num_samples': len(gi_samples),
            'mean_coverage': np.mean(gi_mean),
            'median_coverage': np.median(gi_mean),
            'regions_with_any_coverage': np.sum(gi_mean > 0),
            'regions_with_any_coverage_pct': 100 * np.sum(gi_mean > 0) / len(gi_mean)
        }
    # High coverage cohort
    if other_control_samples:
        other_coverage = control_cov_df[other_control_samples].values
        other_mean = np.mean(other_coverage, axis=1)
        control_cohort_stats['high_coverage_cohort'] = {
            'num_samples': len(other_control_samples),
            'mean_coverage': np.mean(other_mean),
            'median_coverage': np.median(other_mean),
            'regions_with_any_coverage': np.sum(other_mean > 0),
            'regions_with_any_coverage_pct': 100 * np.sum(other_mean > 0) / len(other_mean)
        }
        # Coverage ratio analysis
        if gi_samples:
            both_covered = (gi_mean > 0) & (other_mean > 0)
            coverage_ratio = np.divide(other_mean, gi_mean, 
                                     out=np.zeros_like(other_mean), 
                                     where=both_covered)
            control_cohort_stats['cohort_comparison'] = {
                'regions_covered_in_both': np.sum(both_covered),
                'regions_covered_in_both_pct': 100 * np.sum(both_covered) / len(both_covered),
                'mean_coverage_ratio': np.mean(coverage_ratio[both_covered]),
                'median_coverage_ratio': np.median(coverage_ratio[both_covered]),
                'regions_only_in_high_cov': np.sum((gi_mean == 0) & (other_mean > 0)),
                'regions_only_in_low_cov': np.sum((gi_mean > 0) & (other_mean == 0))
            }
    results['control_cohort_stats'] = control_cohort_stats
    # 5. Cross-analysis: regions with good tumour coverage but minimal control coverage
    if tumour_samples and other_control_samples:
        print("\n=== tumour vs Control Coverage Analysis ===")
        # Define "good" tumour coverage and "minimal" control coverage
        good_tumour_mask = (total_tumour_coverage >= 100) & (num_tumour_samples_per_region >= 3)
        minimal_control_mask = other_mean < 10
        promising_regions = good_tumour_mask & minimal_control_mask
        cross_analysis = {
            'regions_good_tumour_minimal_control': np.sum(promising_regions),
            'regions_good_tumour_minimal_control_pct': 100 * np.sum(promising_regions) / len(promising_regions),
            'regions_good_tumour_coverage': np.sum(good_tumour_mask),
            'regions_minimal_control_coverage': np.sum(minimal_control_mask)
        }
        results['cross_analysis'] = cross_analysis
    return results


def plot_coverage_distributions(cov_df, control_cov_df, results):
    """
    Create visualization plots for coverage distributions
    """
    metadata_cols = ['name', 'direction']
    # 1. Coverage distribution by cell type
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # Get cell types
    sample_cols = [col for col in cov_df.columns if col not in metadata_cols]
    cell_type_groups = defaultdict(list)
    for col in sample_cols:
        cell_type = col.split('_')[0]
        cell_type_groups[cell_type].append(col)
    # Plot 1: Mean coverage by cell type
    ax = axes[0, 0]
    cell_type_means = {}
    for cell_type, samples in cell_type_groups.items():
        if samples:
            mean_cov = np.mean(cov_df[samples].values)
            cell_type_means[cell_type] = mean_cov
    ax.bar(cell_type_means.keys(), cell_type_means.values())
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Mean Coverage')
    ax.set_title('Mean Coverage by Cell Type')
    ax.tick_params(axis='x', rotation=45)
    # Plot 2: Coverage distribution for tumour samples
    ax = axes[0, 1]
    tumour_samples = [col for col in sample_cols if 'oac' in col.lower()]
    if tumour_samples and 'tumour_coverage_per_region' in results:
        tumour_cov = results['tumour_coverage_per_region']
        tumour_cov_nonzero = tumour_cov[tumour_cov > 0]
        ax.hist(np.log10(tumour_cov_nonzero + 1), bins=50, alpha=0.7)
        ax.set_xlabel('log10(Total tumour Coverage + 1)')
        ax.set_ylabel('Number of Regions')
        ax.set_title('Distribution of Total tumour Coverage')
    # Plot 3: Number of tumour samples covering each region
    ax = axes[1, 0]
    if 'num_tumour_samples_per_region' in results:
        num_samples = results['num_tumour_samples_per_region']
        unique, counts = np.unique(num_samples, return_counts=True)
        ax.bar(unique, counts)
        ax.set_xlabel('Number of tumour Samples')
        ax.set_ylabel('Number of Regions')
        ax.set_title('Regions by Number of tumour Samples with Coverage')
    # Plot 4: Control coverage comparison
    ax = axes[1, 1]
    control_sample_cols = [col for col in control_cov_df.columns if col not in metadata_cols]
    gi_samples = [col for col in control_sample_cols if col.startswith('GI')]
    other_samples = [col for col in control_sample_cols if not col.startswith('GI')]
    if gi_samples and other_samples:
        gi_mean = np.mean(control_cov_df[gi_samples].values, axis=1)
        other_mean = np.mean(control_cov_df[other_samples].values, axis=1)
        # Plot coverage comparison
        ax.scatter(np.log10(gi_mean + 1), np.log10(other_mean + 1), 
                  alpha=0.1, s=1)
        ax.set_xlabel('log10(Low Coverage Cohort Mean + 1)')
        ax.set_ylabel('log10(High Coverage Cohort Mean + 1)')
        ax.set_title('Control Coverage: Low vs High Coverage Cohorts')
        ax.plot([0, 4], [0, 4], 'r--', alpha=0.5)  # y=x line
    plt.tight_layout()
    return fig


def generate_coverage_report(results):
    """
    Generate a text report summarizing the coverage analysis
    """
    report = []
    report.append("=== COVERAGE ANALYSIS REPORT ===\n")
    # Sample statistics summary
    if 'sample_stats' in results:
        stats_df = results['sample_stats']
        report.append("SAMPLE COVERAGE SUMMARY:")
        report.append(f"Total samples analyzed: {len(stats_df)}")
        report.append(f"Mean coverage across samples: {stats_df['mean'].mean():.2f}")
        report.append(f"Samples with >50% regions having zero coverage: {sum(stats_df['regions_with_0_pct'] > 50)}")
        report.append("")
    # Cell type statistics
    if 'cell_type_stats' in results:
        ct_stats = results['cell_type_stats']
        report.append("CELL TYPE COVERAGE SUMMARY:")
        for ct, stats in ct_stats.iterrows():
            report.append(f"\n{ct} (n={int(stats['num_samples'])} samples):")
            report.append(f"  - Mean coverage: {stats['mean_coverage_across_samples']:.2f}")
            report.append(f"  - Regions with any coverage: {stats['regions_with_any_coverage_pct']:.1f}%")
            if stats['num_samples'] > 1:
                report.append(f"  - Mean CV across samples: {stats['mean_cv']:.2f}")
    # tumour statistics
    if 'tumour_stats' in results:
        t_stats = results['tumour_stats']
        report.append("\n\ntumour COVERAGE SUMMARY:")
        report.append(f"Number of tumour samples: {t_stats['num_tumour_samples']}")
        report.append(f"Regions covered by all tumours: {t_stats['regions_covered_by_all_tumours_pct']:.1f}%")
        report.append(f"Regions with total coverage ≥100: {t_stats['regions_total_cov_over_100_pct']:.1f}%")
        report.append(f"Regions with total coverage ≥500: {t_stats['regions_total_cov_over_500_pct']:.1f}%")
    # Control cohort statistics
    if 'control_cohort_stats' in results:
        cc_stats = results['control_cohort_stats']
        report.append("\n\nCONTROL COHORT SUMMARY:")
        if 'low_coverage_cohort' in cc_stats:
            lc = cc_stats['low_coverage_cohort']
            report.append(f"Low coverage cohort (GI): {lc['num_samples']} samples, mean coverage {lc['mean_coverage']:.2f}")
        if 'high_coverage_cohort' in cc_stats:
            hc = cc_stats['high_coverage_cohort']
            report.append(f"High coverage cohort: {hc['num_samples']} samples, mean coverage {hc['mean_coverage']:.2f}")
        if 'cohort_comparison' in cc_stats:
            comp = cc_stats['cohort_comparison']
            report.append(f"Mean coverage ratio (high/low): {comp['mean_coverage_ratio']:.1f}x")
    # Cross analysis
    if 'cross_analysis' in results:
        cross = results['cross_analysis']
        report.append("\n\nPROMISING REGIONS:")
        report.append(f"Regions with good tumour coverage AND minimal control coverage: {cross['regions_good_tumour_minimal_control']:,} ({cross['regions_good_tumour_minimal_control_pct']:.2f}%)")
    return "\n".join(report)


# Example usage:
if __name__ == "__main__":
    min_cpgs = 3
    chr = 22
    
    cov = pd.read_csv(f'/users/zetzioni/sharedscratch/loyfer_atlas/pat_by_cell_type/l{min_cpgs}_chr{chr}_coverage.parquet')
    control_cov = pd.read_parquet(f"/users/zetzioni/sharedscratch/loyfer_atlas/controls/l{min_cpgs}_chr{chr}_coverage.parquet")
    mv = pd.read_csv(f'/users/zetzioni/sharedscratch/loyfer_atlas/pat_by_cell_type/l{min_cpgs}_chr{chr}_marker_values.parquet')
    control_mv = pd.read_parquet(f"/users/zetzioni/sharedscratch/loyfer_atlas/controls/l{min_cpgs}_chr{chr}_marker_values.parquet")
    
    # Run analysis
    results = analyze_coverage_distributions(cov, control_cov, mv, control_mv)
    
    # Generate report
    report = generate_coverage_report(results)
    print(report)
    
    # Create plots
    fig = plot_coverage_distributions(cov, control_cov, results)
    plt.savefig('/users/zetzioni/sharedscratch/loyfer_atlas/controls/coverage_analysis.png', dpi=300, bbox_inches='tight')
    
    pass