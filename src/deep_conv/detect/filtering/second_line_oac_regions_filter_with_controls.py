import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # Rename to avoid conflicts
from statsmodels.stats.multitest import multipletests

def prepare_control_data(df_regions, control_mv, control_cov, min_coverage=5):
    """
    Prepare control data with coverage filtering and align with regions
    """
    
    print(f"Input shapes: control_mv {control_mv.shape}, control_cov {control_cov.shape}")
    print(f"Regions: {len(df_regions)}")
    
    # Apply minimum coverage filter
    control_mv_filtered = control_mv.copy()
    control_mv_filtered[control_cov < min_coverage] = np.nan
    
    # Calculate coverage statistics per sample
    coverage_info = {}
    for col in control_cov.columns:
        valid_coverage = control_cov[col].dropna()
        coverage_info[col] = {
            'mean_coverage': valid_coverage.mean(),
            'median_coverage': valid_coverage.median(),
            'coverage_tier': 'high' if valid_coverage.median() >= 10 else 'low'
        }
    
    coverage_df = pd.DataFrame(coverage_info).T
    
    print(f"Coverage tiers: {coverage_df['coverage_tier'].value_counts()}")
    
    return control_mv_filtered, coverage_df

def align_regions_with_controls(df_regions, control_mv_filtered):
    """
    Ensure regions align with control data rows
    """
    
    if len(df_regions) != len(control_mv_filtered):
        print(f"WARNING: Region count mismatch: df_regions has {len(df_regions)}, controls have {len(control_mv_filtered)} rows")
        
        min_regions = min(len(df_regions), len(control_mv_filtered))
        df_regions_aligned = df_regions.iloc[:min_regions].copy()
        control_mv_aligned = control_mv_filtered.iloc[:min_regions].copy()
        
        print(f"Using first {min_regions} regions for alignment")
    else:
        df_regions_aligned = df_regions.copy()
        control_mv_aligned = control_mv_filtered.copy()
    
    df_regions_aligned = df_regions_aligned.reset_index(drop=True)
    control_mv_aligned.index = df_regions_aligned.index
    
    return df_regions_aligned, control_mv_aligned

def analyze_control_performance(df_regions_aligned, control_mv_aligned, coverage_df):
    """
    Analyze each region's performance in controls
    """
    
    results = []
    
    high_cov_samples = coverage_df[coverage_df['coverage_tier'] == 'high'].index
    low_cov_samples = coverage_df[coverage_df['coverage_tier'] == 'low'].index
    
    print(f"Analyzing {len(df_regions_aligned)} regions across {len(control_mv_aligned.columns)} control samples")
    print(f"High coverage samples: {len(high_cov_samples)}, Low coverage samples: {len(low_cov_samples)}")
    
    for idx in df_regions_aligned.index:
        if idx % 5000 == 0:
            print(f"  Processing region {idx:,}/{len(df_regions_aligned):,}")
            
        region = df_regions_aligned.loc[idx]
        control_signals = control_mv_aligned.loc[idx]
        
        # Overall statistics across all controls
        valid_signals = control_signals.dropna()
        n_valid_samples = len(valid_signals)
        
        if n_valid_samples == 0:
            # No valid data for this region
            result = create_empty_result(region, idx)
            results.append(result)
            continue
            
        # Basic statistics
        mean_signal = valid_signals.mean()
        max_signal = valid_signals.max()
        std_signal = valid_signals.std()
        median_signal = valid_signals.median()
        
        # Count of samples with detectable signal
        n_with_signal = (valid_signals > 0.001).sum()  # >0.1% threshold
        n_high_signal = (valid_signals > 0.01).sum()   # >1% threshold
        
        # Coverage-stratified analysis
        high_cov_stats = analyze_coverage_tier(control_signals, high_cov_samples)
        low_cov_stats = analyze_coverage_tier(control_signals, low_cov_samples)
        
        # Statistical tests
        if n_valid_samples >= 3:
            t_stat, p_value = scipy_stats.ttest_1samp(valid_signals, 0)
            p_value_onetail = p_value / 2 if t_stat > 0 else 1.0
        else:
            t_stat, p_value_onetail = np.nan, 1.0
            
        # Wilcoxon signed-rank test
        if n_valid_samples >= 5:
            try:
                w_stat, w_pvalue = scipy_stats.wilcoxon(valid_signals, alternative='greater')
            except:
                w_stat, w_pvalue = np.nan, 1.0
        else:
            w_stat, w_pvalue = np.nan, 1.0
        
        # Coverage robustness
        coverage_robust = assess_coverage_robustness(high_cov_stats, low_cov_stats)
        
        result = {
            'region_idx': idx,
            'region_name': region['name'],
            'chr': region['chr'],
            'start': region['start'],
            'end': region['end'],
            'oac_signal': region['OAC'],
            'oac_coverage': region['OAC_coverage'],
            
            # Control performance
            'n_valid_samples': n_valid_samples,
            'mean_control_signal': mean_signal,
            'max_control_signal': max_signal,
            'median_control_signal': median_signal,
            'std_control_signal': std_signal,
            'n_with_signal': n_with_signal,
            'n_high_signal': n_high_signal,
            'pct_with_signal': n_with_signal / n_valid_samples * 100,
            
            # Statistical tests
            't_stat': t_stat,
            'p_value_ttest': p_value_onetail,
            'w_stat': w_stat,
            'p_value_wilcoxon': w_pvalue,
            
            # Coverage-stratified
            'high_cov_mean': high_cov_stats['mean'],
            'high_cov_max': high_cov_stats['max'],
            'high_cov_n': high_cov_stats['n'],
            'low_cov_mean': low_cov_stats['mean'],
            'low_cov_max': low_cov_stats['max'], 
            'low_cov_n': low_cov_stats['n'],
            'coverage_robust': coverage_robust,
            
            # Signal-to-noise ratio
            'snr_vs_controls': region['OAC'] / (mean_signal + 0.0001),
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def analyze_coverage_tier(control_signals, tier_samples):
    """
    Analyze signals for a specific coverage tier
    """
    if len(tier_samples) == 0:
        return {'mean': np.nan, 'max': np.nan, 'n': 0}
    
    tier_signals = control_signals[tier_samples].dropna()
    
    if len(tier_signals) == 0:
        return {'mean': np.nan, 'max': np.nan, 'n': 0}
    
    return {
        'mean': tier_signals.mean(),
        'max': tier_signals.max(),
        'n': len(tier_signals)
    }

def assess_coverage_robustness(high_cov_stats, low_cov_stats):
    """
    Assess how robust the region is across different coverage levels
    """
    if pd.isna(high_cov_stats['mean']) or pd.isna(low_cov_stats['mean']):
        return 0.5  # Neutral score if can't compare
    
    diff = abs(high_cov_stats['mean'] - low_cov_stats['mean'])
    normalized_diff = min(diff / 0.005, 1.0)
    
    return 1.0 - normalized_diff

def create_empty_result(region, idx):
    """
    Create empty result for regions with no valid control data
    """
    return {
        'region_idx': idx,
        'region_name': region['name'],
        'chr': region['chr'],
        'start': region['start'],
        'end': region['end'],
        'oac_signal': region['OAC'],
        'oac_coverage': region['OAC_coverage'],
        'n_valid_samples': 0,
        'mean_control_signal': np.nan,
        'max_control_signal': np.nan,
        'median_control_signal': np.nan,
        'std_control_signal': np.nan,
        'n_with_signal': 0,
        'n_high_signal': 0,
        'pct_with_signal': 0,
        't_stat': np.nan,
        'p_value_ttest': 1.0,
        'w_stat': np.nan,
        'p_value_wilcoxon': 1.0,
        'high_cov_mean': np.nan,
        'high_cov_max': np.nan,
        'high_cov_n': 0,
        'low_cov_mean': np.nan,
        'low_cov_max': np.nan,
        'low_cov_n': 0,
        'coverage_robust': 0.0,
        'snr_vs_controls': np.inf
    }

def run_stage2_complete(df_regions, control_mv, control_cov, min_coverage=5):
    """
    Complete Stage 2 workflow - FIXED VERSION
    """
    
    print("="*60)
    print("STAGE 2: CONTROL SAMPLE VALIDATION")
    print("="*60)
    
    # Step 1: Data preparation
    print("\nStep 1: Preparing control data...")
    control_mv_filtered, coverage_df = prepare_control_data(
        df_regions, control_mv, control_cov, min_coverage
    )
    
    # Step 2: Align regions
    print("\nStep 2: Aligning regions with control data...")
    df_regions_aligned, control_mv_aligned = align_regions_with_controls(
        df_regions, control_mv_filtered
    )
    
    # Step 3: Statistical analysis
    print("\nStep 3: Analyzing control performance...")
    control_analysis = analyze_control_performance(
        df_regions_aligned, control_mv_aligned, coverage_df
    )
    
    print(f"\nAnalysis complete!")
    print(f"Regions analyzed: {len(control_analysis)}")
    
    return control_analysis, coverage_df

def filter_regions_by_controls(control_analysis, max_mean_signal=0.02, max_max_signal=0.05, min_samples=2):
    """
    Filter regions based on control performance
    """
    
    print("="*60)
    print("FILTERING REGIONS BY CONTROL PERFORMANCE")
    print("="*60)
    
    # Only consider regions with valid control data
    valid_regions = control_analysis[control_analysis['n_valid_samples'] >= min_samples].copy()
    
    print(f"Regions with valid control data: {len(valid_regions)}")
    
    if len(valid_regions) == 0:
        print("❌ No regions have sufficient control data!")
        return pd.DataFrame()
    
    # Apply filtering criteria
    passed_regions = valid_regions[
        (valid_regions['mean_control_signal'] <= max_mean_signal) &  # Mean signal threshold
        (valid_regions['max_control_signal'] <= max_max_signal) &    # Max signal threshold
        (valid_regions['n_valid_samples'] >= min_samples)            # Minimum samples
    ].copy()
    
    print(f"Regions passing control filters: {len(passed_regions)}")
    print(f"Survival rate: {len(passed_regions)/len(valid_regions)*100:.1f}%")
    
    if len(passed_regions) > 0:
        # Calculate quality scores
        passed_regions['stage2_quality_score'] = calculate_stage2_quality_score(passed_regions)
        passed_regions = passed_regions.sort_values('stage2_quality_score', ascending=False)
        
        print(f"\nControl signal statistics:")
        print(f"  Mean control signal: {passed_regions['mean_control_signal'].mean():.4f}")
        print(f"  Max control signal: {passed_regions['max_control_signal'].max():.4f}")
        print(f"  Median SNR: {passed_regions['snr_vs_controls'].median():.1f}")
        
        # Show top 10
        print(f"\nTop 10 regions by quality:")
        top_10 = passed_regions.head(10)[['region_name', 'oac_signal', 'mean_control_signal', 
                                         'max_control_signal', 'snr_vs_controls', 'stage2_quality_score']]
        print(top_10.to_string(index=False))
    
    return passed_regions

def calculate_stage2_quality_score(regions_df):
    """
    Calculate composite quality score for Stage 2
    """
    
    # Normalize components
    signal_score = np.clip((regions_df['oac_signal'] - 0.65) / 0.35, 0, 1)
    
    # Control cleanliness (lower is better)
    control_clean_score = np.clip(1 - (regions_df['mean_control_signal'] / 0.02), 0, 1)
    
    # Statistical significance
    stat_score = np.clip(-np.log10(regions_df['p_value_ttest'].fillna(1.0)) / 3, 0, 1)
    
    # SNR vs controls (log scale, capped)
    snr_score = np.clip(np.log10(regions_df['snr_vs_controls'].fillna(1)) / 3, 0, 1)
    
    # Sample coverage (prefer more samples)
    sample_score = np.clip(regions_df['n_valid_samples'] / 4, 0, 1)
    
    # Weighted combination
    composite_score = (
        0.3 * signal_score +
        0.3 * control_clean_score +
        0.2 * snr_score +
        0.1 * stat_score +
        0.1 * sample_score
    )
    
    return composite_score

def progressive_control_filtering(control_analysis):
    """
    Try different levels of control filtering
    """
    
    strategies = {
        'ultra_strict': {'mean_max': 0.005, 'max_max': 0.01},
        'strict': {'mean_max': 0.01, 'max_max': 0.02}, 
        'moderate': {'mean_max': 0.02, 'max_max': 0.05},
        'lenient': {'mean_max': 0.05, 'max_max': 0.10},
        'very_lenient': {'mean_max': 0.10, 'max_max': 0.20}
    }
    
    results = {}
    
    for strategy_name, params in strategies.items():
        
        filtered = filter_regions_by_controls(
            control_analysis, 
            max_mean_signal=params['mean_max'],
            max_max_signal=params['max_max'],
            min_samples=2
        )
        
        results[strategy_name] = filtered
        print(f"\n{strategy_name}: {len(filtered)} regions")
        
        if len(filtered) >= 400:
            print(f"✅ {strategy_name} provides sufficient regions for final panel!")
            break
        elif len(filtered) >= 100:
            print(f"⚠️  {strategy_name} provides minimal but workable regions")
    
    return results

def run_complete_stage2(df_regions, control_mv, control_cov, min_coverage=5):
    """
    Complete Stage 2: Analysis + Filtering
    """
    
    # Step 1: Analyze control performance
    control_analysis, coverage_info = run_stage2_complete(
        df_regions, control_mv, control_cov, min_coverage
    )
    
    # Step 2: Progressive filtering
    print("\n" + "="*60)
    print("PROGRESSIVE CONTROL FILTERING")
    print("="*60)
    
    filtered_results = progressive_control_filtering(control_analysis)
    
    return control_analysis, filtered_results, coverage_info

def find_overlapping_regions(regions_df, min_overlap_bp=1):
    """
    Group overlapping regions and select the best from each group
    """
    
    print("="*60)
    print("DEDUPLICATING OVERLAPPING REGIONS")  
    print("="*60)
    
    # Sort by chromosome and start position
    regions_sorted = regions_df.sort_values(['chr', 'start', 'end']).reset_index(drop=True)
    
    overlap_groups = []
    current_group = []
    
    for i, region in regions_sorted.iterrows():
        
        # If this is the first region or no overlap with current group
        if not current_group:
            current_group = [i]
        else:
            # Check if this region overlaps with any region in current group
            overlaps_with_group = False
            
            for group_idx in current_group:
                group_region = regions_sorted.loc[group_idx]
                
                # Same chromosome and overlapping coordinates
                if (region['chr'] == group_region['chr'] and
                    not (region['end'] <= group_region['start'] or 
                         region['start'] >= group_region['end'])):
                    
                    # Calculate overlap
                    overlap_start = max(region['start'], group_region['start'])
                    overlap_end = min(region['end'], group_region['end'])
                    overlap_length = overlap_end - overlap_start
                    
                    if overlap_length >= min_overlap_bp:
                        overlaps_with_group = True
                        break
            
            if overlaps_with_group:
                current_group.append(i)
            else:
                # No overlap, save current group and start new one
                if len(current_group) > 0:
                    overlap_groups.append(current_group)
                current_group = [i]
    
    # Don't forget the last group
    if len(current_group) > 0:
        overlap_groups.append(current_group)
    
    print(f"Total regions: {len(regions_sorted)}")
    print(f"Overlap groups found: {len(overlap_groups)}")
    
    # Show overlap statistics
    group_sizes = [len(group) for group in overlap_groups]
    single_regions = sum(1 for size in group_sizes if size == 1)
    multi_regions = len(overlap_groups) - single_regions
    max_overlap = max(group_sizes) if group_sizes else 0
    
    print(f"Independent regions (no overlap): {single_regions}")
    print(f"Overlapping groups: {multi_regions}")
    print(f"Largest overlap group: {max_overlap} regions")
    print(f"Total regions after deduplication: {len(overlap_groups)}")
    
    return overlap_groups, regions_sorted

def select_best_from_overlaps(overlap_groups, regions_sorted, selection_criteria='stage2_quality_score'):
    """
    Select the best region from each overlap group
    """
    
    print(f"\nSelecting best region from each group using: {selection_criteria}")
    
    selected_regions = []
    
    for group in overlap_groups:
        group_regions = regions_sorted.loc[group]
        
        # Select best region based on criteria
        if selection_criteria in group_regions.columns:
            best_idx = group_regions[selection_criteria].idxmax()
        else:
            # Fallback to OAC signal
            best_idx = group_regions['oac_signal'].idxmax()
        
        selected_regions.append(regions_sorted.loc[best_idx])
    
    deduplicated_regions = pd.DataFrame(selected_regions).reset_index(drop=True)
    
    print(f"Selected {len(deduplicated_regions)} non-overlapping regions")
    
    # Show some statistics about what we kept vs removed
    print(f"\nDeduplication impact:")
    print(f"  Original regions: {len(regions_sorted)}")
    print(f"  Deduplicated regions: {len(deduplicated_regions)}")
    print(f"  Reduction: {len(regions_sorted) - len(deduplicated_regions)} regions ({(len(regions_sorted) - len(deduplicated_regions))/len(regions_sorted)*100:.1f}%)")
    
    return deduplicated_regions

def analyze_overlap_patterns(overlap_groups, regions_sorted):
    """
    Analyze the patterns of overlapping regions
    """
    
    print("\n" + "="*40)
    print("OVERLAP PATTERN ANALYSIS")
    print("="*40)
    
    # Group size distribution
    group_sizes = [len(group) for group in overlap_groups]
    size_counts = pd.Series(group_sizes).value_counts().sort_index()
    
    print("Overlap group size distribution:")
    for size, count in size_counts.items():
        if size == 1:
            print(f"  {count} independent regions (no overlap)")
        else:
            print(f"  {count} groups with {size} overlapping regions each")
    
    # Show examples of large overlap groups
    large_groups = [group for group in overlap_groups if len(group) >= 5]
    
    if large_groups:
        print(f"\nExamples of large overlap groups:")
        for i, group in enumerate(large_groups[:3]):  # Show first 3
            group_regions = regions_sorted.loc[group]
            chr_name = group_regions['chr'].iloc[0]
            start_range = f"{group_regions['start'].min()}-{group_regions['end'].max()}"
            print(f"  Group {i+1}: {len(group)} regions on {chr_name}:{start_range}")
            
            # Show range of quality scores in this group
            scores = group_regions['stage2_quality_score']
            print(f"    Quality scores: {scores.min():.3f} - {scores.max():.3f}")


def analyze_final_distribution(final_regions, title="Final Regions"):
    """
    Analyze the distribution of final selected regions
    """
    
    print(f"\n" + "="*50)
    print(f"{title.upper()} ANALYSIS")
    print("="*50)
    
    print(f"Total regions: {len(final_regions)}")
    
    # Chromosome distribution
    print(f"\nChromosome distribution:")
    chr_dist = final_regions['chr'].value_counts().sort_index()
    for chr_name, count in chr_dist.items():
        print(f"  {chr_name}: {count}")
    
    # Quality metrics
    print(f"\nQuality metrics:")
    print(f"  OAC signal: {final_regions['oac_signal'].min():.3f} - {final_regions['oac_signal'].max():.3f}")
    print(f"  Mean control signal: {final_regions['mean_control_signal'].min():.5f} - {final_regions['mean_control_signal'].max():.5f}")
    print(f"  SNR: {final_regions['snr_vs_controls'].min():.1f} - {final_regions['snr_vs_controls'].max():.1f}")
    print(f"  Quality score: {final_regions['stage2_quality_score'].min():.3f} - {final_regions['stage2_quality_score'].max():.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='admix synthetic samples')
    parser.add_argument('--oac_first_line_atlas', type=str, required=True, help='file name for oac first line filtered atlas')
    parser.add_argument('--control_dir', type=str, required=True, help='Directory containing control parquet file for marker values and coverage')
    parser.add_argument('--out_atlas_name', type=str, required=True, help='Atlas name for the filtered output')
    args = parser.parse_args()

    broad_filtered_df = pd.read_csv(args.oac_first_line_atlas, sep="\t")
    broad_filtered_df = broad_filtered_df.drop_duplicates(["name"])

    # Run the analysis
    control_mv = pd.read_parquet(f"{args.control_dir}/marker_values.parquet")
    control_cov = pd.read_parquet(f"{args.control_dir}/coverage.parquet")

    control_analysis, filtered_results, coverage_info= run_complete_stage2(
        broad_filtered_df, control_mv[control_mv.columns[8:]], control_cov[control_cov.columns[8:]], min_coverage=5
    )

    # Run the deduplication
    overlap_groups, regions_sorted = find_overlapping_regions(filtered_results['ultra_strict'])
    analyze_overlap_patterns(overlap_groups, regions_sorted)
    deduplicated_regions = select_best_from_overlaps(overlap_groups, regions_sorted)

    # Analyze the deduplicated results
    analyze_final_distribution(deduplicated_regions, "Deduplicated Regions")
    names = set(deduplicated_regions.region_name.unique())
    final_markers = broad_filtered_df[broad_filtered_df.name.isin(names)]

    print("==== detailed analysis ====")
    print(control_analysis)
    print("==== coverage info ====")
    print(coverage_info)
    print("==== stats ====")

    final_markers[
        [
            "chr",
            "start",
            "end",
            "startCpG",
            "endCpG",
            "target",
            "name",
            "direction",
            "B-cells",
            "CD34-erythroblasts",
            "CD34-megakaryocytes",
            "Colon",
            "Esophagus",
            "Gastric",
            "Granulocytes",
            "Monocytes",
            "NK-cells",
            "OAC",
            "Small-intestine",
            "T-cells",
        ]
    ].to_csv(args.out_atlas_name, sep="\t", index=False)

if __name__ == "__main__":
    main()
