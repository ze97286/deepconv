import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse

def calculate_100_percent_tumor_signal(signal_df: pd.DataFrame, 
                                     tumor_purity_dict: Dict[str, float],
                                     tumor_prefix: str = 'OAC') -> pd.Series:
    """
    Calculate expected signal for 100% tumor purity using linear regression
    on tumor samples with known purity.
    """
    # Get tumor sample columns
    tumor_cols = [col for col in signal_df.columns if col.startswith(tumor_prefix)]
    # Extract tumor signals and purities
    tumor_signals = signal_df[tumor_cols]
    purities = [tumor_purity_dict[col] for col in tumor_cols]
    # Calculate 100% tumor signal for each region
    tumor_100_signals = []
    for idx in signal_df.index:
        signals = tumor_signals.loc[idx].values
        # Remove NaN values
        mask = ~np.isnan(signals)
        if mask.sum() < 3:  # Need at least 3 points for reliable estimation
            tumor_100_signals.append(np.nan)
            continue
        valid_signals = signals[mask]
        valid_purities = np.array(purities)[mask]
        # Linear regression: signal = slope * purity + intercept
        # For 100% purity, signal = slope * 1.0 + intercept
        A = np.vstack([valid_purities, np.ones(len(valid_purities))]).T
        slope, intercept = np.linalg.lstsq(A, valid_signals, rcond=None)[0]
        # Calculate signal at 100% purity
        signal_100 = slope * 1.0 + intercept
        # Only accept if slope is positive and signal is reasonable
        if slope > 0 and 0 < signal_100 <= 1:
            tumor_100_signals.append(signal_100)
        else:
            tumor_100_signals.append(np.nan)
    return pd.Series(tumor_100_signals, index=signal_df.index)

def analyse_thresholds(signal_df: pd.DataFrame, coverage_df: pd.DataFrame, tumour_purity_dict: Dict[str, float]):
    """
    Analyze the data to suggest appropriate filtering thresholds using weighted cell type signals
    """
    print("="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)
    
    # Sample data for faster analysis (use 50k regions instead of 2M)
    sample_size = min(50000, len(signal_df))
    sample_df = signal_df.sample(n=sample_size, random_state=42).copy()
    sample_coverage = coverage_df.loc[sample_df.index].copy()
    
    print(f"Analyzing sample of {sample_size} regions...")
    
    # Calculate 100% tumor purity signal
    print("Calculating 100% tumor purity signals...")
    sample_df['tumor_100'] = calculate_100_percent_tumor_signal(
        sample_df, tumour_purity_dict, 'OAC'
    )
    
    # Get regions with valid tumor signal for analysis
    valid_tumor = sample_df[~sample_df['tumor_100'].isna()].copy()
    print(f"Found {len(valid_tumor)} regions with valid tumor signal")
    
    if len(valid_tumor) == 0:
        print("No regions with valid tumor signal found!")
        return
    
    # Calculate weighted cell type signals (the actual values used in filtering)
    print("Calculating weighted cell type signals for threshold analysis...")
    cell_type_order = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                      "Colon", "Esophagus", "Gastric", "Granulocytes", 
                      "Monocytes", "NK-cells", "Small-intestine", "T-cells"]
    
    # Add target column for calculate_weighted_cell_type_signals
    valid_tumor['target'] = 'OAC'
    valid_coverage = sample_coverage.loc[valid_tumor.index]
    
    merged_signals, _ = calculate_weighted_cell_type_signals(
        valid_tumor, valid_coverage, cell_type_order
    )
    
    # Calculate control signals
    control_cols = [col for col in sample_df.columns if col.startswith('Control')]
    if control_cols:
        valid_tumor['median_control'] = valid_tumor[control_cols].median(axis=1, skipna=True)
        valid_tumor['max_control'] = valid_tumor[control_cols].max(axis=1, skipna=True)
        valid_tumor['median_control'] = valid_tumor['median_control'].fillna(0)
        valid_tumor['max_control'] = valid_tumor['max_control'].fillna(0)
    else:
        valid_tumor['median_control'] = 0
        valid_tumor['max_control'] = 0
    
    # Calculate coverage
    tumor_cols = [col for col in sample_df.columns if col.startswith('OAC')]
    valid_tumor['tumor_coverage'] = sample_coverage[tumor_cols].mean(axis=1)
    
    # Calculate weighted blood/immune and GI signals
    blood_immune_types = ['Granulocytes', 'T-cells', 'B-cells', 'NK-cells', 
                         'Monocytes', 'CD34-erythroblasts', 'CD34-megakaryocytes']
    gi_types = ['Colon', 'Esophagus', 'Gastric', 'Small-intestine']
    
    blood_cols = [col for col in blood_immune_types if col in merged_signals.columns]
    gi_cols = [col for col in gi_types if col in merged_signals.columns]
    
    if blood_cols:
        valid_tumor['max_blood_immune'] = merged_signals[blood_cols].max(axis=1)
        valid_tumor['median_blood_immune'] = merged_signals[blood_cols].median(axis=1)
    else:
        valid_tumor['max_blood_immune'] = 0
        valid_tumor['median_blood_immune'] = 0
        
    if gi_cols:
        valid_tumor['max_gi'] = merged_signals[gi_cols].max(axis=1)
    else:
        valid_tumor['max_gi'] = 0
    
    print(f"Analysis of {len(valid_tumor)} sampled regions with valid 100% tumor signal:")
    print(f"Found {len(blood_cols)} blood/immune and {len(gi_cols)} GI cell types\n")
    
    # Tumor signal analysis
    print("100% Tumor Signal:")
    print(f"  Min: {valid_tumor['tumor_100'].min():.3f}")
    print(f"  25th percentile: {valid_tumor['tumor_100'].quantile(0.25):.3f}")
    print(f"  Median: {valid_tumor['tumor_100'].median():.3f}")
    print(f"  75th percentile: {valid_tumor['tumor_100'].quantile(0.75):.3f}")
    print(f"  Max: {valid_tumor['tumor_100'].max():.3f}")
    # More reasonable tumor signal threshold (50th percentile)
    print(f"  Suggested min_tumor_signal: {valid_tumor['tumor_100'].quantile(0.50):.3f}")
    
    # Blood/immune analysis (using weighted signals)
    print("\nWeighted Blood/Immune Signal:")
    print(f"  Min: {valid_tumor['median_blood_immune'].min():.4f}")
    print(f"  25th percentile: {valid_tumor['median_blood_immune'].quantile(0.25):.4f}")
    print(f"  Median: {valid_tumor['median_blood_immune'].median():.4f}")
    print(f"  75th percentile: {valid_tumor['median_blood_immune'].quantile(0.75):.4f}")
    print(f"  Max: {valid_tumor['median_blood_immune'].max():.4f}")
    # More reasonable blood signal threshold (10th percentile)
    print(f"  Suggested max_blood_signal: {valid_tumor['median_blood_immune'].quantile(0.10):.4f}")
    
    # GI analysis (using weighted signals)
    print("\nWeighted GI Signal:")
    print(f"  Min: {valid_tumor['max_gi'].min():.4f}")
    print(f"  25th percentile: {valid_tumor['max_gi'].quantile(0.25):.4f}")
    print(f"  Median: {valid_tumor['max_gi'].median():.4f}")
    print(f"  75th percentile: {valid_tumor['max_gi'].quantile(0.75):.4f}")
    print(f"  Max: {valid_tumor['max_gi'].max():.4f}")
    # More reasonable GI signal threshold (50th percentile)
    print(f"  Suggested max_gi_signal: {valid_tumor['max_gi'].quantile(0.50):.4f}")
    
    # Control analysis
    print("\nControl Signal:")
    print(f"  Median control - Min: {valid_tumor['median_control'].min():.4f}")
    print(f"  Median control - 5th percentile: {valid_tumor['median_control'].quantile(0.05):.4f}")
    print(f"  Median control - 10th percentile: {valid_tumor['median_control'].quantile(0.10):.4f}")
    print(f"  Median control - 25th percentile: {valid_tumor['median_control'].quantile(0.25):.4f}")
    print(f"  Median control - Median: {valid_tumor['median_control'].median():.4f}")
    
    print(f"  Max control - Min: {valid_tumor['max_control'].min():.4f}")
    print(f"  Max control - 5th percentile: {valid_tumor['max_control'].quantile(0.05):.4f}")
    print(f"  Max control - 10th percentile: {valid_tumor['max_control'].quantile(0.10):.4f}")
    print(f"  Max control - 25th percentile: {valid_tumor['max_control'].quantile(0.25):.4f}")
    print(f"  Max control - Median: {valid_tumor['max_control'].median():.4f}")
    
    # Show percentage of regions with very low control signal
    very_low_control_med = (valid_tumor['median_control'] <= 0.001).sum()
    low_control_med = (valid_tumor['median_control'] <= 0.01).sum()
    very_low_control_max = (valid_tumor['max_control'] <= 0.001).sum()
    low_control_max = (valid_tumor['max_control'] <= 0.01).sum()
    
    print(f"  Regions with median control ≤ 0.001: {very_low_control_med}/{len(valid_tumor)} ({100*very_low_control_med/len(valid_tumor):.1f}%)")
    print(f"  Regions with median control ≤ 0.01: {low_control_med}/{len(valid_tumor)} ({100*low_control_med/len(valid_tumor):.1f}%)")
    print(f"  Regions with max control ≤ 0.001: {very_low_control_max}/{len(valid_tumor)} ({100*very_low_control_max/len(valid_tumor):.1f}%)")
    print(f"  Regions with max control ≤ 0.01: {low_control_max}/{len(valid_tumor)} ({100*low_control_max/len(valid_tumor):.1f}%)")
    
    # Coverage analysis
    print("\nTumor Coverage:")
    print(f"  Min: {valid_tumor['tumor_coverage'].min():.1f}")
    print(f"  25th percentile: {valid_tumor['tumor_coverage'].quantile(0.25):.1f}")
    print(f"  Median: {valid_tumor['tumor_coverage'].median():.1f}")
    print(f"  75th percentile: {valid_tumor['tumor_coverage'].quantile(0.75):.1f}")
    print(f"  Max: {valid_tumor['tumor_coverage'].max():.1f}")
    print(f"  Suggested min_coverage: {valid_tumor['tumor_coverage'].quantile(0.25):.0f}")
    
    # Realistic suggested command
    print("\n" + "="*60)
    print("REALISTIC SUGGESTED COMMAND:")
    print("="*60)
    print(f"python script.py \\")
    print(f"  --min_tumor_signal {valid_tumor['tumor_100'].quantile(0.50):.3f} \\")
    print(f"  --max_blood_signal {valid_tumor['median_blood_immune'].quantile(0.10):.4f} \\")
    print(f"  --max_gi_signal {valid_tumor['max_gi'].quantile(0.50):.4f} \\")
    print(f"  --max_control_median 0.001 \\")
    print(f"  --max_control_max 0.01 \\")
    print(f"  --min_coverage {valid_tumor['tumor_coverage'].quantile(0.25):.0f} \\")
    print(f"  --no_overlap")

def calculate_weighted_cell_type_signals(signal_df: pd.DataFrame, 
                                       coverage_df: pd.DataFrame,
                                       cell_type_order: List[str],
                                       max_cv_threshold: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate weighted average signal for each cell type.
    For each cell type, signal = sum(signal × coverage) / sum(coverage) across all samples.
    Returns both the result and variance statistics.
    """
    result_df = pd.DataFrame(index=signal_df.index)
    variance_stats = pd.DataFrame(index=signal_df.index)
    # Copy metadata columns
    metadata_cols = ['chr', 'start', 'end', 'name', 'direction', 'startCpG', 'endCpG', 'target']
    for col in metadata_cols:
        if col in signal_df.columns:
            result_df[col] = signal_df[col]
            variance_stats[col] = signal_df[col]
    # Calculate weighted signals for each cell type
    for cell_type in cell_type_order:
        if cell_type == 'OAC':
            # For OAC, use the pre-calculated 100% tumor purity signal
            result_df[cell_type] = signal_df['tumor_100']
            variance_stats[f'{cell_type}_cv'] = 0  # No variance for single calculated value
        else:
            # Find all samples for this cell type
            cell_samples = [col for col in signal_df.columns 
                          if col.startswith(cell_type) and col not in metadata_cols and not col.endswith('_coverage')]
            if cell_samples:
                # Calculate weighted average and variance stats
                weighted_signals = []
                cvs = []
                for idx in signal_df.index:
                    signals = []
                    coverages = []
                    for sample in cell_samples:
                        if sample in signal_df.columns and sample in coverage_df.columns:
                            sig = signal_df.loc[idx, sample]
                            cov = coverage_df.loc[idx, sample]
                            if pd.notna(sig) and pd.notna(cov) and cov > 0:
                                signals.append(sig)
                                coverages.append(cov)
                    if len(signals) == 0:
                        weighted_signals.append(np.nan)
                        cvs.append(np.nan)
                        continue
                    # Calculate coefficient of variation
                    if len(signals) > 1:
                        cv = np.std(signals) / (np.mean(signals) + 1e-10)
                        cvs.append(cv)
                        # Apply variance filter if threshold is set
                        if max_cv_threshold is not None and cv > max_cv_threshold:
                            weighted_signals.append(np.nan)
                            continue
                    else:
                        cvs.append(0)  # Single sample, no variance
                    # Calculate weighted average
                    weighted_sum = sum(s * c for s, c in zip(signals, coverages))
                    coverage_sum = sum(coverages)
                    if coverage_sum > 0:
                        weighted_signals.append(weighted_sum / coverage_sum)
                    else:
                        weighted_signals.append(np.nan)
                result_df[cell_type] = weighted_signals
                variance_stats[f'{cell_type}_cv'] = cvs
            else:
                # No samples for this cell type
                result_df[cell_type] = np.nan
                variance_stats[f'{cell_type}_cv'] = np.nan
    return result_df, variance_stats

def select_non_overlapping_regions(filtered_df: pd.DataFrame, 
                                 min_distance: int = 100) -> pd.DataFrame:
    """
    Select non-overlapping regions with highest quality scores.
    """
    # Sort by quality score (tumor_100 signal * SNR)
    filtered_df['quality_score'] = (
        filtered_df['tumor_100'] * 
        np.sqrt(filtered_df['snr_vs_blood'] * filtered_df['snr_vs_gi'])
    )
    sorted_df = filtered_df.sort_values('quality_score', ascending=False).copy()
    selected_indices = []
    selected_regions = []
    for idx, row in sorted_df.iterrows():
        chr_name = row['chr']
        start = row['start']
        end = row['end']
        # Check overlap with already selected regions
        overlaps = False
        for sel_chr, sel_start, sel_end in selected_regions:
            if chr_name == sel_chr:
                # Check if regions are too close
                if not (end + min_distance < sel_start or start > sel_end + min_distance):
                    overlaps = True
                    break
        if not overlaps:
            selected_indices.append(idx)
            selected_regions.append((chr_name, start, end))
    return filtered_df.loc[selected_indices].copy()

def unified_marker_filtering(signal_df: pd.DataFrame,
                           coverage_df: pd.DataFrame,
                           tumor_purity_dict: Dict[str, float],
                           min_tumor_signal: float = 0.6,
                           min_coverage: int = 10,
                           max_blood_signal: float = 0.001,
                           max_gi_signal: float = 0.2,
                           max_gi_to_tumor_ratio: float = 0.25,
                           max_control_median: float = 0.001,
                           max_control_max: float = 0.01,
                           min_snr_blood: float = 10.0,
                           min_snr_gi: float = 3.0,
                           select_non_overlapping: bool = True,
                           max_cv_threshold: float = None) -> pd.DataFrame:
    """
    Unified filtering combining all criteria in a single step.
    Now uses merged cell type signals for differential methylation filtering.
    """
    print("="*60)
    print("UNIFIED MARKER FILTERING")
    print("="*60)
    
    # Step 1: Calculate 100% tumor purity signal
    print("Calculating 100% tumor purity signals...")
    signal_df['tumor_100'] = calculate_100_percent_tumor_signal(
        signal_df, tumor_purity_dict, 'OAC'
    )
    
    # Step 2: Calculate merged cell type signals with variance filtering (excluding controls)
    print("Calculating merged cell type signals...")
    cell_type_order = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                      "Colon", "Esophagus", "Gastric", "Granulocytes", 
                      "Monocytes", "NK-cells", "Small-intestine", "T-cells"]
    
    # Add target column for calculate_weighted_cell_type_signals
    signal_df['target'] = 'OAC'
    
    merged_signals, variance_stats = calculate_weighted_cell_type_signals(
        signal_df, coverage_df, cell_type_order, max_cv_threshold
    )
    
    print(f"Merged signals shape: {merged_signals.shape}")
    print(f"Merged signals columns: {list(merged_signals.columns)}")
    
    # Step 3: Calculate coverage requirements
    tumor_cols = [col for col in signal_df.columns if col.startswith('OAC')]
    signal_df['tumor_coverage'] = coverage_df[tumor_cols].mean(axis=1)
    
    # Step 4: Calculate control signals (strict approach: median ≤ 0.1%, max ≤ 1%)
    control_cols = [col for col in signal_df.columns if col.startswith('Control')]
    if control_cols:
        # Calculate both median and max control signals
        signal_df['median_control'] = signal_df[control_cols].median(axis=1, skipna=True)
        signal_df['max_control'] = signal_df[control_cols].max(axis=1, skipna=True)
        # If all controls are NaN, set to 0
        signal_df['median_control'] = signal_df['median_control'].fillna(0)
        signal_df['max_control'] = signal_df['max_control'].fillna(0)
    else:
        signal_df['median_control'] = 0
        signal_df['max_control'] = 0
    
    # Step 5: Calculate differential methylation metrics using merged signals
    # Group blood/immune cell types
    blood_immune_types = ['Granulocytes', 'T-cells', 'B-cells', 'NK-cells', 
                         'Monocytes', 'CD34-erythroblasts', 'CD34-megakaryocytes']
    gi_types = ['Colon', 'Esophagus', 'Gastric', 'Small-intestine']
    
    # Calculate max signals from merged cell type signals
    blood_cols = [col for col in blood_immune_types if col in merged_signals.columns]
    gi_cols = [col for col in gi_types if col in merged_signals.columns]
    
    if blood_cols:
        signal_df['max_blood_immune'] = merged_signals[blood_cols].max(axis=1)
        signal_df['median_blood_immune'] = merged_signals[blood_cols].median(axis=1)
    else:
        signal_df['max_blood_immune'] = 0
        signal_df['median_blood_immune'] = 0
        
    if gi_cols:
        signal_df['max_gi'] = merged_signals[gi_cols].max(axis=1)
    else:
        signal_df['max_gi'] = 0
    
    # Step 6: Calculate ratios and SNR using merged signals
    signal_df['gi_to_tumor_ratio'] = signal_df['max_gi'] / (signal_df['tumor_100'] + 1e-10)
    signal_df['snr_vs_blood'] = signal_df['tumor_100'] / (signal_df['max_blood_immune'] + 1e-10)
    signal_df['snr_vs_gi'] = signal_df['tumor_100'] / (signal_df['max_gi'] + 1e-10)
    
    print(f"Found {len(blood_cols)} blood/immune cell types with valid signals")
    print(f"Found {len(gi_cols)} GI cell types with valid signals") 
    print(f"Found {len(control_cols)} control samples")
    print(f"Found {len(tumor_cols)} tumor samples")
    
    # Step 7: Apply unified filtering criteria with detailed breakdown
    print("\nApplying filters step by step:")
    
    # Start with regions that have valid tumor signal
    step1 = signal_df[~signal_df['tumor_100'].isna()]
    print(f"  Regions with valid 100% tumor signal: {len(step1):,}")
    
    # Strong tumor signal
    step2 = step1[step1['tumor_100'] >= min_tumor_signal]
    print(f"  After tumor signal ≥ {min_tumor_signal}: {len(step2):,}")
    
    # Sufficient coverage
    step3 = step2[step2['tumor_coverage'] >= min_coverage]
    print(f"  After coverage ≥ {min_coverage}: {len(step3):,}")
    
    # Control signal filter - both median and max
    step4a = step3[step3['median_control'] <= max_control_median]
    print(f"  After control median ≤ {max_control_median}: {len(step4a):,}")
    
    step4 = step4a[step4a['max_control'] <= max_control_max]
    print(f"  After control max ≤ {max_control_max}: {len(step4):,}")
    
    # Blood/immune filter
    step5 = step4[step4['median_blood_immune'] <= max_blood_signal]
    print(f"  After blood signal ≤ {max_blood_signal}: {len(step5):,}")
    
    # Blood SNR filter
    step6 = step5[step5['snr_vs_blood'] >= min_snr_blood]
    print(f"  After blood SNR ≥ {min_snr_blood}: {len(step6):,}")
    
    # GI signal filter
    step7 = step6[step6['max_gi'] <= max_gi_signal]
    print(f"  After GI signal ≤ {max_gi_signal}: {len(step7):,}")
    
    # GI ratio filter
    step8 = step7[step7['gi_to_tumor_ratio'] <= max_gi_to_tumor_ratio]
    print(f"  After GI/tumor ratio ≤ {max_gi_to_tumor_ratio}: {len(step8):,}")
    
    # GI SNR filter
    filtered = step8[step8['snr_vs_gi'] >= min_snr_gi]
    print(f"  After GI SNR ≥ {min_snr_gi}: {len(filtered):,}")
    
    filtered = filtered.copy()
    
    print(f"\nFiltering results:")
    print(f"  Initial regions: {len(signal_df):,}")
    print(f"  Regions with valid 100% tumor signal: {(~signal_df['tumor_100'].isna()).sum():,}")
    print(f"  Regions passing all filters: {len(filtered):,}")
    
    # Select non-overlapping regions if requested
    if select_non_overlapping and len(filtered) > 0:
        filtered = select_non_overlapping_regions(filtered)
        print(f"  Non-overlapping regions selected: {len(filtered):,}")
    
    # Quality statistics
    if len(filtered) > 0:
        print(f"\nQuality distribution of filtered regions:")
        print(f"  100% tumor signal: {filtered['tumor_100'].min():.3f} - {filtered['tumor_100'].max():.3f}")
        print(f"  Max blood/immune: {filtered['max_blood_immune'].min():.4f} - {filtered['max_blood_immune'].max():.4f}")
        print(f"  Max GI: {filtered['max_gi'].min():.3f} - {filtered['max_gi'].max():.3f}")
        print(f"  Median control: {filtered['median_control'].min():.4f} - {filtered['median_control'].max():.4f}")
        print(f"  Max control: {filtered['max_control'].min():.4f} - {filtered['max_control'].max():.4f}")
        print(f"  SNR vs blood: {filtered['snr_vs_blood'].min():.1f} - {filtered['snr_vs_blood'].max():.1f}")
        print(f"  SNR vs GI: {filtered['snr_vs_gi'].min():.1f} - {filtered['snr_vs_gi'].max():.1f}")
    
    return filtered

def main():
    parser = argparse.ArgumentParser(description='Unified marker filtering')
    parser.add_argument('--signal_file', type=str, required=True, 
                       help='Path to signal matrix parquet file')
    parser.add_argument('--coverage_file', type=str, required=True,
                       help='Path to coverage matrix parquet file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output filtered regions file')
    parser.add_argument('--atlas', type=str, required=True, help='Input atlas file')
    
    # Filtering thresholds
    parser.add_argument('--min_tumor_signal', type=float, default=0.6,
                       help='Minimum tumor signal at 100% purity')
    parser.add_argument('--min_coverage', type=int, default=10,
                       help='Minimum coverage required')
    parser.add_argument('--max_blood_signal', type=float, default=0.001,
                       help='Maximum median blood/immune signal')
    parser.add_argument('--max_gi_signal', type=float, default=0.2,
                       help='Maximum GI tissue signal')
    parser.add_argument('--max_control_median', type=float, default=0.001,
                       help='Maximum median control signal (default: 0.001 = 0.1%)')
    parser.add_argument('--max_control_max', type=float, default=0.01,
                       help='Maximum individual control signal (default: 0.01 = 1%)')
    parser.add_argument('--max_cv_threshold', type=float, default=0.0,
                       help='Maximum coefficient of variation for within-cell-type samples (0 = no filtering)')
    parser.add_argument('--no_overlap', action='store_true',
                       help='Select non-overlapping regions')
    parser.add_argument('--analyse_thresholds', action='store_true',
                       help='Analyze data to suggest appropriate thresholds')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading signal matrix from {args.signal_file}...")
    signal_df = pd.read_parquet(args.signal_file)
    
    print(f"Loading coverage matrix from {args.coverage_file}...")
    coverage_df = pd.read_parquet(args.coverage_file)
    
    atlas = pd.read_csv(args.atlas, sep="\t")
    signal_df[atlas.columns[:8]] = atlas[atlas.columns[:8]]
    
    # Load tumor purity
    tumour_purity_dict = {
        'OAC_069-009_ScrBsl_tumour_cna_corrected':0.5171,
        'OAC_071-011_ScrBsl_tumour_cna_corrected':0.2361,
        'OAC_071-014_ScrBsl_tumour_cna_corrected':0.07926,
        'OAC_071-021_ScrBsl_tumour_cna_corrected':0.4766,
        'OAC_071-022_ScrBsl_tumour_cna_corrected':0.4108,
        'OAC_071-030_ScrBsl_tumour_cna_corrected':0.0801,
        'OAC_071-043_ScrBsl_tumour_cna_corrected':0.4607,
        'OAC_129-001_ScrBsl_tumour_cna_corrected':0.6921
    }
    
    # If analyse_thresholds is requested, do that and exit
    if args.analyse_thresholds:
        analyse_thresholds(signal_df, coverage_df, tumour_purity_dict)
        return
    
    # Apply filtering (without overlap selection first)
    filtered_df = unified_marker_filtering(
        signal_df, coverage_df, tumour_purity_dict,
        min_tumor_signal=args.min_tumor_signal,
        min_coverage=args.min_coverage,
        max_blood_signal=args.max_blood_signal,
        max_gi_signal=args.max_gi_signal,
        max_control_median=args.max_control_median,
        max_control_max=args.max_control_max,
        select_non_overlapping=False,  # Don't select non-overlapping yet
        max_cv_threshold=args.max_cv_threshold if args.max_cv_threshold > 0 else None
    )

    
    # The filtered_df already has the target column and merged signals from unified_marker_filtering
    # Just need to get the final atlas format with all cell types
    cell_type_order = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                      "Colon", "Esophagus", "Gastric", "Granulocytes", 
                      "Monocytes", "NK-cells", "OAC", "Small-intestine", "T-cells"]
    
    print("\nCalculating final weighted cell type signals for atlas...")
    atlas_df, variance_stats = calculate_weighted_cell_type_signals(
        filtered_df, coverage_df, cell_type_order, args.max_cv_threshold if args.max_cv_threshold > 0 else None
    )
    
    # Now apply non-overlapping selection if requested
    if args.no_overlap and len(atlas_df) > 0:
        print(f"Applying non-overlapping selection to {len(atlas_df)} regions...")
        atlas_df = select_non_overlapping_regions(atlas_df)
        print(f"Selected {len(atlas_df)} non-overlapping regions")
    
    # Ensure proper column order
    metadata_cols = ['chr', 'start', 'end', 'name', 'direction', 'startCpG', 'endCpG', 'target']
    output_cols = metadata_cols + cell_type_order
    atlas_df = atlas_df[output_cols]
    
    # Save results
    atlas_df.to_csv(args.output_file, sep='\t', index=False)
    print(f"\nSaved {len(atlas_df)} filtered regions to {args.output_file}")
    
    # Print summary statistics
    print(f"\nAtlas summary:")
    print(f"  Total regions: {len(atlas_df)}")
    
    # Analyze variance statistics
    print(f"\nWithin-cell-type variance analysis:")
    for cell_type in cell_type_order:
        if cell_type != 'OAC':
            cv_col = f'{cell_type}_cv'
            if cv_col in variance_stats.columns:
                cvs = variance_stats[cv_col].dropna()
                if len(cvs) > 0:
                    print(f"  {cell_type}: CV range {cvs.min():.3f} - {cvs.max():.3f}, median {cvs.median():.3f}")
                    high_var_count = (cvs > 0.5).sum()
                    if high_var_count > 0:
                        print(f"    {high_var_count} regions with CV > 0.5")
    
    # Check for high variance regions that were filtered out
    if args.max_cv_threshold > 0:
        total_regions_before_variance_filter = len(filtered_df)
        regions_with_high_variance = total_regions_before_variance_filter - len(atlas_df)
        if regions_with_high_variance > 0:
            print(f"  Regions filtered due to high within-cell-type variance: {regions_with_high_variance}")
    
    # Save variance statistics for inspection
    variance_output = args.output_file.replace('.tsv', '_variance_stats.tsv')
    variance_stats.to_csv(variance_output, sep='\t', index=False)
    print(f"\nSaved variance statistics to {variance_output}")
    
    # Chromosome distribution
    if 'chr' in atlas_df.columns:
        chr_counts = atlas_df['chr'].value_counts().head(10)
        print(f"\nChromosome distribution:")
        for chr_name, count in chr_counts.items():
            print(f"  {chr_name}: {count} regions")

if __name__ == "__main__":
    main()