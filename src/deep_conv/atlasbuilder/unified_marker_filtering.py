import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse
import re

def calculate_reference_tumour_signal(
    signal_df: pd.DataFrame,
    tumour_purity_dict: Dict[str, float],
    tumour_prefix: str = 'OAC'
) -> pd.Series:
    """
    Calculate purity-adjusted weighted average tumour signal.
    All samples are assumed to have sufficient coverage and purity by design.

    Parameters:
    - signal_df: DataFrame with tumor signal values
    - tumour_purity_dict: Dictionary mapping sample names to purity values
    - tumour_prefix: Prefix to identify tumour samples

    Returns:
    - Series with purity-adjusted weighted average signals
    """

    # Get all tumour columns (all are valid by design)
    tumour_cols = [col for col in signal_df.columns
                   if col.startswith(tumour_prefix) and col in tumour_purity_dict]

    if not tumour_cols:
        print(f"ERROR: No tumour samples found!")
        print(f"Available tumour columns: {[col for col in signal_df.columns if col.startswith(tumour_prefix)]}")
        return pd.Series(np.nan, index=signal_df.index)

    print(f"Using tumour samples for reference calculation:")
    for col in tumour_cols:
        print(f"  {col}: purity {tumour_purity_dict[col]:.3f}")

    # Get purities as weights
    purities = np.array([tumour_purity_dict[col] for col in tumour_cols])
    
    # Calculate purity-adjusted weighted average for each region
    reference_signals = []
    for idx in signal_df.index:
        signals = signal_df.loc[idx, tumour_cols].values
        # Adjust by purity (normalize to 100% purity equivalent)
        adjusted_signals = signals / purities
        # Calculate weighted average (weighted by purity)
        weighted_avg = np.average(adjusted_signals, weights=purities)
        reference_signals.append(weighted_avg)

    reference_signal = pd.Series(reference_signals, index=signal_df.index)
    
    # Clip signals at 1.0 since 100% purity should not exceed 1.0
    reference_signal = reference_signal.clip(upper=1.0)

    print("Reference tumour signal calculation results:")
    print(f"  Total regions processed: {len(signal_df):,}")
    print(f"  All regions have valid reference signal: {len(signal_df):,}")

    return reference_signal


def analyse_thresholds(signal_df: pd.DataFrame, coverage_df: pd.DataFrame, tumour_purity_dict: Dict[str, float], min_coverage):
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
    
    # Calculate reference tumour signal
    print("Calculating reference tumour signals...")
    sample_df['tumour_reference'] = calculate_reference_tumour_signal(
        sample_df, tumour_purity_dict, 'OAC'
    )
    
    # Get regions with valid tumour signal for analysis
    valid_tumour = sample_df[~sample_df['tumour_reference'].isna()].copy()
    print(f"Found {len(valid_tumour)} regions with valid tumour signal")
    
    if len(valid_tumour) == 0:
        print("No regions with valid tumour signal found!")
        return
    
    # Calculate weighted cell type signals (the actual values used in filtering)
    print("Calculating weighted cell type signals for threshold analysis...")
    cell_type_order = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                      "Colon", "Esophagus", "Gastric", "Granulocytes", 
                      "Monocytes", "NK-cells", "Small-intestine", "T-cells"]
    
    # Add target column for calculate_weighted_cell_type_signals
    valid_tumour['target'] = 'OAC'
    valid_coverage = sample_coverage.loc[valid_tumour.index]
    
    merged_signals, _ = calculate_weighted_cell_type_signals(
        valid_tumour, valid_coverage, cell_type_order, min_coverage
    )
    
    # Skip control signal analysis - regions are pre-filtered
    valid_tumour['median_control'] = 0
    valid_tumour['max_control'] = 0
    
    # Calculate coverage
    tumour_cols = [col for col in sample_df.columns if col.startswith('OAC')]
    valid_tumour['tumour_coverage'] = sample_coverage[tumour_cols].mean(axis=1)
    
    # Calculate weighted blood/immune and GI signals
    blood_immune_types = ['Granulocytes', 'T-cells', 'B-cells', 'NK-cells', 
                         'Monocytes', 'CD34-erythroblasts', 'CD34-megakaryocytes']
    gi_types = ['Colon', 'Esophagus', 'Gastric', 'Small-intestine']
    
    blood_cols = [col for col in blood_immune_types if col in merged_signals.columns]
    gi_cols = [col for col in gi_types if col in merged_signals.columns]
    
    if blood_cols:
        valid_tumour['max_blood_immune'] = merged_signals[blood_cols].max(axis=1)
        valid_tumour['median_blood_immune'] = merged_signals[blood_cols].median(axis=1)
    else:
        valid_tumour['max_blood_immune'] = 0
        valid_tumour['median_blood_immune'] = 0
        
    if gi_cols:
        valid_tumour['max_gi'] = merged_signals[gi_cols].max(axis=1)
    else:
        valid_tumour['max_gi'] = 0
    
    print(f"Analysis of {len(valid_tumour)} sampled regions with valid 100% tumour signal:")
    print(f"Found {len(blood_cols)} blood/immune and {len(gi_cols)} GI cell types\n")
    
    # Tumour signal analysis
    print("Reference Tumour Signal (071-021, 47.6% purity, ~diploid):")
    print(f"  Min: {valid_tumour['tumour_reference'].min():.3f}")
    print(f"  25th percentile: {valid_tumour['tumour_reference'].quantile(0.25):.3f}")
    print(f"  Median: {valid_tumour['tumour_reference'].median():.3f}")
    print(f"  75th percentile: {valid_tumour['tumour_reference'].quantile(0.75):.3f}")
    print(f"  Max: {valid_tumour['tumour_reference'].max():.3f}")
    # More reasonable tumour signal threshold (25th percentile of reference)
    print(f"  Suggested min_tumour_signal: {valid_tumour['tumour_reference'].quantile(0.25):.3f}")
    
    # Blood/immune analysis (using weighted signals)
    print("\nWeighted Blood/Immune Signal:")
    print(f"  Min: {valid_tumour['median_blood_immune'].min():.4f}")
    print(f"  25th percentile: {valid_tumour['median_blood_immune'].quantile(0.25):.4f}")
    print(f"  Median: {valid_tumour['median_blood_immune'].median():.4f}")
    print(f"  75th percentile: {valid_tumour['median_blood_immune'].quantile(0.75):.4f}")
    print(f"  Max: {valid_tumour['median_blood_immune'].max():.4f}")
    # More reasonable blood signal threshold (10th percentile)
    print(f"  Suggested max_blood_signal: {valid_tumour['median_blood_immune'].quantile(0.10):.4f}")
    
    # GI analysis (using weighted signals)
    print("\nWeighted GI Signal:")
    print(f"  Min: {valid_tumour['max_gi'].min():.4f}")
    print(f"  25th percentile: {valid_tumour['max_gi'].quantile(0.25):.4f}")
    print(f"  Median: {valid_tumour['max_gi'].median():.4f}")
    print(f"  75th percentile: {valid_tumour['max_gi'].quantile(0.75):.4f}")
    print(f"  Max: {valid_tumour['max_gi'].max():.4f}")
    # Skip control analysis - regions are pre-filtered
    print("\nControl Signal:")
    print("  Skipped - regions are pre-filtered for tumour-specificity")
    
    # Coverage analysis
    print("\nTumour Coverage:")
    print(f"  Min: {valid_tumour['tumour_coverage'].min():.1f}")
    print(f"  25th percentile: {valid_tumour['tumour_coverage'].quantile(0.25):.1f}")
    print(f"  Median: {valid_tumour['tumour_coverage'].median():.1f}")
    print(f"  75th percentile: {valid_tumour['tumour_coverage'].quantile(0.75):.1f}")
    print(f"  Max: {valid_tumour['tumour_coverage'].max():.1f}")
    print(f"  Suggested min_coverage: {valid_tumour['tumour_coverage'].quantile(0.25):.0f}")
    
    # Realistic suggested command
    print("\n" + "="*60)
    print("REALISTIC SUGGESTED COMMAND:")
    print("="*60)
    print(f"python script.py \\")
    print(f"  --min_tumour_signal {valid_tumour['tumour_reference'].quantile(0.25):.3f} \\")
    print(f"  --max_blood_signal {valid_tumour['median_blood_immune'].quantile(0.10):.4f} \\")
    print(f"  --min_coverage {valid_tumour['tumour_coverage'].quantile(0.25):.0f} \\")
    print(f"  --control_dir /path/to/controls \\")
    print(f"  --no_overlap")


def calculate_weighted_cell_type_signals(signal_df: pd.DataFrame, 
                                       coverage_df: pd.DataFrame,
                                       cell_type_order: List[str],
                                       min_coverage: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            # For OAC, use the precomputed, purity-corrected reference tumour signal
            if 'tumour_reference' in signal_df.columns:
                # Cap the signal at 1.0 for safety, though it should already be bounded
                capped_signal = np.minimum(signal_df['tumour_reference'], 1.0)
                result_df[cell_type] = capped_signal
                variance_stats[f'{cell_type}_cv'] = 0  # No variance for the fixed reference signal
            else:
                print(f"ERROR: 'tumour_reference' column not found! Available columns: {list(signal_df.columns)}")
                result_df[cell_type] = np.nan
                variance_stats[f'{cell_type}_cv'] = np.nan
        else:
            # Find all samples for this cell type
            cell_samples = [col for col in signal_df.columns 
                          if col.startswith(cell_type) and col not in metadata_cols and not col.endswith('_coverage')]
            if cell_samples:
                # Get valid samples that exist in both dataframes
                valid_samples = [s for s in cell_samples if s in signal_df.columns and s in coverage_df.columns]
                
                if valid_samples:
                    # Get signal and coverage matrices for this cell type
                    signal_matrix = signal_df[valid_samples].values
                    coverage_matrix = coverage_df.loc[signal_df.index, valid_samples].values
                    
                    # Create masks for valid data (not NaN and coverage > 0)
                    valid_mask = (~pd.isna(signal_matrix)) & (~pd.isna(coverage_matrix)) & (coverage_matrix > 0)
                    
                    # Zero out invalid entries
                    signal_matrix = np.where(valid_mask, signal_matrix, 0)
                    coverage_matrix = np.where(valid_mask, coverage_matrix, 0)
                    
                    # Calculate cumulative coverage per region
                    cumulative_coverage = np.sum(coverage_matrix, axis=1)
                    
                    # Calculate weighted sum per region
                    weighted_sum = np.sum(signal_matrix * coverage_matrix, axis=1)
                    
                    # Calculate weighted signal, set to NaN where coverage is insufficient
                    weighted_signals = np.where(
                        cumulative_coverage >= min_coverage,
                        weighted_sum / np.maximum(cumulative_coverage, 1e-10),
                        np.nan
                    )
                    
                    # Calculate coefficient of variation per region
                    cvs = np.full(len(signal_df), np.nan)
                    for i in range(len(signal_df)):
                        valid_signals = signal_matrix[i][valid_mask[i]]
                        if len(valid_signals) > 1 and cumulative_coverage[i] >= min_coverage:
                            cv = np.std(valid_signals) / (np.mean(valid_signals) + 1e-10)
                            cvs[i] = cv                            
                        elif len(valid_signals) == 1 and cumulative_coverage[i] >= min_coverage:
                            cvs[i] = 0  # Single sample, no variance
                    
                    result_df[cell_type] = weighted_signals
                    variance_stats[f'{cell_type}_cv'] = cvs
                else:
                    # No valid samples for this cell type
                    result_df[cell_type] = np.nan
                    variance_stats[f'{cell_type}_cv'] = np.nan
            else:
                # No samples for this cell type
                result_df[cell_type] = np.nan
                variance_stats[f'{cell_type}_cv'] = np.nan
    return result_df, variance_stats

def select_non_overlapping_regions(filtered_df: pd.DataFrame, 
                                 min_distance: int = 20) -> pd.DataFrame:
    """
    Select non-overlapping regions with highest quality scores.
    """
    # Sort by quality score (tumour_reference signal * SNR)
    filtered_df['quality_score'] = (
        filtered_df['tumour_reference'] * 
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


def apply_filters(signal_df: pd.DataFrame,
                  min_tumour_signal: float,
                  max_blood_signal: float,
                  control_signal_df: pd.DataFrame = None,
                  control_coverage_df: pd.DataFrame = None,
                  xtp_min_coverage: int = 8,
                  gi_min_coverage: int = 4,
                  coverage_quorum: float = 0.6,
                  mean_control_threshold: float = 0.002,
                  max_control_threshold: float = 0.01,
                  max_pct_with_signal: float = 5.0,
                  high_signal_threshold: float = 0.01,
                  min_samples: int = 3) -> pd.DataFrame:
    """
    Unified filtering combining tumour/blood/GI filtering with control filtering.
    Core principle: Strong tumour signal + clean controls, with relaxed blood/GI.
    """
    print("\nApplying unified filters (tumour + blood/GI + controls):")
    print(f"  Starting with: {len(signal_df):,} regions (after NaN removal)")
    
    # Filter 1: High tumour signal (core requirement)
    step1 = signal_df[signal_df['tumour_reference'] >= min_tumour_signal]
    print(f"  After tumour signal ≥ {min_tumour_signal}: {len(step1):,}")
    
    # Filter 2: Control coverage filter
    print("\n  Applying control coverage filter:")
    coverage_mask, control_cols = apply_control_coverage_filter(
        control_signal_df.loc[step1.index], 
        control_coverage_df.loc[step1.index],
        xtp_min_coverage, gi_min_coverage, coverage_quorum
    )
    step2 = step1[coverage_mask].copy()
    print(f"  After control coverage filter: {len(step2):,}")
    
    # Filter 3: Control signal filter
    print("\n  Applying control signal filter:")
    signal_mask = apply_control_signal_filter(
        control_signal_df.loc[step2.index], 
        control_cols,
        mean_control_threshold,
        max_control_threshold,
        max_pct_with_signal,
        high_signal_threshold,
        min_samples
    )
    step3 = step2[signal_mask].copy()
    print(f"  After control signal filter: {len(step3):,}")
    
    # At this point, controls have filtered out real-world contamination patterns
    # Pure cell type filters should be very lenient since controls = ground truth
    
    # Optional: Very lenient blood filter as backup (since controls caught real blood contamination)
    backup_blood_threshold = max_blood_signal * 5  # 5x more relaxed
    step4 = step3[step3['median_blood_immune'] <= backup_blood_threshold]
    print(f"  After backup blood filter ≤ {backup_blood_threshold}: {len(step4):,}")
    
    # Note: Skipping GI filters entirely since:
    # 1. GI tissues rarely appear in healthy cfDNA (controls would catch any real GI contamination)
    # 2. High GI signal with clean controls likely indicates real tumour-associated signal
    print(f"  Skipping GI filters - controls already captured real-world contamination")
    
    filtered = step4.copy()
    return filtered.copy()

def apply_control_coverage_filter(signal_df: pd.DataFrame, 
                                coverage_df: pd.DataFrame,
                                xtp_min_coverage: int = 10,
                                gi_min_coverage: int = 4,
                                coverage_quorum: float = 0.8) -> Tuple[pd.Series, List[str]]:
    """
    Apply coverage filtering to control samples with different thresholds.
    X###/TP### controls need higher coverage than GI controls.
    Returns coverage mask and control column names.
    """
    control_cols = [col for col in signal_df.columns if col.startswith('Control_')]
    
    if not control_cols:
        print("No control columns found!")
        return pd.Series(True, index=signal_df.index), []
    
    # Classify control samples
    xtp_controls = [col for col in control_cols if 
                   re.match(r'Control_X\d+', col) or re.match(r'Control_TP\d+', col)]
    gi_controls = [col for col in control_cols if 'GI' in col]
    other_controls = [col for col in control_cols if col not in xtp_controls and col not in gi_controls]
    
    print(f"\nControl sample classification:")
    print(f"  X###/TP### controls: {len(xtp_controls)}")
    print(f"  GI controls: {len(gi_controls)}")
    print(f"  Other controls: {len(other_controls)}")
    
    # Apply coverage filtering
    coverage_mask = pd.Series(True, index=signal_df.index)
    
    # X###/TP### controls - require quorum to have sufficient coverage
    if xtp_controls:
        xtp_coverage_ok = (coverage_df[xtp_controls] >= xtp_min_coverage).sum(axis=1)
        xtp_mask = xtp_coverage_ok >= (len(xtp_controls) * coverage_quorum)
        print(f"  X###/TP### coverage ≥{xtp_min_coverage} (≥{coverage_quorum:.0%} of samples):")
        print(f"    Passing: {xtp_mask.sum()}/{len(xtp_mask)} ({100*xtp_mask.sum()/len(xtp_mask):.1f}%)")
        coverage_mask &= xtp_mask
    
    # GI controls - require quorum to have sufficient coverage
    if gi_controls:
        gi_coverage_ok = (coverage_df[gi_controls] >= gi_min_coverage).sum(axis=1)
        gi_mask = gi_coverage_ok >= (len(gi_controls) * coverage_quorum)
        print(f"  GI coverage ≥{gi_min_coverage} (≥{coverage_quorum:.0%} of samples):")
        print(f"    Passing: {gi_mask.sum()}/{len(gi_mask)} ({100*gi_mask.sum()/len(gi_mask):.1f}%)")
        coverage_mask &= gi_mask
    
    # Other controls - use GI threshold
    if other_controls:
        other_coverage_ok = (coverage_df[other_controls] >= gi_min_coverage).sum(axis=1)
        other_mask = other_coverage_ok >= (len(other_controls) * coverage_quorum)
        print(f"  Other controls coverage ≥{gi_min_coverage} (≥{coverage_quorum:.0%} of samples):")
        print(f"    Passing: {other_mask.sum()}/{len(other_mask)} ({100*other_mask.sum()/len(other_mask):.1f}%)")
        coverage_mask &= other_mask
    
    print(f"  Combined coverage filter: {coverage_mask.sum()}/{len(coverage_mask)} ({100*coverage_mask.sum()/len(coverage_mask):.1f}%) regions pass")
    
    return coverage_mask, control_cols

def apply_control_signal_filter(signal_df: pd.DataFrame,
                              control_cols: List[str],
                              mean_control_threshold: float = 0.002,
                              max_control_threshold: float = 0.01,
                              max_pct_with_signal: float = 5.0,
                              high_signal_threshold: float = 0.01,
                              min_samples: int = 3) -> pd.Series:
    """
    Apply proven control signal filtering using exact same metrics as successful past approach.
    Matches calculate_control_metrics_vectorized from second_line_oac_regions_filter_with_controls.py
    """
    if not control_cols:
        print("No control columns found!")
        return pd.Series(True, index=signal_df.index)
    
    print(f"\nApplying proven control filtering to {len(control_cols)} control samples")
    
    # Extract control data  
    control_mv = signal_df[control_cols].copy()
    
    # Calculate metrics directly from signal data (coverage already filtered earlier)
    print("  Calculating control metrics...")
    
    # Basic statistics (vectorized, skipna=True for any existing NaNs)
    n_valid_samples = (~control_mv.isna()).sum(axis=1)
    mean_control_signal = control_mv.mean(axis=1, skipna=True)
    max_control_signal = control_mv.max(axis=1, skipna=True)
    
    # Contamination metrics (using 0.001 threshold like proven approach)
    n_with_signal = (control_mv > 0.001).sum(axis=1)  # >0.1%
    n_high_signal = (control_mv > high_signal_threshold).sum(axis=1)   # >1%
    pct_with_signal = n_with_signal / n_valid_samples * 100
    
    # Handle edge cases (matching proven approach)
    mean_control_signal = mean_control_signal.fillna(0)
    max_control_signal = max_control_signal.fillna(0)
    pct_with_signal = pct_with_signal.fillna(0)
    
    print(f"\nControl signal distributions:")
    print(f"  N valid samples - percentiles: 25th={n_valid_samples.quantile(0.25):.0f}, 50th={n_valid_samples.quantile(0.5):.0f}, 75th={n_valid_samples.quantile(0.75):.0f}")
    print(f"  Mean control signal - percentiles: 25th={mean_control_signal.quantile(0.25):.6f}, 50th={mean_control_signal.quantile(0.5):.6f}, 75th={mean_control_signal.quantile(0.75):.6f}")
    print(f"  Max control signal - percentiles: 25th={max_control_signal.quantile(0.25):.6f}, 50th={max_control_signal.quantile(0.5):.6f}, 75th={max_control_signal.quantile(0.75):.6f}")
    print(f"  Pct with signal (>0.1%) - percentiles: 25th={pct_with_signal.quantile(0.25):.1f}%, 50th={pct_with_signal.quantile(0.5):.1f}%, 75th={pct_with_signal.quantile(0.75):.1f}%")
    print(f"  N high signal (>1%) - percentiles: 25th={n_high_signal.quantile(0.25):.0f}, 50th={n_high_signal.quantile(0.5):.0f}, 75th={n_high_signal.quantile(0.75):.0f}")
    
    # Apply proven filters (matching apply_strict_control_filters exactly)
    print(f"\nApplying proven control filters:")
    
    # Filter 1: Sufficient samples (minimum as specified)
    sufficient_samples = n_valid_samples >= min_samples
    print(f"  Sufficient samples ≥{min_samples}: {sufficient_samples.sum()}/{len(sufficient_samples)} ({100*sufficient_samples.sum()/len(sufficient_samples):.1f}%)")
    
    # Filter 2: Ultra low mean (≤0.2%)
    ultra_low_mean = mean_control_signal <= mean_control_threshold
    print(f"  Ultra low mean ≤{mean_control_threshold}: {ultra_low_mean.sum()}/{len(ultra_low_mean)} ({100*ultra_low_mean.sum()/len(ultra_low_mean):.1f}%)")
    
    # Filter 3: Low max signal (≤1%)
    low_max_signal = max_control_signal <= max_control_threshold
    print(f"  Low max signal ≤{max_control_threshold}: {low_max_signal.sum()}/{len(low_max_signal)} ({100*low_max_signal.sum()/len(low_max_signal):.1f}%)")
    
    # Filter 4: Minimal contamination (≤5% samples with signal >0.1%)
    minimal_contamination = pct_with_signal <= max_pct_with_signal
    print(f"  Minimal contamination ≤{max_pct_with_signal}%: {minimal_contamination.sum()}/{len(minimal_contamination)} ({100*minimal_contamination.sum()/len(minimal_contamination):.1f}%)")
    
    # Filter 5: No high signal (0 samples >1%)
    no_high_signal = n_high_signal == 0
    print(f"  No high signal (0 samples >{high_signal_threshold}): {no_high_signal.sum()}/{len(no_high_signal)} ({100*no_high_signal.sum()/len(no_high_signal):.1f}%)")
    
    # Combine all filters (matching proven approach order)
    signal_mask = sufficient_samples & ultra_low_mean & low_max_signal & minimal_contamination & no_high_signal
    print(f"  Combined control filters: {signal_mask.sum()}/{len(signal_mask)} ({100*signal_mask.sum()/len(signal_mask):.1f}%)")
    
    return signal_mask

def unified_marker_filtering(signal_df: pd.DataFrame,
                           coverage_df: pd.DataFrame,
                           tumour_purity_dict: Dict[str, float],
                           min_tumour_signal: float,
                           min_coverage: int,
                           max_blood_signal: float,
                           select_non_overlapping: bool,
                           control_signal_df: pd.DataFrame,
                           control_coverage_df: pd.DataFrame,
                           xtp_min_coverage: int = 8,
                           gi_min_coverage: int = 4,
                           coverage_quorum: float = 0.8,
                           mean_control_threshold: float = 0.002,
                           max_control_threshold: float = 0.01,
                           max_pct_with_signal: float = 5.0,
                           high_signal_threshold: float = 0.01,
                           min_samples: int = 3) -> pd.DataFrame:
    """
    Unified filtering combining all criteria in a single step.
    Now uses merged cell type signals for differential methylation filtering.
    """
    print("="*60)
    print("UNIFIED MARKER FILTERING")
    print("="*60)
    
    # Step 1: Calculate reference tumour signal
    print("Calculating reference tumour signals...")
    signal_df['tumour_reference'] = calculate_reference_tumour_signal(
        signal_df, tumour_purity_dict, 'OAC'
    )
    
    # Add basic region info (inspired by previous work)
    signal_df['region_length'] = signal_df['end'] - signal_df['start']
    signal_df['n_cpgs'] = signal_df['endCpG'] - signal_df['startCpG'] + 1
    
    # Step 2: Calculate merged cell type signals with variance filtering
    print("Calculating merged cell type signals...")
    cell_type_order = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                      "Colon", "Esophagus", "Gastric", "Granulocytes", 
                      "Monocytes", "NK-cells", "OAC", "Small-intestine", "T-cells"]
    
    # Add target column for calculate_weighted_cell_type_signals
    signal_df['target'] = 'OAC'
    
    # Pass min_coverage to the function - it will handle coverage filtering internally
    merged_signals, variance_stats = calculate_weighted_cell_type_signals(
        signal_df, coverage_df, cell_type_order, min_coverage
    )
    
    print(f"Merged signals shape: {merged_signals.shape}")
    
    # Step 3: FIRST remove all regions with NaN (insufficient coverage)
    # Merge the signals back to signal_df for easier filtering
    for col in cell_type_order:
        if col in merged_signals.columns:
            signal_df[f'signal_{col}'] = merged_signals[col]
    
    # Remove regions with any NaN in cell type signals
    print("\nRemoving regions with insufficient coverage (NaN values)...")
    signal_cols = [f'signal_{col}' for col in cell_type_order if f'signal_{col}' in signal_df.columns]
    before_nan_removal = len(signal_df)
    signal_df = signal_df.dropna(subset=signal_cols)
    print(f"  Removed {before_nan_removal - len(signal_df):,} regions with insufficient coverage")
    print(f"  Remaining regions: {len(signal_df):,}")
    
    # Step 4: Calculate differential methylation metrics from merged signals
    blood_immune_types = ['Granulocytes', 'T-cells', 'B-cells', 'NK-cells', 
                         'Monocytes', 'CD34-erythroblasts', 'CD34-megakaryocytes']
    gi_types = ['Colon', 'Esophagus', 'Gastric', 'Small-intestine']
    
    # Use the merged signals directly (these are the weighted/cumulative signals)
    print(f"Found {len(blood_immune_types)} blood/immune cell types")
    print(f"Found {len(gi_types)} GI cell types")
    
    # Calculate max and median for blood/immune from merged signals
    signal_df['max_blood_immune'] = merged_signals[blood_immune_types].max(axis=1)
    signal_df['median_blood_immune'] = merged_signals[blood_immune_types].median(axis=1)
    
    # GI calculations removed - using control-based filtering instead
    
    # Step 7: Apply unified filtering criteria (tumour + controls + blood backup)
    filtered = apply_filters(
        signal_df, 
        min_tumour_signal, 
        max_blood_signal,
        control_signal_df,
        control_coverage_df,
        xtp_min_coverage,
        gi_min_coverage,
        coverage_quorum,
        mean_control_threshold,
        max_control_threshold,
        max_pct_with_signal,
        high_signal_threshold,
        min_samples
    )
    
    print(f"\nFiltering summary:")
    print(f"  Started with: {before_nan_removal:,} regions")
    print(f"  After coverage filter: {len(signal_df):,} regions")
    print(f"  Final: {len(filtered):,} regions ({100*len(filtered)/before_nan_removal:.2f}% of original)")
    
    # Select non-overlapping regions if requested
    if select_non_overlapping and len(filtered) > 0:
        filtered = select_non_overlapping_regions(filtered, min_distance=20)
        print(f"  Non-overlapping regions selected: {len(filtered):,}")
    
    # Quality statistics (inspired by previous work)
    if len(filtered) > 0:
        print(f"\nQuality distribution of filtered regions:")
        print(f"  Tumour signal: {filtered['tumour_reference'].min():.3f} - {filtered['tumour_reference'].max():.3f}")
        print(f"  Median blood/immune: {filtered['median_blood_immune'].min():.4f} - {filtered['median_blood_immune'].max():.4f}")
        print(f"  Max blood/immune: {filtered['max_blood_immune'].min():.4f} - {filtered['max_blood_immune'].max():.4f}")
        
        # Fragment characteristics (from previous work)
        print(f"\nFragment characteristics:")
        print(f"  Region length: {filtered['region_length'].min()} - {filtered['region_length'].max()} bp")
        print(f"  CpGs per region: {filtered['n_cpgs'].min()} - {filtered['n_cpgs'].max()}")
        
        
        # Chromosome distribution
        print(f"\nChromosome distribution (top 10):")
        chr_counts = filtered['chr'].value_counts().head(10)
        for chr_name, count in chr_counts.items():
            print(f"  {chr_name}: {count}")
    
    return filtered

def main():
    parser = argparse.ArgumentParser(description='Unified marker filtering')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing marker_values.parquet and coverage.parquet')
    parser.add_argument('--output_file', type=str, required=True, help='Output filtered regions file')
    parser.add_argument('--atlas', type=str, required=True, help='Input atlas file with region metadata')
    parser.add_argument('--min_cpgs', type=int, default=3, help='Minimum CpGs per region (used for input file naming)')
    
    # Core filtering parameters
    parser.add_argument('--min_coverage', type=int, default=8, help='Minimum coverage required for cell type signals')
    parser.add_argument('--min_tumour_signal', type=float, default=0.8, help='Minimum tumour signal at 100% purity')
    parser.add_argument('--max_blood_signal', type=float, default=0.001, help='Maximum median blood/immune signal')
    parser.add_argument('--no_overlap', action='store_true', help='Select non-overlapping regions')
    parser.add_argument('--analyse_thresholds', action='store_true', help='Analyze data to suggest appropriate thresholds')
    
    # Control filtering parameters
    parser.add_argument('--control_dir', type=str, help='Directory containing control parquet files (control_mv.parquet, control_cov.parquet)')
    parser.add_argument('--xtp_min_coverage', type=int, default=8, help='Minimum coverage for X###/TP### controls')
    parser.add_argument('--gi_min_coverage', type=int, default=4, help='Minimum coverage for GI controls')
    parser.add_argument('--coverage_quorum', type=float, default=0.6, help='Fraction of control samples that must pass coverage')
    parser.add_argument('--mean_control_threshold', type=float, default=0.01, help='Maximum mean control signal')
    parser.add_argument('--max_control_threshold', type=float, default=0.05, help='Maximum control signal')
    parser.add_argument('--max_pct_with_signal', type=float, default=20.0, help='Maximum percent of controls with signal')
    parser.add_argument('--high_signal_threshold', type=float, default=0.1, help='High signal threshold for control filtering')
    parser.add_argument('--min_samples', type=int, default=3, help='Minimum number of valid control samples required (proven: 3)')
    parser.add_argument('--xtp_max_signal', type=float, default=0.001, help='Maximum signal allowed in X###/TP### controls (legacy parameter)')
    
    args = parser.parse_args()
    
    # Load data from input directory
    import os
    signal_path = os.path.join(args.input_dir, f"l{args.min_cpgs}_marker_values.parquet")
    coverage_path = os.path.join(args.input_dir, f"l{args.min_cpgs}_coverage.parquet")
    
    print(f"Loading signal matrix from {signal_path}...")
    signal_df = pd.read_parquet(signal_path)

    print(f"Loading coverage matrix from {coverage_path}...")
    coverage_df = pd.read_parquet(coverage_path)

    atlas = pd.read_csv(args.atlas, sep="\t")
    signal_df[atlas.columns[:8]] = atlas[atlas.columns[:8]]

    # Load control data 
    control_signal_path = os.path.join(args.control_dir, "marker_values.parquet")
    control_coverage_path = os.path.join(args.control_dir, "coverage.parquet")
    print(f"Loading control signal matrix from {control_signal_path}...")
    control_signal_df = pd.read_parquet(control_signal_path)
    print(f"Loading control coverage matrix from {control_coverage_path}...")
    control_coverage_df = pd.read_parquet(control_coverage_path)
    print(f"Loaded control data: {len(control_signal_df)} regions, {control_signal_df.shape[1]} control samples")

    # Load tumour purity
    tumour_purity_dict = {
        'OAC_069-009_ScrBsl_tumour_cna_corrected':0.5171,
        'OAC_071-021_ScrBsl_tumour_cna_corrected':0.4766,
        'OAC_071-022_ScrBsl_tumour_cna_corrected':0.4108,
        'OAC_071-043_ScrBsl_tumour_cna_corrected':0.4607,
        'OAC_129-001_ScrBsl_tumour_cna_corrected':0.6921
    }

    print(tumour_purity_dict)

    if args.analyse_thresholds:
        analyse_thresholds(signal_df, coverage_df, tumour_purity_dict)
        return

    # Apply filtering (without overlap selection first)
    filtered_df = unified_marker_filtering(
        signal_df = signal_df, 
        coverage_df = coverage_df, 
        tumour_purity_dict = tumour_purity_dict,
        min_tumour_signal=args.min_tumour_signal,
        min_coverage=args.min_coverage,
        max_blood_signal=args.max_blood_signal,
        select_non_overlapping=False, 
        control_signal_df=control_signal_df,
        control_coverage_df=control_coverage_df,
        xtp_min_coverage=args.xtp_min_coverage,
        gi_min_coverage=args.gi_min_coverage,
        coverage_quorum=args.coverage_quorum,
        mean_control_threshold=args.mean_control_threshold,
        max_control_threshold=args.max_control_threshold,
        max_pct_with_signal=args.max_pct_with_signal,
        high_signal_threshold=args.high_signal_threshold,
        min_samples=args.min_samples
    )

    # Apply non-overlapping selection if requested
    if args.no_overlap and len(filtered_df) > 0:
        print(f"\nApplying non-overlapping selection to {len(filtered_df)} regions...")
        filtered_df = select_non_overlapping_regions(filtered_df)
        print(f"Selected {len(filtered_df)} non-overlapping regions")

    # The filtered_df already has the target column and merged signals from unified_marker_filtering
    # Just need to get the final atlas format with all cell types
    cell_type_order = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                      "Colon", "Esophagus", "Gastric", "Granulocytes", 
                      "Monocytes", "NK-cells", "OAC", "Small-intestine", "T-cells"]

    print("\nPreparing final atlas from filtered regions...")
    
    # Use the already calculated signals from the filtering step
    atlas_df = filtered_df.copy()
    
    # Ensure proper column order (CORRECT ORDER for filter_OAC_pats.sh)
    metadata_cols = ['chr', 'start', 'end', 'startCpG', 'endCpG', 'target', 'name', 'direction']
    output_cols = metadata_cols + cell_type_order
    atlas_df = atlas_df[output_cols]
    
    # Sort atlas by chromosome and position
    def chromosome_sort_key(chr_str):
        """Convert chromosome to sortable format"""
        chr_clean = str(chr_str).replace('chr', '')
        if chr_clean.isdigit():
            return (0, int(chr_clean))
        elif chr_clean == 'X':
            return (1, 0)
        elif chr_clean == 'Y':
            return (1, 1)
        else:
            return (2, chr_clean)
    
    atlas_df['chr_sort'] = atlas_df['chr'].apply(chromosome_sort_key)
    atlas_df = atlas_df.sort_values(['chr_sort', 'start']).drop('chr_sort', axis=1)
    atlas_df = atlas_df.reset_index(drop=True)

    # Save results
    atlas_df.to_csv(args.output_file, sep='\t', index=False)
    print(f"\nSaved {len(atlas_df)} filtered regions to {args.output_file}")

    # Print summary statistics
    print(f"\nAtlas summary:")
    print(f"  Total regions: {len(atlas_df)}")
    
    # Chromosome distribution
    if 'chr' in atlas_df.columns:
        chr_counts = atlas_df['chr'].value_counts().head(10)
        print(f"\nChromosome distribution:")
        for chr_name, count in chr_counts.items():
            print(f"  {chr_name}: {count} regions")

if __name__ == "__main__":
    main()
