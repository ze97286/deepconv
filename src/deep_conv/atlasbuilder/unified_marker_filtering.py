import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse

def calculate_reference_tumour_signal(signal_df: pd.DataFrame, 
                                      tumour_purity_dict: Dict[str, float],
                                      tumour_prefix: str = 'OAC') -> pd.Series:
    """
    Use signal from the reference tumor sample (071-021 at 47.6% purity, ~diploid).
    This avoids artifacts from copy number losses and provides a realistic
    detection reference at moderate purity.
    """
    # Reference sample: 071-021_ScrBsl_tumour (47.6% purity, 1.968 ploidy)
    reference_sample = '071-021_ScrBsl_tumour'
    
    # Find the reference sample column
    reference_col = None
    for col in signal_df.columns:
        if col.startswith(tumour_prefix) and reference_sample in col:
            reference_col = col
            break
    
    if reference_col is None:
        print(f"ERROR: Reference sample {reference_sample} not found!")
        print(f"Available tumor columns: {[col for col in signal_df.columns if col.startswith(tumour_prefix)]}")
        return pd.Series(np.nan, index=signal_df.index)
    
    print(f"Using reference tumor sample: {reference_col}")
    print(f"Reference purity: 47.6%, Reference ploidy: ~1.968 (near-diploid)")
    
    reference_signals = signal_df[reference_col].copy()
    valid_count = (~reference_signals.isna()).sum()
    
    print(f"Reference tumor signal calculation results:")
    print(f"  Total regions processed: {len(signal_df):,}")
    print(f"  Regions with valid reference signal: {valid_count:,} ({100*valid_count/len(signal_df):.1f}%)")
    print(f"  Regions with missing signal: {len(signal_df) - valid_count:,} ({100*(len(signal_df) - valid_count)/len(signal_df):.1f}%)")
    
    return reference_signals

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
        valid_tumour, valid_coverage, cell_type_order
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
    # More reasonable GI signal threshold (50th percentile)
    print(f"  Suggested max_gi_signal: {valid_tumour['max_gi'].quantile(0.50):.4f}")
    
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
    print(f"  --max_gi_signal {valid_tumour['max_gi'].quantile(0.50):.4f} \\")
    print(f"  --min_coverage {valid_tumour['tumour_coverage'].quantile(0.25):.0f} \\")
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
            # For OAC, use the reference tumour signal extrapolated to 100% purity (capped at 1.0)
            if 'tumour_reference' in signal_df.columns:
                reference_purity = 0.476  # 47.6% purity for 071-021 sample
                extrapolated_signal = signal_df['tumour_reference'] / reference_purity
                result_df[cell_type] = np.minimum(extrapolated_signal, 1.0)  # Cap at 1.0
                variance_stats[f'{cell_type}_cv'] = 0  # No variance for single calculated value
            else:
                print(f"ERROR: 'tumour_reference' column not found! Available columns: {list(signal_df.columns)}")
                result_df[cell_type] = np.nan
                variance_stats[f'{cell_type}_cv'] = np.nan
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

def unified_marker_filtering(signal_df: pd.DataFrame,
                           coverage_df: pd.DataFrame,
                           tumour_purity_dict: Dict[str, float],
                           min_tumour_signal: float = 0.6,
                           min_coverage: int = 10,
                           max_blood_signal: float = 0.001,
                           max_gi_signal: float = 0.2,
                           max_gi_to_tumour_ratio: float = 0.25,
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
    
    # Step 1: Calculate reference tumour signal
    print("Calculating reference tumour signals...")
    signal_df['tumour_reference'] = calculate_reference_tumour_signal(
        signal_df, tumour_purity_dict, 'OAC'
    )
    
    # Step 2: Calculate merged cell type signals with variance filtering (excluding controls)
    print("Calculating merged cell type signals...")
    cell_type_order = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                      "Colon", "Esophagus", "Gastric", "Granulocytes", 
                      "Monocytes", "NK-cells", "OAC", "Small-intestine", "T-cells"]  # Added OAC to the list
    
    # Add target column for calculate_weighted_cell_type_signals
    signal_df['target'] = 'OAC'
    
    # IMPORTANT: signal_df now has tumour_reference column which will be used for OAC
    merged_signals, variance_stats = calculate_weighted_cell_type_signals(
        signal_df, coverage_df, cell_type_order, max_cv_threshold
    )
    
    print(f"Merged signals shape: {merged_signals.shape}")
    print(f"Merged signals columns: {list(merged_signals.columns)}")
    
    # Debug: check if OAC column exists and has the extrapolated tumour values
    if 'OAC' in merged_signals.columns:
        print(f"\n  DEBUG - OAC in merged signals (extrapolated to 100% purity):")
        for i in range(min(5, len(signal_df))):
            idx = signal_df.index[i]
            if idx in merged_signals.index:
                raw_ref = signal_df.loc[idx, 'tumour_reference']
                extrapolated = merged_signals.loc[idx, 'OAC']
                print(f"    Region {idx}: raw_reference={raw_ref:.6f}, extrapolated_OAC={extrapolated:.6f} (ratio: {extrapolated/raw_ref:.2f})")
            else:
                print(f"    Region {idx}: not found in merged_signals!")
    
    # Step 3: Calculate coverage requirements
    tumour_cols = [col for col in signal_df.columns if col.startswith('OAC')]
    signal_df['tumour_coverage'] = coverage_df[tumour_cols].mean(axis=1)
    
    # Step 4: Skip control signal check - regions are pre-filtered for tumour-specificity
    control_cols = []  # Force empty to skip control filtering
    print("  Skipping control filtering - regions were pre-filtered in tumour-purity correlation step.")
    signal_df['median_control'] = 0
    signal_df['max_control'] = 0
    
    # Step 5: Calculate differential methylation metrics using merged signals
    # Group blood/immune cell types
    blood_immune_types = ['Granulocytes', 'T-cells', 'B-cells', 'NK-cells', 
                         'Monocytes', 'CD34-erythroblasts', 'CD34-megakaryocytes']
    gi_types = ['Colon', 'Esophagus', 'Gastric', 'Small-intestine']
    
    # Get the actual sample columns from coverage_df for each cell type
    # We need to check coverage for ALL samples, not just the merged cell type columns
    blood_sample_cols = []
    gi_sample_cols = []
    
    for col in coverage_df.columns:
        if any(col.startswith(cell_type) for cell_type in blood_immune_types):
            blood_sample_cols.append(col)
        elif any(col.startswith(cell_type) for cell_type in gi_types):
            gi_sample_cols.append(col)
    
    print(f"\n  DEBUG - Found {len(blood_sample_cols)} blood/immune sample columns for coverage check")
    print(f"  DEBUG - Found {len(gi_sample_cols)} GI sample columns for coverage check")
    
    # Calculate coverage for all relevant samples
    signal_df['min_blood_coverage'] = coverage_df[blood_sample_cols].min(axis=1) if blood_sample_cols else 0
    signal_df['min_gi_coverage'] = coverage_df[gi_sample_cols].min(axis=1) if gi_sample_cols else 0
    signal_df['min_all_coverage'] = signal_df[['tumour_coverage', 'min_blood_coverage', 'min_gi_coverage']].min(axis=1)
    
    # Calculate max signals from merged cell type signals
    blood_cols = [col for col in blood_immune_types if col in merged_signals.columns]
    gi_cols = [col for col in gi_types if col in merged_signals.columns]
    
    # Debug: check if we found the right columns
    print(f"\n  DEBUG - Found blood columns in merged signals: {blood_cols}")
    print(f"  DEBUG - Found GI columns in merged signals: {gi_cols}")
    
    # Debug: check if merged_signals has the right indices
    print(f"  DEBUG - signal_df indices match merged_signals: {signal_df.index.equals(merged_signals.index)}")
    
    if blood_cols:
        # Debug: check merged blood signals
        print(f"\n  DEBUG - Merged blood signals (first 5 regions):")
        for i in range(min(5, len(merged_signals))):
            blood_vals = merged_signals[blood_cols].iloc[i].values
            print(f"    Region {i}: min={np.nanmin(blood_vals):.6f}, max={np.nanmax(blood_vals):.6f}, median={np.nanmedian(blood_vals):.6f}")
        
        # Fix: ensure we're aligning the indices correctly
        if not signal_df.index.equals(merged_signals.index):
            print(f"  WARNING: Index mismatch! Realigning merged signals...")
            # Use loc to ensure proper alignment
            signal_df['max_blood_immune'] = merged_signals.loc[signal_df.index, blood_cols].max(axis=1)
            signal_df['median_blood_immune'] = merged_signals.loc[signal_df.index, blood_cols].median(axis=1)
        else:
            signal_df['max_blood_immune'] = merged_signals[blood_cols].max(axis=1)
            signal_df['median_blood_immune'] = merged_signals[blood_cols].median(axis=1)
        
        # Debug: check calculated blood columns
        print(f"\n  DEBUG - Calculated blood signals (overall):")
        print(f"    Median blood - min: {signal_df['median_blood_immune'].min():.6f}, max: {signal_df['median_blood_immune'].max():.6f}")
        print(f"    Max blood - min: {signal_df['max_blood_immune'].min():.6f}, max: {signal_df['max_blood_immune'].max():.6f}")
    else:
        signal_df['max_blood_immune'] = 0
        signal_df['median_blood_immune'] = 0
        
    if gi_cols:
        # Fix: ensure we're aligning the indices correctly
        if not signal_df.index.equals(merged_signals.index):
            signal_df['max_gi'] = merged_signals.loc[signal_df.index, gi_cols].max(axis=1)
        else:
            signal_df['max_gi'] = merged_signals[gi_cols].max(axis=1)
    else:
        signal_df['max_gi'] = 0
    
    # Step 6: Calculate ratios and SNR using RAW reference signals (for realistic detection)
    signal_df['gi_to_tumour_ratio'] = signal_df['max_gi'] / (signal_df['tumour_reference'] + 1e-10)
    signal_df['snr_vs_blood'] = signal_df['tumour_reference'] / (signal_df['max_blood_immune'] + 1e-10)
    signal_df['snr_vs_gi'] = signal_df['tumour_reference'] / (signal_df['max_gi'] + 1e-10)
    
    print(f"Found {len(blood_cols)} blood/immune cell types with valid signals")
    print(f"Found {len(gi_cols)} GI cell types with valid signals") 
    print(f"Found {len(tumour_cols)} tumour samples")
    
    # Step 7: Apply unified filtering criteria with detailed breakdown
    print("\nApplying filters step by step:")
    
    # Start with regions that have valid tumour signal
    step1 = signal_df[~signal_df['tumour_reference'].isna()]
    print(f"  Regions with valid reference tumour signal: {len(step1):,}")
    
    # Strong tumour signal (reference case at 47.6% purity)
    step2 = step1[step1['tumour_reference'] >= min_tumour_signal]
    print(f"  After min tumour signal ≥ {min_tumour_signal}: {len(step2):,}")
    
    # Sufficient coverage - check ALL cell types have minimum coverage
    step3 = step2[step2['min_all_coverage'] >= min_coverage]
    print(f"  After coverage ≥ {min_coverage} (all cell types): {len(step3):,}")
    print(f"    Breakdown: tumor coverage OK: {(step2['tumour_coverage'] >= min_coverage).sum():,}")
    print(f"    Blood coverage OK: {(step2['min_blood_coverage'] >= min_coverage).sum():,}")  
    print(f"    GI coverage OK: {(step2['min_gi_coverage'] >= min_coverage).sum():,}")
    print(f"    All coverage OK: {len(step3):,}")
    
    # Skip control filtering - regions are pre-filtered for tumour-specificity
    step4 = step3
    print(f"  Skipping control filtering (regions pre-filtered): {len(step4):,}")
    
    # Debug blood signals before filtering
    if len(step4) > 0:
        print(f"\n  DEBUG - Blood signal distribution at step4:")
        print(f"    Median blood - min: {step4['median_blood_immune'].min():.6f}, max: {step4['median_blood_immune'].max():.6f}")
        print(f"    Median blood - 25th percentile: {step4['median_blood_immune'].quantile(0.25):.6f}")
        print(f"    Median blood - median: {step4['median_blood_immune'].median():.6f}")
        print(f"    Max blood - min: {step4['max_blood_immune'].min():.6f}, max: {step4['max_blood_immune'].max():.6f}")
    
    # Blood/immune filter
    step5 = step4[step4['median_blood_immune'] <= max_blood_signal]
    print(f"\n  After blood signal ≤ {max_blood_signal}: {len(step5):,}")
    
    # Blood SNR filter
    step6 = step5[step5['snr_vs_blood'] >= min_snr_blood]
    print(f"  After blood SNR ≥ {min_snr_blood}: {len(step6):,}")
    
    # Debug GI signals before filtering
    if len(step6) > 0:
        print(f"\n  DEBUG - GI signal distribution at step6:")
        print(f"    Max GI - value: {step6['max_gi'].iloc[0]:.6f}")
        print(f"    GI/tumour ratio - value: {step6['gi_to_tumour_ratio'].iloc[0]:.6f}")
        print(f"    Individual GI signals for the remaining region:")
        for gi_type in gi_cols:
            if gi_type in merged_signals.columns:
                idx = step6.index[0]
                print(f"      {gi_type}: {merged_signals.loc[idx, gi_type]:.6f}")
    
    # GI signal filter
    step7 = step6[step6['max_gi'] <= max_gi_signal]
    print(f"\n  After GI signal ≤ {max_gi_signal}: {len(step7):,}")
    
    # GI ratio filter
    step8 = step7[step7['gi_to_tumour_ratio'] <= max_gi_to_tumour_ratio]
    print(f"  After GI/tumour ratio ≤ {max_gi_to_tumour_ratio}: {len(step8):,}")
    
    # GI SNR filter
    filtered = step8[step8['snr_vs_gi'] >= min_snr_gi]
    print(f"  After GI SNR ≥ {min_snr_gi}: {len(filtered):,}")
    
    filtered = filtered.copy()
    
    print(f"\nFiltering results:")
    print(f"  Initial regions: {len(signal_df):,}")
    print(f"  Regions with valid reference tumour signal: {(~signal_df['tumour_reference'].isna()).sum():,}")
    print(f"  Regions passing all filters: {len(filtered):,}")
    
    # Select non-overlapping regions if requested
    if select_non_overlapping and len(filtered) > 0:
        filtered = select_non_overlapping_regions(filtered)
        print(f"  Non-overlapping regions selected: {len(filtered):,}")
    
    # Quality statistics
    if len(filtered) > 0:
        print(f"\nQuality distribution of filtered regions:")
        print(f"  Reference tumour signal: {filtered['tumour_reference'].min():.3f} - {filtered['tumour_reference'].max():.3f}")
        print(f"  Max blood/immune: {filtered['max_blood_immune'].min():.4f} - {filtered['max_blood_immune'].max():.4f}")
        print(f"  Max GI: {filtered['max_gi'].min():.3f} - {filtered['max_gi'].max():.3f}")
        if control_cols:
            print(f"  Median control: {filtered['median_control'].min():.4f} - {filtered['median_control'].max():.4f}")
            print(f"  Max control: {filtered['max_control'].min():.4f} - {filtered['max_control'].max():.4f}")
        else:
            print(f"  Control signals: Not applicable (tumour-specific regions pre-filtered)")
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
    parser.add_argument('--min_tumour_signal', type=float, default=0.6,
                       help='Minimum tumour signal at 100% purity')
    parser.add_argument('--min_coverage', type=int, default=10,
                       help='Minimum coverage required')
    parser.add_argument('--max_blood_signal', type=float, default=0.001,
                       help='Maximum median blood/immune signal')
    parser.add_argument('--max_gi_signal', type=float, default=0.2,
                       help='Maximum GI tissue signal')
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
    
    # Load tumour purity
    tumour_purity_dict = {
        'OAC_069-009_ScrBsl_tumour':0.5171,
        'OAC_071-011_ScrBsl_tumour':0.2361,
        'OAC_071-014_ScrBsl_tumour':0.07926,
        'OAC_071-021_ScrBsl_tumour':0.4766,
        'OAC_071-022_ScrBsl_tumour':0.4108,
        'OAC_071-030_ScrBsl_tumour':0.0801,
        'OAC_071-043_ScrBsl_tumour':0.4607,
        'OAC_129-001_ScrBsl_tumour':0.6921
    }
    
    print(tumour_purity_dict)

    # If analyse_thresholds is requested, do that and exit
    if args.analyse_thresholds:
        analyse_thresholds(signal_df, coverage_df, tumour_purity_dict)
        return
    
    
    # Apply filtering (without overlap selection first)
    filtered_df = unified_marker_filtering(
        signal_df, coverage_df, tumour_purity_dict,
        min_tumour_signal=args.min_tumour_signal,
        min_coverage=args.min_coverage,
        max_blood_signal=args.max_blood_signal,
        max_gi_signal=args.max_gi_signal,
        select_non_overlapping=False,  # Don't select non-overlapping yet
        max_cv_threshold=args.max_cv_threshold if args.max_cv_threshold > 0 else None
    )

    
    # Apply non-overlapping selection if requested (do this BEFORE converting to atlas format)
    if args.no_overlap and len(filtered_df) > 0:
        print(f"\nApplying non-overlapping selection to {len(filtered_df)} regions...")
        filtered_df = select_non_overlapping_regions(filtered_df)
        print(f"Selected {len(filtered_df)} non-overlapping regions")
    
    # The filtered_df already has the target column and merged signals from unified_marker_filtering
    # Just need to get the final atlas format with all cell types
    cell_type_order = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                      "Colon", "Esophagus", "Gastric", "Granulocytes", 
                      "Monocytes", "NK-cells", "OAC", "Small-intestine", "T-cells"]
    
    print("\nCalculating final weighted cell type signals for atlas...")
    atlas_df, variance_stats = calculate_weighted_cell_type_signals(
        filtered_df, coverage_df, cell_type_order, args.max_cv_threshold if args.max_cv_threshold > 0 else None
    )
    
    # Ensure proper column order (CORRECT ORDER for filter_OAC_pats.sh)
    metadata_cols = ['chr', 'start', 'end', 'startCpG', 'endCpG', 'target', 'name', 'direction']
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
    base_name = args.output_file.rsplit('.', 1)[0] if '.' in args.output_file else args.output_file
    variance_output = f"{base_name}_variance_stats.tsv"
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