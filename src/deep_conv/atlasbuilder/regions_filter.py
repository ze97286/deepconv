import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def prefilter_regions_with_min_coverage(mv_df, cov_df, control_mv_df, control_cov_df,
                                       min_coverage_per_sample=5,
                                       min_tumor_samples_covered=4,
                                       min_total_tumor_coverage=100,
                                       min_cell_types_covered=3):
    """
    Pre-filter regions based on MEANINGFUL coverage requirements
    """
    sample_cols = [col for col in cov_df.columns if col not in ['name', 'direction']]
    control_sample_cols = [col for col in control_cov_df.columns if col not in ['name', 'direction']]
    tumor_samples = [col for col in sample_cols if 'OAC' in col or 'tumour' in col]
    high_cov_controls = [col for col in control_sample_cols if not col.startswith('Control_GI')]
    print(f"Found {len(tumor_samples)} tumor samples and {len(high_cov_controls)} high-coverage controls")
    # Filter 1: Require MEANINGFUL coverage in tumor samples
    # Count samples with at least min_coverage_per_sample reads
    tumor_well_covered = (cov_df[tumor_samples] >= min_coverage_per_sample).sum(axis=1)
    tumor_covered = tumor_well_covered >= min_tumor_samples_covered
    # Filter 2: Total tumor coverage (higher threshold)
    total_tumor_coverage = cov_df[tumor_samples].sum(axis=1)
    sufficient_tumor_coverage = total_tumor_coverage >= min_total_tumor_coverage
    # Filter 3: Meaningful coverage in normal cell types
    cell_type_groups = {}
    for col in sample_cols:
        if col not in tumor_samples:
            cell_type = col.split('_')[0]
            if cell_type not in cell_type_groups:
                cell_type_groups[cell_type] = []
            cell_type_groups[cell_type].append(col)
    # For each cell type, check if ANY sample has meaningful coverage
    cell_types_with_coverage = pd.DataFrame()
    for cell_type, samples in cell_type_groups.items():
        # Cell type is "covered" if at least one sample has ≥min_coverage
        cell_types_with_coverage[cell_type] = (cov_df[samples] >= min_coverage_per_sample).any(axis=1)
    num_cell_types_covered = cell_types_with_coverage.sum(axis=1)
    sufficient_normal_coverage = num_cell_types_covered >= min_cell_types_covered
    # Filter 4: Meaningful coverage in controls
    # Require at least 5 total reads across high-coverage controls
    total_control_coverage = control_cov_df[high_cov_controls].sum(axis=1)
    control_detectable = total_control_coverage >= 5
    # Combine all filters
    keep_regions = (tumor_covered & 
                   sufficient_tumor_coverage & 
                   sufficient_normal_coverage & 
                   control_detectable)
    # Print filtering cascade
    print(f"\nFiltering cascade (with min coverage = {min_coverage_per_sample}):")
    print(f"  Starting regions: {len(cov_df):,}")
    print(f"  After tumor coverage (≥{min_tumor_samples_covered} samples with ≥{min_coverage_per_sample} reads): {tumor_covered.sum():,} ({tumor_covered.sum()/len(cov_df)*100:.1f}%)")
    cumulative = tumor_covered
    print(f"  After total tumor coverage (≥{min_total_tumor_coverage}): {(cumulative & sufficient_tumor_coverage).sum():,} ({(cumulative & sufficient_tumor_coverage).sum()/len(cov_df)*100:.1f}%)")
    cumulative = cumulative & sufficient_tumor_coverage
    print(f"  After normal coverage (≥{min_cell_types_covered} cell types with ≥{min_coverage_per_sample} reads): {(cumulative & sufficient_normal_coverage).sum():,} ({(cumulative & sufficient_normal_coverage).sum()/len(cov_df)*100:.1f}%)")
    cumulative = cumulative & sufficient_normal_coverage
    print(f"  After control detectability (≥5 total reads): {(cumulative & control_detectable).sum():,} ({(cumulative & control_detectable).sum()/len(cov_df)*100:.1f}%)")
    print(f"  Final retained: {keep_regions.sum():,} ({keep_regions.sum()/len(cov_df)*100:.1f}%)")
    # Apply filters
    mv_filtered = mv_df[keep_regions].copy()
    cov_filtered = cov_df[keep_regions].copy()
    control_mv_filtered = control_mv_df[keep_regions].copy()
    control_cov_filtered = control_cov_df[keep_regions].copy()
    return mv_filtered, cov_filtered, control_mv_filtered, control_cov_filtered

def check_residual_pattern(X, y, regression_results):
    """Check if residuals show systematic patterns indicating heterogeneity"""
    if len(X) < 4:
        return 0
    # Calculate residuals
    slope, intercept = regression_results[0], regression_results[1]
    predicted = X.flatten() * slope + intercept
    residuals = y - predicted
    # Check for patterns:
    # 1. Increasing variance with X (heteroscedasticity)
    X_median = np.median(X)
    low_X_residuals = residuals[X.flatten() <= X_median]
    high_X_residuals = residuals[X.flatten() > X_median]
    if len(low_X_residuals) > 1 and len(high_X_residuals) > 1:
        variance_ratio = np.var(high_X_residuals) / (np.var(low_X_residuals) + 0.01)
        heteroscedasticity_score = max(0, np.log2(variance_ratio))
    else:
        heteroscedasticity_score = 0
    # 2. Non-random residual distribution (runs test)
    # Count runs of positive/negative residuals
    residual_signs = np.sign(residuals)
    runs = 1
    for i in range(1, len(residual_signs)):
        if residual_signs[i] != residual_signs[i-1]:
            runs += 1
    # Expected runs for random residuals
    n_pos = np.sum(residual_signs > 0)
    n_neg = np.sum(residual_signs < 0)
    if n_pos > 0 and n_neg > 0:
        expected_runs = 1 + (2 * n_pos * n_neg) / len(residuals)
        runs_score = abs(runs - expected_runs) / (expected_runs + 1)
    else:
        runs_score = 0
    # Combined heterogeneity score
    return heteroscedasticity_score + runs_score

def calculate_robust_regression_stats(X, y, use_robust=True):
    """Calculate regression with outlier resistance and heterogeneity detection"""
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return {
            'r2': 0, 
            'slope': 0, 
            'intercept': 0, 
            'p_value': 1, 
            'std_err': np.inf, 
            'n_points': mask.sum(),
            'heterogeneity_score': 0,
            'use_robust': False
        }
    X_clean = X[mask]
    y_clean = y[mask]
    # Standard regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(X_clean.flatten(), y_clean)
    # Calculate residuals
    predicted = X_clean * slope + intercept
    residuals = y_clean - predicted
    residual_std = np.std(residuals)
    # Simple outlier detection: points > 2.5 SD from regression line
    outlier_mask = np.abs(residuals) > 2.5 * residual_std
    n_outliers = np.sum(outlier_mask)
    # Heterogeneity score based on:
    # 1. Fraction of outliers
    # 2. Spread of residuals relative to signal range
    outlier_fraction = n_outliers / len(y_clean)
    relative_residual_spread = residual_std / (np.ptp(y_clean) + 0.01)
    heterogeneity_score = outlier_fraction + relative_residual_spread
    # Use robust regression if we have outliers and enough points
    use_robust_regression = use_robust and n_outliers > 0 and len(y_clean) >= 4
    if use_robust_regression:
        try:
            # Fit without outliers
            X_no_outliers = X_clean[~outlier_mask]
            y_no_outliers = y_clean[~outlier_mask]
            if len(X_no_outliers) >= 3:
                robust_slope, robust_intercept, _, _, _ = stats.linregress(
                    X_no_outliers.flatten(), y_no_outliers
                )
                # Use robust estimates if they differ substantially
                if abs(robust_slope - slope) / (abs(slope) + 0.01) > 0.2:
                    slope = robust_slope
                    intercept = robust_intercept
                    use_robust_regression = True
                else:
                    use_robust_regression = False
        except:
            use_robust_regression = False
    return {
        'r2': r_value**2,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_value,
        'std_err': std_err,
        'n_points': len(y_clean),
        'heterogeneity_score': heterogeneity_score,
        'use_robust': use_robust_regression,
        'n_outliers': n_outliers
    }

def calculate_weighted_cell_type_signals(mv_df, cov_df, cell_type_samples):
    """Calculate inverse-variance weighted signal for a cell type, handling missing data"""
    signals = mv_df[cell_type_samples]
    coverages = cov_df[cell_type_samples]
    # Create mask for valid (non-NaN) values
    valid_mask = ~signals.isna()
    # Calculate variance only using valid values (need at least 2 valid samples)
    valid_counts = valid_mask.sum(axis=1)
    variances = signals.var(axis=1, ddof=1, skipna=True)
    # Set variance to NaN if we have fewer than 2 valid samples
    variances[valid_counts < 2] = np.nan
    # For regions with only 1 valid sample, use that sample's value directly
    single_sample_mask = (valid_counts == 1)
    # Inverse variance weights, only for valid values
    weights = pd.DataFrame(index=coverages.index, columns=coverages.columns)
    for col in cell_type_samples:
        # Only assign weights where we have valid signal AND variance
        mask = valid_mask[col] & ~variances.isna()
        weights.loc[mask, col] = coverages.loc[mask, col] / (variances[mask] + 0.01)
    # Fill NaN weights with 0
    weights = weights.fillna(0)
    # Normalize weights to sum to 1 for each region (only across valid samples)
    weight_sums = weights.sum(axis=1)
    weights = weights.div(weight_sums.replace(0, 1), axis=0)  # Avoid division by zero
    # Calculate weighted mean (will be NaN if no valid samples)
    weighted_signal = (signals * weights).sum(axis=1, skipna=True)
    # For single sample regions, use that sample's value
    for i in single_sample_mask[single_sample_mask].index:
        valid_col = valid_mask.loc[i][valid_mask.loc[i]].index[0]
        weighted_signal.loc[i] = signals.loc[i, valid_col]
    # Set to NaN if no valid samples
    weighted_signal[valid_counts == 0] = np.nan
    # Effective sample size (accounting for weighting)
    eff_n = pd.Series(index=weights.index)
    mask = weight_sums > 0
    eff_n[mask] = weight_sums[mask]**2 / (weights[mask]**2).sum(axis=1)
    eff_n[~mask] = 0
    eff_n[single_sample_mask] = 1
    return weighted_signal, eff_n

def process_region_batch(batch_data):
    """Fixed batch processing with constrained extrapolation"""
    batch_idx, mv_batch, cov_batch, tumor_samples, tumor_purities = batch_data
    batch_size = len(mv_batch)
    # Initialize all result arrays
    results = {
        'r2': np.zeros(batch_size),
        'slope': np.zeros(batch_size),
        'intercept': np.zeros(batch_size),
        'p_value': np.ones(batch_size),
        'std_err': np.full(batch_size, np.inf),
        'heterogeneity_score': np.zeros(batch_size),
        'used_robust_regression': np.zeros(batch_size, dtype=bool),
        'n_outliers': np.zeros(batch_size),
        'tumor_100_signal': np.zeros(batch_size),
        'extrapolation_uncertainty': np.zeros(batch_size)
    }
    # Process each region in the batch
    for i in range(batch_size):
        tumor_signals = mv_batch[tumor_samples].iloc[i].values
        if not np.isnan(tumor_signals).all():
            stats_dict = calculate_robust_regression_stats(tumor_purities, tumor_signals)
            # Store all results
            for key in ['r2', 'slope', 'intercept', 'p_value', 'std_err', 
                       'heterogeneity_score', 'n_outliers']:
                if key in stats_dict:
                    results[key][i] = stats_dict[key]
            results['used_robust_regression'][i] = stats_dict['use_robust']
            signal_100_raw = stats_dict['intercept'] + stats_dict['slope'] * 1.0
            signal_100 = np.clip(signal_100_raw, 0.0, 1.0)
            results['tumor_100_signal'][i] = signal_100
            # Calculate uncertainty
            uncertainty_multiplier = 1.5 if stats_dict['use_robust'] else 1.0
            if stats_dict['n_points'] > 2:
                valid_mask = ~np.isnan(tumor_signals)
                x_mean = tumor_purities[valid_mask].mean()
                x_var = tumor_purities[valid_mask].var()
                if x_var > 0:
                    se_pred = stats_dict['std_err'] * np.sqrt(
                        1/stats_dict['n_points'] + (1.0 - x_mean)**2 / (x_var * stats_dict['n_points'])
                    )
                    ci_width = 1.96 * se_pred * uncertainty_multiplier
                else:
                    ci_width = np.inf
            else:
                ci_width = np.inf
            results['extrapolation_uncertainty'][i] = ci_width
    return batch_idx, results

def calculate_region_scores_vectorized(mv_df, cov_df, control_mv_df, control_cov_df, 
                                           tumor_purity_dict, 
                                           blood_immune_types=['B-cells', 'T-cells', 'NK-cells', 
                                                             'Granulocytes', 'Monocytes',
                                                             'CD34-erythroblasts', 'CD34-megakaryocytes'],
                                           tissue_types=['Esophagus', 'Gastric', 'Colon', 'Small-intestine'],
                                           batch_size=10000, n_jobs=-1):
    """
    FIXED version with all corrections applied
    """
    print(f"Processing {len(mv_df):,} regions with full feature set...")
    
    # Get sample groups
    tumor_samples = [col for col in mv_df.columns if 'OAC' in col or 'tumour' in col]
    control_samples = [col for col in control_mv_df.columns if col not in ['name', 'direction']]
    high_cov_controls = [col for col in control_samples if not col.startswith('GI')]
    
    # FIX 1: Don't divide by 100 if purities are already in 0-1 scale
    tumor_purities = np.array([tumor_purity_dict[sample.split('_', 1)[1]] 
                              for sample in tumor_samples])  # No /100.0!
    
    # 1. PARALLELIZED TUMOR-PURITY CORRELATION
    print("Step 1: Calculating tumor correlations with heterogeneity detection...")
    
    # Create batches
    n_batches = (len(mv_df) + batch_size - 1) // batch_size
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(mv_df))
        batches.append((
            i,
            mv_df.iloc[start_idx:end_idx],
            cov_df.iloc[start_idx:end_idx],
            tumor_samples,
            tumor_purities
        ))
    
    # Process in parallel
    if n_jobs == 1:
        results = [process_region_batch(batch) for batch in tqdm(batches)]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_region_batch)(batch) 
            for batch in tqdm(batches, desc="Processing batches")
        )
    
    # Sort and combine results
    results.sort(key=lambda x: x[0])
    
    # Initialize scores DataFrame
    scores = pd.DataFrame(index=mv_df.index)
    
    # Combine batch results
    for key in results[0][1].keys():
        scores[key] = np.concatenate([r[1][key] for r in results])
    
    # Calculate heterogeneity-adjusted R²
    print("Step 2: Calculating heterogeneity-adjusted metrics...")
    tumor_coverage_cv = cov_df[tumor_samples].std(axis=1) / (cov_df[tumor_samples].mean(axis=1) + 1)
    scores['heterogeneity_adjusted_r2'] = (
        scores['r2'] * 
        (1 / (1 + tumor_coverage_cv)) * 
        (1 / (1 + scores['heterogeneity_score']))
    )
    
    # 2. CELL TYPE SPECIFIC SIGNALS
    print("Step 3: Calculating cell type signals with variance weighting...")
    
    # Group samples by cell type
    cell_type_groups = {}
    for col in mv_df.columns:
        if col not in tumor_samples + ['name', 'direction']:
            cell_type = col.split('_')[0]
            if cell_type not in cell_type_groups:
                cell_type_groups[cell_type] = []
            cell_type_groups[cell_type].append(col)
    
    # Calculate weighted signals for each cell type
    blood_immune_signals = []
    tissue_signals = []
    all_normal_samples = []
    
    for cell_type, samples in tqdm(cell_type_groups.items(), desc="Processing cell types"):
        if len(samples) >= 1:
            if len(samples) == 1:
                weighted_signal = mv_df[samples[0]]
            else:
                weighted_signal, eff_n = calculate_weighted_cell_type_signals(
                    mv_df, cov_df, samples
                )
            
            all_normal_samples.extend(samples)
            
            if cell_type in blood_immune_types:
                blood_immune_signals.append(weighted_signal)
            elif cell_type in tissue_types:
                tissue_signals.append(weighted_signal)
    
    # FIX 2: Use MAXIMUM of all normal samples, not percentiles
    print("Step 4: Calculating differential methylation (using maximum)...")
    
    # Get maximum signal across ALL normal samples
    normal_max = mv_df[all_normal_samples].max(axis=1)
    
    # Simple differential: tumor - max(normal)
    scores['differential'] = np.log2(
        (scores['tumor_100_signal'] + 0.01) / (normal_max + 0.01)
    )
    
    # Also calculate log-ratio for regions where normal_max > 0
    scores['log_ratio_vs_normal_max'] = np.log2(
        (scores['tumor_100_signal'] + 0.01) / (normal_max + 0.01)
    )
    
    # 4. CONTROL BACKGROUND
    print("Step 5: Analyzing control signals...")
    
    high_cov_signal = control_mv_df[high_cov_controls].fillna(0)
    high_cov_coverage = control_cov_df[high_cov_controls]
    
    # FIX 3: Use maximum control signal, not average
    scores['control_signal_max'] = control_mv_df[high_cov_controls].max(axis=1)
    scores['control_signal_avg'] = (high_cov_signal * high_cov_coverage).sum(axis=1) / (high_cov_coverage.sum(axis=1) + 1)
    
    control_cv = high_cov_signal.std(axis=1) / (high_cov_signal.mean(axis=1) + 0.01)
    scores['control_consistency'] = 1 / (1 + control_cv)
    
    # 5. COVERAGE QUALITY SCORES
    print("Step 6: Calculating coverage metrics...")
    
    min_tumor_coverage = cov_df[tumor_samples].min(axis=1)
    scores['min_tumor_coverage'] = np.log10(min_tumor_coverage + 1)
    
    coverage_balance = cov_df[tumor_samples].min(axis=1) / (cov_df[tumor_samples].max(axis=1) + 1)
    scores['coverage_balance'] = coverage_balance
    
    # 6. WITHIN CELL TYPE CONSISTENCY
    print("Step 7: Calculating cell type consistency...")
    
    consistency_scores = []
    for cell_type, samples in cell_type_groups.items():
        if len(samples) >= 2:
            ct_cv = mv_df[samples].std(axis=1) / (mv_df[samples].mean(axis=1) + 0.01)
            consistency_scores.append(1 / (1 + ct_cv))
    
    if consistency_scores:
        scores['cell_type_consistency'] = pd.concat(consistency_scores, axis=1).mean(axis=1)
    else:
        scores['cell_type_consistency'] = 1
    
    # FIX 4: Add sanity checks
    scores['extrapolation_valid'] = scores['tumor_100_signal'] <= 1.0
    
    # 7. REVISED COMBINED SCORE
    print("Step 8: Calculating combined scores...")
    
    scores['combined'] = (
        scores['heterogeneity_adjusted_r2'] * 3.0 +
        scores['differential'] * 10.0 +  # Much higher weight on differential
        (1 / (scores['control_signal_max'] + 0.01)) * 3.0 +  # Much stronger control penalty
        scores['min_tumor_coverage'] * 0.5 +
        scores['coverage_balance'] * 0.5 +
        scores['cell_type_consistency'] * 1.0 -
        np.log10(scores['intercept'].abs() + 0.01) * 1.0 -
        np.log10(scores['p_value'] + 1e-10) * 0.5 -
        np.log10(scores['std_err'] + 0.01) * 0.5 -
        scores['n_outliers'] * 0.5 -
        (~scores['extrapolation_valid']) * 100.0  # Huge penalty for invalid extrapolation
    )
    
    # Revised quality flags
    scores['high_quality'] = (
        (scores['r2'] > 0.5) & 
        (scores['p_value'] < 0.05) &
        (scores['intercept'] < 0.1) &
        (scores['differential'] > 0.3) &  # At least 30% higher than any normal
        (scores['control_signal_max'] < 0.05) &  # No control sample > 5%
        (scores['min_tumor_coverage'] > np.log10(10)) &
        (scores['heterogeneity_score'] < 0.5) &
        (scores['n_outliers'] <= 1) &
        (scores['extrapolation_valid'])  # Must have valid extrapolation
    )
    
    # Summary statistics
    print(f"\nCompleted!")
    print(f"  Total regions: {len(scores):,}")
    print(f"  High quality regions: {scores['high_quality'].sum():,}")
    print(f"  Regions with invalid extrapolation: {(~scores['extrapolation_valid']).sum():,}")
    print(f"  Mean control signal (max): {scores['control_signal_max'].mean():.3f}")
    
    return scores

def remove_overlapping_regions(scores_df, mv_df, max_overlap=0.5):
    """Remove overlapping regions, keeping the highest scoring ones"""
    # Extract chromosome positions from names
    regions = []
    for idx in scores_df.index:
        name = mv_df.loc[idx, 'name']
        try:
            chrom, pos_range = name.split(':')
            start, end = map(int, pos_range.split('-'))
            regions.append((idx, chrom, start, end, scores_df.loc[idx, 'combined']))
        except:
            print(f"Skipping malformed region name: {name}")
            continue
    
    # Sort by score (descending)
    regions.sort(key=lambda x: x[4], reverse=True)
    
    # Keep non-overlapping regions
    kept_indices = []
    kept_regions = []
    
    for idx, chrom, start, end, score in regions:
        # Check overlap with already kept regions
        overlaps = False
        for kept_idx, kept_chrom, kept_start, kept_end, _ in kept_regions:
            if chrom == kept_chrom:
                overlap_start = max(start, kept_start)
                overlap_end = min(end, kept_end)
                if overlap_start < overlap_end:
                    overlap_size = overlap_end - overlap_start
                    region_size = end - start
                    if region_size > 0 and overlap_size / region_size > max_overlap:
                        overlaps = True
                        break
        
        if not overlaps:
            kept_indices.append(idx)
            kept_regions.append((idx, chrom, start, end, score))
    
    return scores_df.loc[kept_indices]

def format_regions_for_output(scores_df, mv_df, cov_df, regions_mapping):
    """Format regions in the requested output format with all scoring data"""
    
    # Cell type groups for aggregation
    cell_type_mapping = {
        'B-cells': ['B-cells'],
        'CD34-erythroblasts': ['CD34-erythroblasts'],
        'CD34-megakaryocytes': ['CD34-megakaryocytes'],
        'Colon': ['Colon'],
        'Esophagus': ['Esophagus'],
        'Gastric': ['Gastric'],
        'Granulocytes': ['Granulocytes'],
        'Monocytes': ['Monocytes'],
        'NK-cells': ['NK-cells'],
        'OAC': ['OAC'],
        'Small-intestine': ['Small-intestine'],
        'T-cells': ['T-cells']
    }
    
    # Create a mapping from region name to CpG info
    cpg_info = regions_mapping.set_index('name')[['startCpG', 'endCpG']].to_dict('index')
    
    output_rows = []
    
    for idx in scores_df.index:
        region_name = mv_df.loc[idx, 'name']
        direction = mv_df.loc[idx, 'direction']
        
        # Parse region
        try:
            chrom, pos_range = region_name.split(':')
            start, end = map(int, pos_range.split('-'))
        except:
            print(f"Skipping malformed region: {region_name}")
            continue
        
        # Get CpG counts from regions_mapping
        if region_name in cpg_info:
            startCpG = cpg_info[region_name]['startCpG']
            endCpG = cpg_info[region_name]['endCpG']
        else:
            print(f"Warning: No CpG info for {region_name}, using defaults")
            startCpG = 1
            endCpG = 3
        
        # Get reference values for each cell type
        cell_type_values = {}
        
        for cell_type, prefixes in cell_type_mapping.items():
            if cell_type == 'OAC':
                # Use the extrapolated 100% tumor signal
                cell_type_values[cell_type] = scores_df.loc[idx, 'tumor_100_signal']
            else:
                # Get columns for this cell type
                cols = [col for col in mv_df.columns 
                       if any(col.startswith(prefix + '_') for prefix in prefixes)]
                
                if cols:
                    # Use the variance-weighted mean or maximum
                    # Here using maximum to be conservative
                    values = mv_df.loc[idx, cols]
                    cell_type_values[cell_type] = values.max()
                else:
                    cell_type_values[cell_type] = 0.0
        
        # Determine target (1 for tumor-specific, 0 for normal)
        target = 1 if scores_df.loc[idx, 'high_quality'] else 0
        
        # Build row with basic info
        row = {
            'chr': chrom,
            'start': start,
            'end': end,
            'startCpG': startCpG,
            'endCpG': endCpG,
            'name': region_name,
            'direction': direction,
            'target': target
        }
        
        # Add cell type values
        for cell_type in ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes",
                         "Colon", "Esophagus", "Gastric", "Granulocytes",
                         "Monocytes", "NK-cells", "OAC", "Small-intestine", "T-cells"]:
            row[cell_type] = cell_type_values.get(cell_type, 0.0)
        
        # Add all scoring metrics
        scoring_columns = [
            'r2', 'slope', 'intercept', 'p_value', 'std_err', 'heterogeneity_score',
            'used_robust_regression', 'n_outliers', 'tumor_100_signal',
            'extrapolation_uncertainty', 'heterogeneity_adjusted_r2',
            'differential', 'log_ratio_vs_normal_max', 'control_signal_max',
            'control_signal_avg', 'control_consistency', 'min_tumor_coverage',
            'coverage_balance', 'cell_type_consistency', 'extrapolation_valid',
            'combined'
        ]
        
        for col in scoring_columns:
            if col in scores_df.columns:
                row[col] = scores_df.loc[idx, col]
            else:
                print(f"Warning: Column {col} not found in scores")
                row[col] = None
        
        output_rows.append(row)
    
    # Create DataFrame with all columns
    output_df = pd.DataFrame(output_rows)
    
    # Reorder columns to have the basic info first, then cell types, then scoring metrics
    basic_cols = ['chr', 'start', 'end', 'startCpG', 'endCpG', 'name', 'direction', 'target']
    cell_type_cols = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes",
                      "Colon", "Esophagus", "Gastric", "Granulocytes",
                      "Monocytes", "NK-cells", "OAC", "Small-intestine", "T-cells"]
    
    column_order = basic_cols + cell_type_cols + scoring_columns
    output_df = output_df[column_order]
    
    return output_df

tumor_purity_dict = {
        '069-009_ScrBsl_tumour_cna_corrected':0.5171,
        '071-011_ScrBsl_tumour_cna_corrected':0.2361,
        '071-014_ScrBsl_tumour_cna_corrected':0.07926,
        '071-021_ScrBsl_tumour_cna_corrected':0.4766,
        '071-022_ScrBsl_tumour_cna_corrected':0.4108,
        '071-030_ScrBsl_tumour_cna_corrected':0.0801,
        '071-043_ScrBsl_tumour_cna_corrected':0.4607,
        '129-001_ScrBsl_tumour_cna_corrected':0.6921
    }


def process(pat_dir, control_dir, marker_regions_dir, min_cpgs, chr):
    print("reading cov")
    cov = pd.read_parquet(f"{pat_dir}/l{min_cpgs}_chr{chr}_coverage.parquet")
    print("reading control_cov")
    control_cov = pd.read_parquet(f"{control_dir}/l{min_cpgs}_chr{chr}_coverage.parquet")
    print("reading marker_values")
    mv = pd.read_parquet(f"{pat_dir}/l{min_cpgs}_chr{chr}_marker_values.parquet")
    print("reading control marker_values")
    control_mv = pd.read_parquet(f"{control_dir}/l{min_cpgs}_chr{chr}_marker_values.parquet")
    mv_filtered, cov_filtered, control_mv_filtered, control_cov_filtered = prefilter_regions_with_min_coverage(mv, cov, control_mv, control_cov)
    scores = calculate_region_scores_vectorized(
        mv_filtered, cov_filtered, control_mv_filtered, control_cov_filtered,
        tumor_purity_dict,
        batch_size=10000, 
        n_jobs=8  
    )
    scores['region_name'] = mv_filtered.loc[scores.index, 'name']
    regions_mapping = pd.read_csv(f"{marker_regions_dir}/regions_chr{chr}_{min_cpgs}_500.bed.gz", sep="\t")
    print("Selecting high-quality regions...")
    high_quality_regions = scores[scores['high_quality']]
    print(f"Found {len(high_quality_regions)} high-quality regions")
    print("\nRemoving overlapping regions...")
    non_overlapping = remove_overlapping_regions(high_quality_regions, mv_filtered)
    print(f"Kept {len(non_overlapping)} non-overlapping regions")
    print("\nFormatting for output...")
    final_output = format_regions_for_output(non_overlapping, mv_filtered, cov_filtered, regions_mapping)
    final_output.to_parquet(f"{pat_dir}/l{min_cpgs}_chr{chr}_final_regions", index=False)
    return final_output

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process pat files for UXM analysis')
    parser.add_argument('--pat_dir', required=True, help='Directory containing tumor pat files')
    parser.add_argument('--control_dir', help='Directory containing control pat files')
    parser.add_argument('--marker_regions_dir', help='Directory containing regions mapping files')
    parser.add_argument('--min_cpgs', type=int, required=True, help='Minimum CpGs required')
    parser.add_argument("--chr", type=int, required=True, help="Chromosome number (1-22)")
    args = parser.parse_args()

    process(args.pat_dir, args.control_dir, args.marker_regions_dir, args.min_cpgs, args.chr)

if __name__ == "__main__":
    main()
