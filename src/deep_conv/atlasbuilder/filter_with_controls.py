import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import glob

def identify_cfdna_compatible_regions(df: pd.DataFrame, 
                                    max_region_length: int = 300,
                                    min_cpg_density: float = 0.03,
                                    max_cpg_spread: int = 200) -> pd.DataFrame:
    """
    Identify regions compatible with cfDNA fragment analysis.
    
    Args:
        df: DataFrame with region information
        max_region_length: Maximum region length for cfDNA compatibility
        min_cpg_density: Minimum CpGs per base pair
        max_cpg_spread: Maximum distance between first and last CpG
        
    Returns:
        DataFrame filtered for cfDNA-compatible regions
    """
    # Calculate metrics
    df = df.copy()
    df['region_length'] = df['end'] - df['start']
    df['n_cpgs'] = df['endCpG'] - df['startCpG']
    df['cpg_density'] = df['n_cpgs'] / df['region_length']
    
    # For cfDNA, we need regions where a 151bp fragment can cover 4+ CpGs
    # This means CpGs should be relatively clustered
    df['cpg_span'] = df['endCpG'] - df['startCpG']  # Assuming this represents actual span
    
    # Filter for cfDNA compatibility
    compatible = df[
        (df['region_length'] <= max_region_length) & 
        (df['cpg_density'] >= min_cpg_density) &
        (df['n_cpgs'] >= 4) & 
        (df['n_cpgs'] <= 20) &  # Too many CpGs can dilute signal
        (df['cpg_span'] <= max_cpg_spread)  # CpGs should be clustered
    ].copy()
    
    print(f"cfDNA-compatible regions: {len(compatible)} of {len(df)} ({len(compatible)/len(df)*100:.1f}%)")
    print(f"Mean length: {compatible['region_length'].mean():.1f}bp")
    print(f"Mean CpGs: {compatible['n_cpgs'].mean():.1f}")
    print(f"Mean CpG density: {compatible['cpg_density'].mean():.3f}")
    
    return compatible

def calculate_control_statistics(df_work: pd.DataFrame, 
                               control_samples: List[str],
                               min_coverage: int = 10) -> pd.DataFrame:
    """
    Calculate comprehensive control statistics for tumor fraction estimation.
    """
    # Filter by coverage
    for sample in control_samples:
        cov_col = f'{sample}_coverage'
        if cov_col in df_work.columns:
            mask = df_work[cov_col] < min_coverage
            df_work.loc[mask, sample] = np.nan
    
    # Calculate statistics across controls
    control_cols = [col for col in control_samples if col in df_work.columns]
    
    # Basic statistics
    df_work['control_mean'] = df_work[control_cols].mean(axis=1, skipna=True)
    df_work['control_median'] = df_work[control_cols].median(axis=1, skipna=True)
    df_work['control_std'] = df_work[control_cols].std(axis=1, skipna=True)
    df_work['control_cv'] = df_work['control_std'] / (df_work['control_mean'] + 0.001)  # Coefficient of variation
    
    # Robust statistics (less affected by outliers)
    df_work['control_25p'] = df_work[control_cols].quantile(0.25, axis=1)
    df_work['control_75p'] = df_work[control_cols].quantile(0.75, axis=1)
    df_work['control_iqr'] = df_work['control_75p'] - df_work['control_25p']
    df_work['control_robust_cv'] = df_work['control_iqr'] / (df_work['control_median'] + 0.001)
    
    # Coverage statistics
    df_work['n_controls_covered'] = df_work[control_cols].notna().sum(axis=1)
    df_work['frac_controls_covered'] = df_work['n_controls_covered'] / len(control_cols)
    
    # Detection statistics
    df_work['n_controls_positive'] = (df_work[control_cols] > 0.005).sum(axis=1)
    df_work['frac_controls_positive'] = df_work['n_controls_positive'] / df_work['n_controls_covered']
    
    return df_work

def select_markers_for_tumor_fraction(
    df: pd.DataFrame,
    control_mv: pd.DataFrame,
    control_cov: pd.DataFrame,
    n_markers: int = 400,
    # Target parameters
    target_col: str = 'OAC',
    target_coverage_col: str = 'OAC_coverage',
    # Control parameters
    control_samples: List[str] = None,
    use_high_coverage_controls: bool = True,
    # Filtering thresholds
    max_region_length: int = 300,          # cfDNA compatibility
    min_target_signal: float = 0.3,        # 30% minimum in tumor
    min_target_coverage: int = 100,
    min_control_coverage: int = 10,        # Per-region coverage
    max_control_cv: float = 1.0,           # Coefficient of variation
    min_signal_difference: float = 0.2,     # Minimum tumor - control difference
    # Scoring weights
    signal_weight: float = 1.0,
    consistency_weight: float = 2.0,        # Emphasize consistent baseline
    coverage_weight: float = 0.5,
    cfdna_weight: float = 1.5,             # Bonus for cfDNA compatibility
    # Other parameters
    overlap_threshold: int = 50,
    require_cfDna_compatible: bool = False  # Strict requirement?
) -> Dict:
    """
    Select markers optimized for NNLS/ML-based tumor fraction estimation.
    
    Key principles:
    1. Consistent background across controls (more important than zero background)
    2. Distinct tumor vs normal patterns
    3. cfDNA compatibility for reliable measurement
    4. Sufficient coverage across all regions
    """
    
    # Define control samples
    if control_samples is None:
        all_controls = [
            'GI10873', 'GI10881', 'GI10882', 'GI10887', 'GI10888', 'GI9774',
            'TP277_Ctrl_plasma_md', 'X2881_Ctrl_plasma_md', 
            'X3161_Ctrl_plasma_md', 'X3421_Ctrl_plasma_md'
        ]
        if use_high_coverage_controls:
            # Use only high-coverage plasma controls for baseline
            control_samples = ['TP277_Ctrl_plasma_md', 'X2881_Ctrl_plasma_md', 
                             'X3161_Ctrl_plasma_md', 'X3421_Ctrl_plasma_md']
            print("Using high-coverage plasma controls for baseline estimation")
        else:
            control_samples = all_controls
    
    # Step 1: Identify cfDNA-compatible regions
    print("\nStep 1: Identifying cfDNA-compatible regions...")
    df_work = df.copy()
    df_work['region_length'] = df_work['end'] - df_work['start']
    df_work['n_cpgs'] = df_work['endCpG'] - df_work['startCpG']
    
    cfdna_compatible = identify_cfdna_compatible_regions(df_work, max_region_length)
    df_work['is_cfdna_compatible'] = df_work.index.isin(cfdna_compatible.index)
    
    if require_cfDna_compatible:
        df_work = cfdna_compatible
        print(f"Strictly using cfDNA-compatible regions: {len(df_work)}")
    
    # Step 2: Merge control data
    print("\nStep 2: Merging control data...")
    control_mv_data = control_mv[['name', 'direction'] + control_samples].copy()
    df_work = df_work.merge(control_mv_data, on=['name', 'direction'], how='left')
    
    control_cov_data = control_cov[['name', 'direction'] + control_samples].copy()
    cov_rename = {col: f'{col}_coverage' for col in control_samples}
    control_cov_data = control_cov_data.rename(columns=cov_rename)
    df_work = df_work.merge(control_cov_data, on=['name', 'direction'], how='left')
    
    # Step 3: Calculate control statistics
    print("\nStep 3: Calculating control statistics...")
    df_work = calculate_control_statistics(df_work, control_samples, min_control_coverage)
    
    # Step 4: Apply filters
    print("\nStep 4: Applying filters...")
    print(f"Initial regions: {len(df_work)}")
    
    # Basic filters
    df_work = df_work[df_work[target_col] >= min_target_signal]
    print(f"After target signal filter (≥{min_target_signal}): {len(df_work)}")
    
    df_work = df_work[df_work[target_coverage_col] >= min_target_coverage]
    print(f"After target coverage filter (≥{min_target_coverage}): {len(df_work)}")
    
    # Control coverage filter
    df_work = df_work[df_work['n_controls_covered'] >= len(control_samples) * 0.5]
    print(f"After control coverage filter (≥50% controls): {len(df_work)}")
    
    # Consistency filter - remove highly variable regions
    df_work = df_work[df_work['control_cv'] <= max_control_cv]
    print(f"After consistency filter (CV≤{max_control_cv}): {len(df_work)}")
    
    # Signal difference filter
    df_work['signal_difference'] = df_work[target_col] - df_work['control_median']
    df_work = df_work[df_work['signal_difference'] >= min_signal_difference]
    print(f"After signal difference filter (≥{min_signal_difference}): {len(df_work)}")
    
    # Step 5: Calculate scores for tumor fraction estimation
    print("\nStep 5: Calculating composite scores...")
    
    # Signal strength (normalized)
    df_work['signal_score'] = df_work['signal_difference'] / df_work['signal_difference'].max()
    
    # Consistency score (inverse of CV, normalized)
    df_work['consistency_score'] = 1 / (1 + df_work['control_cv'])
    df_work['consistency_score'] = df_work['consistency_score'] / df_work['consistency_score'].max()
    
    # Coverage score
    df_work['coverage_score'] = df_work['frac_controls_covered']
    
    # cfDNA compatibility bonus
    df_work['cfdna_score'] = df_work['is_cfdna_compatible'].astype(float)
    
    # Composite score
    df_work['composite_score'] = (
        signal_weight * df_work['signal_score'] +
        consistency_weight * df_work['consistency_score'] +
        coverage_weight * df_work['coverage_score'] +
        cfdna_weight * df_work['cfdna_score']
    )
    
    # Step 6: Remove overlapping regions
    print("\nStep 6: Removing overlapping regions...")
    df_before = len(df_work)
    df_work = remove_overlapping_regions(df_work, overlap_threshold)
    print(f"After overlap removal: {len(df_work)} (removed {df_before - len(df_work)})")
    
    # Step 7: Select markers
    print("\nStep 7: Selecting markers...")
    selected = select_diverse_markers(df_work, n_markers, target_col)
    
    # Step 8: Validation
    print("\nStep 8: Validating selection...")
    validation = validate_tf_markers(selected, control_samples, target_col)
    
    # Add rank
    selected = selected.sort_values('composite_score', ascending=False).copy()
    selected['rank'] = range(1, len(selected) + 1)
    
    return {
        'selected_markers': selected,
        'validation_results': validation,
        'all_candidates': df_work
    }

def select_diverse_markers(df: pd.DataFrame, n_markers: int, target_col: str) -> pd.DataFrame:
    """
    Select markers with diversity in signal levels and genomic distribution.
    """
    selected_dfs = []
    
    # Category 1: High signal, cfDNA-compatible, low variance
    high_quality = df[
        (df['is_cfdna_compatible']) & 
        (df['control_cv'] < 0.5) &
        (df[target_col] > 0.5)
    ]
    if len(high_quality) > 0:
        n_select = min(len(high_quality), n_markers // 3)
        selected_dfs.append(high_quality.nlargest(n_select, 'composite_score'))
        print(f"Selected {n_select} high-quality cfDNA-compatible markers")
    
    # Category 2: Moderate signal, very consistent
    consistent = df[
        (df['control_cv'] < 0.3) &
        (df[target_col] > 0.3) &
        (df['signal_difference'] > 0.25)
    ]
    if len(consistent) > 0:
        remaining = n_markers - sum(len(s) for s in selected_dfs)
        n_select = min(len(consistent), remaining // 2)
        # Exclude already selected
        already_selected = pd.concat(selected_dfs).index if selected_dfs else pd.Index([])
        available = consistent[~consistent.index.isin(already_selected)]
        if len(available) > 0:
            selected_dfs.append(available.nlargest(n_select, 'composite_score'))
            print(f"Selected {n_select} highly consistent markers")
    
    # Category 3: Fill with best remaining
    if selected_dfs:
        selected = pd.concat(selected_dfs)
    else:
        selected = pd.DataFrame()
    
    remaining = n_markers - len(selected)
    if remaining > 0:
        available = df[~df.index.isin(selected.index)]
        if len(available) > 0:
            additional = available.nlargest(remaining, 'composite_score')
            selected = pd.concat([selected, additional])
            print(f"Selected {len(additional)} additional markers")
    
    # Ensure we don't exceed n_markers
    return selected.iloc[:n_markers]

def remove_overlapping_regions(df: pd.DataFrame, min_distance: int = 50) -> pd.DataFrame:
    """
    Remove overlapping regions, keeping those with highest scores.
    """
    df_sorted = df.sort_values(['chr', 'composite_score'], ascending=[True, False])
    selected_indices = []
    
    for chrom in df_sorted['chr'].unique():
        chrom_df = df_sorted[df_sorted['chr'] == chrom]
        selected_regions = []
        
        for idx, row in chrom_df.iterrows():
            start, end = row['startCpG'], row['endCpG']
            
            # Check distance from selected regions
            too_close = False
            for sel_start, sel_end in selected_regions:
                if start <= sel_end and end >= sel_start:
                    too_close = True
                    break
                distance = min(abs(start - sel_end), abs(sel_start - end))
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected_indices.append(idx)
                selected_regions.append((start, end))
    
    return df.loc[selected_indices]

def validate_tf_markers(selected: pd.DataFrame, control_samples: List[str], 
                       target_col: str) -> Dict:
    """
    Validate markers for tumor fraction estimation.
    """
    results = {
        'n_markers': len(selected),
        'tumor_signal': {
            'mean': selected[target_col].mean(),
            'std': selected[target_col].std(),
            'min': selected[target_col].min(),
            'max': selected[target_col].max()
        },
        'control_baseline': {
            'mean': selected['control_mean'].mean(),
            'std_of_means': selected['control_mean'].std(),
            'mean_cv': selected['control_cv'].mean(),
            'max_cv': selected['control_cv'].max()
        },
        'signal_difference': {
            'mean': selected['signal_difference'].mean(),
            'min': selected['signal_difference'].min(),
            'std': selected['signal_difference'].std()
        },
        'cfdna_compatibility': {
            'n_compatible': selected['is_cfdna_compatible'].sum(),
            'pct_compatible': selected['is_cfdna_compatible'].sum() / len(selected) * 100,
            'mean_length': selected['region_length'].mean(),
            'mean_cpgs': selected['n_cpgs'].mean()
        },
        'coverage': {
            'mean_controls_covered': selected['frac_controls_covered'].mean(),
            'min_controls_covered': selected['n_controls_covered'].min()
        },
        'chromosomes': selected['chr'].nunique()
    }
    
    # Check for potential issues
    high_variance = selected[selected['control_cv'] > 0.5]
    if len(high_variance) > 0:
        print(f"\nWARNING: {len(high_variance)} markers have high variance (CV>0.5)")
        print(high_variance[['chr', 'startCpG', 'endCpG', 'control_mean', 
                            'control_cv', target_col]].head())
    
    # Distribution check
    print(f"\n=== MARKER DISTRIBUTION ===")
    print(f"Tumor signal distribution:")
    print(f"  0.3-0.5: {((selected[target_col] >= 0.3) & (selected[target_col] < 0.5)).sum()}")
    print(f"  0.5-0.7: {((selected[target_col] >= 0.5) & (selected[target_col] < 0.7)).sum()}")
    print(f"  0.7-1.0: {(selected[target_col] >= 0.7).sum()}")
    
    print(f"\nControl baseline distribution:")
    print(f"  0-1%: {(selected['control_mean'] < 0.01).sum()}")
    print(f"  1-3%: {((selected['control_mean'] >= 0.01) & (selected['control_mean'] < 0.03)).sum()}")
    print(f"  3-5%: {((selected['control_mean'] >= 0.03) & (selected['control_mean'] < 0.05)).sum()}")
    print(f"  >5%: {(selected['control_mean'] >= 0.05).sum()}")
    
    return results

def prepare_for_nnls(selected_markers: pd.DataFrame, 
                    control_samples: List[str],
                    target_col: str = 'OAC') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare reference matrix for NNLS tumor fraction estimation.
    
    Returns:
        reference_matrix: (n_markers, 2) matrix with [normal, tumor] signals
        marker_names: Array of marker identifiers
    """
    # Create reference matrix
    normal_signal = selected_markers['control_median'].values
    tumor_signal = selected_markers[target_col].values
    
    reference_matrix = np.column_stack([normal_signal, tumor_signal])
    marker_names = selected_markers['name'].values
    
    print(f"\nReference matrix shape: {reference_matrix.shape}")
    print(f"Normal signal range: {normal_signal.min():.3f} - {normal_signal.max():.3f}")
    print(f"Tumor signal range: {tumor_signal.min():.3f} - {tumor_signal.max():.3f}")
    
    return reference_matrix, marker_names

def main():
    df = pd.read_parquet(glob.glob("/users/zetzioni/sharedscratch/loyfer_atlas/marker_regions/parts/*OAC*.parquet"))
    df = df.drop_duplicates(["name"])
    max_background_threshold = 0.005 
    min_signal_to_background_ratio = 100  

    df = df[
        (df['max_background'] < max_background_threshold) &
        (df['snr'] > min_signal_to_background_ratio)
    ]
    
    control_mv = pd.read_parquet("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/marker_values.parquet")
    control_cov = pd.read_parquet("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/coverage.parquet")


    results = select_markers_for_tumor_fraction(
        df=df,
        control_mv=control_mv,
        control_cov=control_cov,
        n_markers=400,
        # Use high-coverage controls for stable baseline
        use_high_coverage_controls=True,
        # Balance between cfDNA compatibility and signal
        require_cfDna_compatible=False,  # Don't strictly require, but give bonus
        max_region_length=300,
        # Focus on consistency for NNLS
        max_control_cv=0.8,  # Allow some variance but not too much
        min_signal_difference=0.25,  # 25% difference minimum
        # Adjust weights to emphasize consistency
        consistency_weight=2.0,
        cfdna_weight=1.5
    )