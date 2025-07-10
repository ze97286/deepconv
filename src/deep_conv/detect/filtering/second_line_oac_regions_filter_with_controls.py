import numpy as np
import pandas as pd

def prepare_control_data_efficient(control_mv, control_cov, min_coverage=5):
    """
    Efficiently prepare control data for large-scale analysis
    """
    print(f"Preparing control data: {control_mv.shape[0]:,} regions × {control_mv.shape[1]} samples")
    
    # Apply minimum coverage filter
    control_mv_filtered = control_mv.copy()
    control_mv_filtered[control_cov < min_coverage] = np.nan
    
    # Calculate per-sample coverage statistics
    coverage_stats = {
        'per_sample_median_coverage': control_cov.median(axis=0),
        'per_sample_mean_coverage': control_cov.mean(axis=0),
        'high_coverage_samples': (control_cov.median(axis=0) >= 10).sum(),
        'total_samples': control_cov.shape[1]
    }
    
    print(f"High coverage samples (≥10): {coverage_stats['high_coverage_samples']}")
    print(f"Total samples: {coverage_stats['total_samples']}")
    
    return control_mv_filtered, coverage_stats

def calculate_control_metrics_vectorized(control_mv_filtered):
    """
    Vectorized calculation of control performance metrics
    """
    print("Calculating control metrics for all regions...")
    
    # Basic statistics (vectorized)
    n_valid_samples = (~control_mv_filtered.isna()).sum(axis=1)
    mean_signal = control_mv_filtered.mean(axis=1, skipna=True)
    max_signal = control_mv_filtered.max(axis=1, skipna=True)
    median_signal = control_mv_filtered.median(axis=1, skipna=True)
    std_signal = control_mv_filtered.std(axis=1, skipna=True)
    
    # Contamination metrics
    n_with_signal = (control_mv_filtered > 0.001).sum(axis=1)  # >0.1%
    n_high_signal = (control_mv_filtered > 0.01).sum(axis=1)   # >1%
    pct_with_signal = n_with_signal / n_valid_samples * 100
    
    # Handle edge cases
    mean_signal = mean_signal.fillna(0)
    max_signal = max_signal.fillna(0)
    median_signal = median_signal.fillna(0)
    std_signal = std_signal.fillna(0)
    pct_with_signal = pct_with_signal.fillna(0)
    
    metrics = pd.DataFrame({
        'n_valid_samples': n_valid_samples,
        'mean_control_signal': mean_signal,
        'max_control_signal': max_signal,
        'median_control_signal': median_signal,
        'std_control_signal': std_signal,
        'n_with_signal': n_with_signal,
        'n_high_signal': n_high_signal,
        'pct_with_signal': pct_with_signal
    }, index=control_mv_filtered.index)
    
    return metrics

def apply_strict_control_filters(df_regions, control_metrics, min_samples=3):
    """
    Apply strict control-based filtering criteria
    """
    print("\n" + "="*60)
    print("APPLYING STRICT CONTROL FILTERS")
    print("="*60)
    
    print(f"Starting with {len(df_regions):,} candidate regions")
    
    # Merge region info with control metrics
    merged = df_regions.reset_index(drop=True).join(control_metrics, how='inner')
    print(f"After alignment: {len(merged):,} regions")
    
    # Apply strict filters step by step
    filters = [
        ('sufficient_samples', lambda df: df['n_valid_samples'] >= min_samples),
        ('ultra_low_mean', lambda df: df['mean_control_signal'] <= 0.002),  # ≤0.2% mean
        ('low_max_signal', lambda df: df['max_control_signal'] <= 0.01),    # ≤1% max
        ('minimal_contamination', lambda df: df['pct_with_signal'] <= 5.0),  # ≤5% samples with signal
        ('no_high_signal', lambda df: df['n_high_signal'] == 0)             # No samples >1%
    ]
    
    filtered = merged.copy()
    
    for filter_name, filter_func in filters:
        before_count = len(filtered)
        filtered = filtered[filter_func(filtered)]
        after_count = len(filtered)
        
        print(f"{filter_name}: {before_count:,} → {after_count:,} regions "
              f"({100*after_count/before_count:.1f}% survival)")
        
        if after_count == 0:
            print("❌ No regions survive this filter!")
            break
    
    return filtered

def calculate_enhanced_quality_scores(filtered_regions):
    """
    Calculate comprehensive quality scores for ranking
    """
    print("\nCalculating quality scores...")
    
    # Component scores (0-1 scale)
    scores = {}
    
    # 1. Cancer signal strength (higher is better)
    scores['cancer_strength'] = np.clip((filtered_regions['OAC'] - 0.5) / 0.5, 0, 1)
    
    # 2. Control cleanliness (lower control signal is better)
    scores['control_clean'] = np.clip(1 - (filtered_regions['mean_control_signal'] / 0.002), 0, 1)
    
    # 3. Signal separation (cancer vs control ratio)
    separation_ratio = filtered_regions['OAC'] / (filtered_regions['mean_control_signal'] + 0.0001)
    scores['separation'] = np.clip(np.log10(separation_ratio) / 3, 0, 1)  # Log scale, cap at 1000x
    
    # 4. Sample robustness (more samples = better)
    scores['sample_robust'] = np.clip(filtered_regions['n_valid_samples'] / 10, 0, 1)
    
    # 5. Consistency (low std relative to mean)
    rel_std = filtered_regions['std_control_signal'] / (filtered_regions['mean_control_signal'] + 0.0001)
    scores['consistency'] = np.clip(1 - (rel_std / 2), 0, 1)
    
    # 6. Technical quality (coverage and CpG count if available)
    if 'OAC_coverage' in filtered_regions.columns and 'n_cpgs' in filtered_regions.columns:
        scores['technical'] = np.clip(
            (filtered_regions['OAC_coverage'] / 100) * 0.7 + 
            (np.clip(filtered_regions['n_cpgs'] - 5, 0, 15) / 15) * 0.3, 
            0, 1
        )
    elif 'OAC_coverage' in filtered_regions.columns:
        scores['technical'] = np.clip(filtered_regions['OAC_coverage'] / 100, 0, 1)
    else:
        scores['technical'] = np.full(len(filtered_regions), 0.5)  # Neutral score
    
    # Composite score with weights optimized for clinical deployment
    composite_score = (
        0.25 * scores['control_clean'] +      # Highest weight: specificity critical
        0.20 * scores['cancer_strength'] +    # Cancer signal matters
        0.20 * scores['separation'] +         # Good signal/noise ratio
        0.15 * scores['sample_robust'] +      # Reliable across samples
        0.10 * scores['consistency'] +        # Reproducible measurements
        0.10 * scores['technical']            # Technical quality
    )
    
    # Add individual scores to dataframe
    for score_name, score_values in scores.items():
        filtered_regions[f'score_{score_name}'] = score_values
    
    filtered_regions['composite_quality_score'] = composite_score
    
    return filtered_regions

def select_final_marker_panel(scored_regions, target_markers=5000):
    """
    Select final marker panel with diversity considerations
    """
    print(f"\nSelecting final panel of {target_markers:,} markers...")
    
    # Sort by quality score
    ranked = scored_regions.sort_values('composite_quality_score', ascending=False)
    
    if len(ranked) <= target_markers:
        print(f"Only {len(ranked):,} regions available - using all")
        return ranked
    
    # Simple selection: take top N by quality
    selected = ranked.head(target_markers).copy()
    
    # Add selection metadata
    selected['selection_rank'] = range(1, len(selected) + 1)
    selected['selection_percentile'] = (selected['selection_rank'] / len(ranked)) * 100
    
    print(f"Selected {len(selected):,} markers")
    print(f"Quality score range: {selected['composite_quality_score'].min():.3f} - "
          f"{selected['composite_quality_score'].max():.3f}")
    print(f"Cancer signal range: {selected['OAC'].min():.3f} - {selected['OAC'].max():.3f}")
    print(f"Control signal range: {selected['mean_control_signal'].min():.4f} - "
          f"{selected['mean_control_signal'].max():.4f}")
    
    return selected

def run_enhanced_stage2_filtering(df_regions, control_mv, control_cov, 
                                target_markers=5000, min_coverage=5):
    """
    Complete enhanced Stage 2 filtering pipeline
    """
    print("="*60)
    print("ENHANCED STAGE 2: SCALABLE CONTROL-BASED FILTERING")
    print("="*60)
    
    # Step 1: Prepare control data
    control_mv_filtered, coverage_stats = prepare_control_data_efficient(
        control_mv, control_cov, min_coverage
    )
    
    # Step 2: Calculate control metrics (vectorized for speed)
    control_metrics = calculate_control_metrics_vectorized(control_mv_filtered)
    
    # Step 3: Apply strict control filters
    filtered_regions = apply_strict_control_filters(df_regions, control_metrics)
    
    if len(filtered_regions) == 0:
        print("❌ No regions passed strict control filtering!")
        return None, None
    
    # Step 4: Calculate quality scores
    scored_regions = calculate_enhanced_quality_scores(filtered_regions)
    
    # Step 5: Select final panel
    final_panel = select_final_marker_panel(scored_regions, target_markers)
    
    # Summary statistics
    print("\n" + "="*60)
    print("FINAL PANEL SUMMARY")
    print("="*60)
    print(f"Input regions: {len(df_regions):,}")
    print(f"After control filtering: {len(filtered_regions):,}")
    print(f"Final panel size: {len(final_panel):,}")
    print(f"Overall survival rate: {len(final_panel)/len(df_regions)*100:.2f}%")
    
    return final_panel, {
        'control_metrics': control_metrics,
        'filtered_regions': filtered_regions,
        'coverage_stats': coverage_stats
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced second-line filtering for scalable marker selection')
    parser.add_argument('--oac_first_line_atlas', type=str, required=True, help='file name for oac first line filtered atlas')
    parser.add_argument('--control_dir', type=str, required=True, help='Directory containing control parquet file for marker values and coverage')
    parser.add_argument('--out_atlas_name', type=str, required=True, help='Atlas name for the filtered output')
    parser.add_argument('--target_markers', type=int, default=5000, help='Target number of markers for final panel')
    args = parser.parse_args()

    print("Loading input data...")
    broad_filtered_df = pd.read_csv(args.oac_first_line_atlas, sep="\t")
    broad_filtered_df = broad_filtered_df.drop_duplicates(["name"])
    
    # Load control data
    control_mv = pd.read_parquet(f"{args.control_dir}/marker_values.parquet")
    control_cov = pd.read_parquet(f"{args.control_dir}/coverage.parquet")
    
    # Skip first 8 columns (metadata) and use only sample columns
    print(f"Control data shape: MV {control_mv.shape}, COV {control_cov.shape}")
    control_mv_samples = control_mv[control_mv.columns[8:]]
    control_cov_samples = control_cov[control_cov.columns[8:]]

    # Run enhanced filtering pipeline
    final_panel, metadata = run_enhanced_stage2_filtering(
        broad_filtered_df, control_mv_samples, control_cov_samples, 
        target_markers=10000, min_coverage=5
    )

    if final_panel is not None:
        # Map back to original dataframe using index alignment
        # The enhanced filtering preserves the original index, so we can use it directly
        if 'name' in final_panel.columns:
            selected_names = set(final_panel['name'].unique())
        else:
            # Use index-based mapping if name column isn't available
            selected_indices = final_panel.index.tolist()
            selected_names = set(broad_filtered_df.iloc[selected_indices]['name'])
            
        final_markers = broad_filtered_df[broad_filtered_df['name'].isin(selected_names)]

        print(f"\n==== FINAL RESULTS ====")
        print(f"Selected {len(final_markers)} markers for final atlas")
        print(f"Saved to: {args.out_atlas_name}")

        # Save the final atlas with all original columns
        final_markers[
            [
                "chr", "start", "end", "startCpG", "endCpG", "target", "name", "direction",
                "B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", "Colon", 
                "Esophagus", "Gastric", "Granulocytes", "Monocytes", "NK-cells", 
                "OAC", "Small-intestine", "T-cells"
            ]
        ].to_csv(args.out_atlas_name, sep="\t", index=False)
        
        # Save detailed results with quality scores
        if len(final_panel) > 0:
            quality_file = args.out_atlas_name.replace('.tsv', '_quality_scores.tsv')
            final_panel.to_csv(quality_file, sep="\t", index=False)
            print(f"Quality scores saved to: {quality_file}")
    else:
        print("❌ No markers survived the filtering process!")
        # Create empty output file
        pd.DataFrame(columns=["chr", "start", "end", "name"]).to_csv(args.out_atlas_name, sep="\t", index=False)

if __name__ == "__main__":
    main()
