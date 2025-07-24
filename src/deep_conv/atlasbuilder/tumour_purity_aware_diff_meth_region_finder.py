import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import glob
import re

def apply_mixed_coverage_filter(mv_data, cov_data, tumor_min_cov=10, xtp_control_min_cov=10, gi_control_min_cov=5, control_quorum=0.6):
    """
    Apply mixed coverage filtering based on sample types
    Uses stricter coverage for X### and TP### controls, more lenient for GI controls
    Uses quorum approach for controls (60% must have sufficient coverage)
    """
    all_sample_cols = [col for col in cov_data.columns if col not in ['name', 'direction']]
    
    # Classify samples more specifically
    # X### and TP### controls need higher coverage (10 reads)
    xtp_controls = [col for col in all_sample_cols if col.startswith('Control_') and 
                   (col.startswith('Control_X') or col.startswith('Control_TP'))]
    # GI controls need lower coverage (5 reads)  
    gi_controls = [col for col in all_sample_cols if col.startswith('Control_') and 'GI' in col]
    # Any other controls (fallback)
    other_controls = [col for col in all_sample_cols if col.startswith('Control_') and 
                     col not in xtp_controls and col not in gi_controls]
    tumor_samples = [col for col in all_sample_cols if not col.startswith('Control_')]
    
    print(f"  Sample classification:")
    print(f"    X###/TP### controls (high coverage): {len(xtp_controls)}")
    print(f"    GI controls (low coverage): {len(gi_controls)}")
    print(f"    Other controls: {len(other_controls)}")
    print(f"    Tumor samples: {len(tumor_samples)}")
    
    # Create coverage masks - require ALL tumor samples but only quorum of controls
    tumor_mask = (cov_data[tumor_samples] >= tumor_min_cov).all(axis=1) if tumor_samples else pd.Series(True, index=cov_data.index)
    
    # For controls, use quorum approach (60% must have sufficient coverage)
    if xtp_controls:
        xtp_control_sufficient = (cov_data[xtp_controls] >= xtp_control_min_cov).sum(axis=1)
        xtp_control_mask = xtp_control_sufficient >= (len(xtp_controls) * control_quorum)
    else:
        xtp_control_mask = pd.Series(True, index=cov_data.index)
    
    if gi_controls:
        gi_control_sufficient = (cov_data[gi_controls] >= gi_control_min_cov).sum(axis=1)
        gi_control_mask = gi_control_sufficient >= (len(gi_controls) * control_quorum)
    else:
        gi_control_mask = pd.Series(True, index=cov_data.index)
        
    if other_controls:
        other_control_sufficient = (cov_data[other_controls] >= gi_control_min_cov).sum(axis=1)  # Use GI threshold for other controls
        other_control_mask = other_control_sufficient >= (len(other_controls) * control_quorum)
    else:
        other_control_mask = pd.Series(True, index=cov_data.index)
    
    # Combined mask
    combined_mask = tumor_mask & xtp_control_mask & gi_control_mask & other_control_mask
    
    print(f"  Coverage filtering results (quorum = {control_quorum:.0%}):")
    print(f"    Tumor ≥{tumor_min_cov} (all): {tumor_mask.sum()}/{len(tumor_mask)} ({100*tumor_mask.sum()/len(tumor_mask):.1f}%)")
    print(f"    X###/TP### controls ≥{xtp_control_min_cov} (≥{control_quorum:.0%}): {xtp_control_mask.sum()}/{len(xtp_control_mask)} ({100*xtp_control_mask.sum()/len(xtp_control_mask):.1f}%)")
    print(f"    GI controls ≥{gi_control_min_cov} (≥{control_quorum:.0%}): {gi_control_mask.sum()}/{len(gi_control_mask)} ({100*gi_control_mask.sum()/len(gi_control_mask):.1f}%)")
    if other_controls:
        print(f"    Other controls ≥{gi_control_min_cov} (≥{control_quorum:.0%}): {other_control_mask.sum()}/{len(other_control_mask)} ({100*other_control_mask.sum()/len(other_control_mask):.1f}%)")
    print(f"    Combined: {combined_mask.sum()}/{len(combined_mask)} ({100*combined_mask.sum()/len(combined_mask):.1f}%)")
    
    return mv_data[combined_mask].copy(), cov_data[combined_mask].copy()

def analyse_tumour_purity_correlation(filtered_mv, tumor_purity_dict, min_correlation=0.8,
                                    max_control_signal=0.01, check_controls=True, step1_only=False):
      """
      Find regions where methylation signal has a strong LINEAR relationship with tumor purity
      using proper linear regression validation (not just correlation)
      AND have low signal in control samples (tumor-specific)
      """
      # Get sample columns that have purity info
      sample_cols = [col for col in filtered_mv.columns if col in tumor_purity_dict]
      purities = np.array([tumor_purity_dict[col] for col in sample_cols])
      print(f"Analyzing {len(sample_cols)} samples with purity data")
      print(f"Tumor purities: {purities}")
      
      # Get control columns if checking controls
      control_cols = []
      if check_controls:
          control_cols = [col for col in filtered_mv.columns if col.startswith('Control_')]
          print(f"Found {len(control_cols)} control samples for specificity check")
      # Pre-extract numeric data for all samples
      numeric_data = filtered_mv[sample_cols].values  # This should be clean float64
      
      # Vectorized linear regression using numpy - MUCH faster than sklearn
      print("Processing regions with vectorized linear regression...")
      
      # Pre-compute constants for all regions
      n = len(purities)
      sum_x = np.sum(purities)
      sum_x2 = np.sum(purities**2)
      mean_x = np.mean(purities)
      
      # Check if all samples have valid data (vectorized)
      valid_counts = (~np.isnan(numeric_data)).sum(axis=1)
      all_valid_mask = valid_counts == n
      
      # Initialize result arrays
      num_regions = len(filtered_mv)
      r2_scores = np.full(num_regions, np.nan)
      slopes = np.full(num_regions, np.nan)
      intercepts = np.full(num_regions, np.nan)
      rmses = np.full(num_regions, np.nan)
      
      # Process only regions with all valid samples
      valid_indices = np.where(all_valid_mask)[0]
      print(f"Found {len(valid_indices)} regions with complete data out of {num_regions}")
      
      if len(valid_indices) > 0:
          # Get signals for valid regions
          valid_signals = numeric_data[valid_indices, :]
          
          # Check for non-constant signals (vectorized)
          signal_stds = np.std(valid_signals, axis=1)
          non_constant_mask = signal_stds > 0
          valid_indices = valid_indices[non_constant_mask]
          valid_signals = valid_signals[non_constant_mask, :]
          
          print(f"Processing {len(valid_indices)} regions with non-constant signals...")
          
          # Vectorized linear regression calculations
          # For each region: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - (sum(x))²)
          sum_y = np.sum(valid_signals, axis=1)
          sum_xy = np.sum(valid_signals * purities, axis=1)
          mean_y = np.mean(valid_signals, axis=1)
          
          # Calculate slopes and intercepts
          denominator = n * sum_x2 - sum_x**2
          slopes_valid = (n * sum_xy - sum_x * sum_y) / denominator
          intercepts_valid = mean_y - slopes_valid * mean_x
          
          # Calculate predictions and residuals
          predictions = slopes_valid[:, np.newaxis] * purities + intercepts_valid[:, np.newaxis]
          residuals = valid_signals - predictions
          
          # Calculate R² and RMSE
          ss_res = np.sum(residuals**2, axis=1)
          ss_tot = np.sum((valid_signals - mean_y[:, np.newaxis])**2, axis=1)
          r2_valid = 1 - (ss_res / (ss_tot + 1e-10))
          rmse_valid = np.sqrt(ss_res / n)
          
          # Store results
          r2_scores[valid_indices] = r2_valid
          slopes[valid_indices] = slopes_valid
          intercepts[valid_indices] = intercepts_valid
          rmses[valid_indices] = rmse_valid
      
      print(f"Linear regression complete!")
      
      # Report processing stats
      valid_r2 = ~np.isnan(r2_scores)
      print(f"  Regions with valid regression: {valid_r2.sum()}")
      print(f"  Regions with missing data: {(~all_valid_mask).sum()}")
      print(f"  Regions with constant signal: {all_valid_mask.sum() - valid_r2.sum()}")
      # Create results dataframe
      results = pd.DataFrame({
          'region': filtered_mv.index,
          'name': filtered_mv['name'],
          'r2_score': r2_scores,
          'slope': slopes,
          'intercept': intercepts,
          'rmse': rmses,
          'valid_samples': valid_counts
      })
      
      # Define criteria for good linear relationship
      # 1. High R² (explains variance well)
      # 2. Intercept near zero (signal should be low at 0% purity)
      # 3. Positive reasonable slope (higher purity = higher signal)
      # 4. Low RMSE relative to signal range
      
      # Add mean signal across samples
      results['mean_signal'] = np.nanmean(numeric_data, axis=1)
      
      # Calculate max signal for RMSE normalization
      max_signals = np.nanmax(numeric_data, axis=1)
      results['normalized_rmse'] = results['rmse'] / (max_signals + 1e-10)
      
      # Define significance based on multiple criteria
      results['significant'] = (
          (results['r2_score'] >= min_correlation) &  # High R²
          (np.abs(results['intercept']) <= 0.1) &     # Intercept near zero
          (results['slope'] > 0.1) &                  # Positive meaningful slope
          (results['slope'] < 2.0) &                  # Not unreasonably steep
          (results['normalized_rmse'] < 0.2)          # Low residuals relative to signal
      )
      
      # Add control filtering if requested and not in step1_only mode
      if check_controls and control_cols and not step1_only:
          print(f"\nApplying control filtering (median signal <= {max_control_signal})...")
          control_data = filtered_mv[control_cols].values
          print(f"  Control data shape: {control_data.shape}")
          
          # Debug: Check control data
          print(f"  Control data range: {np.nanmin(control_data):.4f} - {np.nanmax(control_data):.4f}")
          
          control_max = np.nanmax(control_data, axis=1)
          control_median = np.nanmedian(control_data, axis=1)
          results['control_max'] = control_max
          results['control_median'] = control_median
          
          # Debug: Show some examples
          print(f"\n  Examples of control values:")
          for i in range(min(5, len(control_max))):
              print(f"    Region {i}: median = {control_median[i]:.4f}, max = {control_max[i]:.4f}")
          
          # Update significant regions to include control filter - use median as primary filter
          control_filter = control_median <= max_control_signal
          results['significant'] = results['significant'] & control_filter
          
          # Store the linearity-only results before applying control filter  
          linearity_only = (
              (results['r2_score'] >= min_correlation) &
              (np.abs(results['intercept']) <= 0.1) &
              (results['slope'] > 0.1) &
              (results['slope'] < 2.0) &
              (results['normalized_rmse'] < 0.2)
          )
          print(f"Regions passing tumor-purity linearity: {linearity_only.sum()}")
          print(f"Regions passing control filter (median): {control_filter.sum()}")
          print(f"Regions passing BOTH filters: {results['significant'].sum()}")
          
          # Show control signal distribution
          print(f"\nControl signal distribution (MEDIAN):")
          print(f"  Min: {np.nanmin(control_median):.4f}")
          print(f"  25th percentile: {np.nanpercentile(control_median, 25):.4f}")
          print(f"  50th percentile: {np.nanpercentile(control_median, 50):.4f}")
          print(f"  75th percentile: {np.nanpercentile(control_median, 75):.4f}")
          print(f"  Max: {np.nanmax(control_median):.4f}")
          
          print(f"\nControl signal distribution (MAX):")
          print(f"  Min: {np.nanmin(control_max):.4f}")
          print(f"  25th percentile: {np.nanpercentile(control_max, 25):.4f}")
          print(f"  50th percentile: {np.nanpercentile(control_max, 50):.4f}")
          print(f"  75th percentile: {np.nanpercentile(control_max, 75):.4f}")
          print(f"  Max: {np.nanmax(control_max):.4f}")
          
          # Show what we'd get with different thresholds
          print(f"\nRegions passing different control thresholds (MEDIAN):")
          for thresh in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
              passing = (control_median <= thresh).sum()
              print(f"  <= {thresh}: {passing} regions ({100*passing/len(control_median):.1f}%)")
      elif step1_only:
          print(f"\nStep 1 mode: Skipping control filtering")
          # Add placeholder columns for consistency
          results['control_max'] = np.nan
          results['control_median'] = np.nan
          
      # Sort by R² (descending) - we want best linear fits at the top
      results = results.reindex(results['r2_score'].sort_values(ascending=False).index)
      print(f"\nResults:")
      print(f"Regions with R² ≥ {min_correlation}: {(results['r2_score'] >= min_correlation).sum()}")
      print(f"Regions with intercept near zero (|b| ≤ 0.1): {(np.abs(results['intercept']) <= 0.1).sum()}")
      print(f"Regions with positive meaningful slope (0.1 < m < 2.0): {((results['slope'] > 0.1) & (results['slope'] < 2.0)).sum()}")
      print(f"Regions with low normalized RMSE (< 0.2): {(results['normalized_rmse'] < 0.2).sum()}")
      print(f"Regions with valid data: {(results['valid_samples'] >= len(purities)).sum()}")
      print(f"Final significant regions (tumor-specific with good linearity): {results['significant'].sum()}")
      
      # Report on problematic regions
      negative_slope = results['slope'] < 0
      high_intercept = np.abs(results['intercept']) > 0.2
      print(f"\nProblematic regions:")
      print(f"  Negative slope (biologically wrong): {negative_slope.sum()}")
      print(f"  High intercept (|b| > 0.2): {high_intercept.sum()}")
      
      # Show distribution of key metrics
      valid_results = results[results['r2_score'].notna()]
      if len(valid_results) > 0:
          print(f"\nLinear regression metrics distribution:")
          print(f"  R² scores: min={valid_results['r2_score'].min():.3f}, median={valid_results['r2_score'].median():.3f}, max={valid_results['r2_score'].max():.3f}")
          print(f"  Slopes: min={valid_results['slope'].min():.3f}, median={valid_results['slope'].median():.3f}, max={valid_results['slope'].max():.3f}")
          print(f"  Intercepts: min={valid_results['intercept'].min():.3f}, median={valid_results['intercept'].median():.3f}, max={valid_results['intercept'].max():.3f}")
      return results, sample_cols, purities

def plot_top_correlations(filtered_mv, results, sample_cols, purities, output_dir, min_cpgs, n_plots=30):
    """Plot signal vs tumor purity for top regions with best linear fit"""
    fig, axes = plt.subplots(10, 3, figsize=(15, 12))
    fig.suptitle('Top Tumor-Specific Regions (Best Linear Relationships)', fontsize=16)
    axes = axes.flatten()
    # Get top significant regions (now based on R² and other criteria)
    top_regions = results[results['significant']].head(n_plots)
    if len(top_regions) == 0:
        print("No significant regions found to plot!")
        return
    for idx, (_, region) in enumerate(top_regions.iterrows()):
        if idx >= n_plots:
            break
        ax = axes[idx]
        region_idx = region['region']
        # Get signals and convert to numeric
        signals = filtered_mv.loc[region_idx, sample_cols].values
        signals = pd.to_numeric(signals, errors='coerce')
        # Remove any NaN values
        valid_mask = ~np.isnan(signals)
        valid_signals = signals[valid_mask]
        valid_purities = np.array(purities)[valid_mask]
        if len(valid_signals) < 2:
            print(f"Skipping region {region['name']} - insufficient data")
            continue
        # Plot
        ax.scatter(valid_purities, valid_signals, alpha=0.7, s=50)
        ax.set_xlabel('Tumor Purity')
        ax.set_ylabel('Methylation Signal')
        
        # Add regression line using the stored parameters
        x_range = np.linspace(0, max(valid_purities)*1.1, 100)
        y_pred = region['slope'] * x_range + region['intercept']
        ax.plot(x_range, y_pred, "r--", alpha=0.8, label=f"y = {region['slope']:.2f}x + {region['intercept']:.3f}")
        
        # Add title with R² and RMSE
        ax.set_title(f"{region['name']}\nR²={region['r2_score']:.3f}, RMSE={region['rmse']:.3f}")
        ax.legend(fontsize=8, loc='lower right')
        
        # Set y-axis to start at 0 to show intercept
        ax.set_ylim(bottom=0)
        
    # Hide empty subplots
    for idx in range(len(top_regions), n_plots):
        axes[idx].set_visible(False)
    plt.tight_layout()
    output_file = f'{output_dir}/l{min_cpgs}_tumour_purity_linear_regression.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")

def extract_chromosome_from_results(results):
      """
      Extract chromosome information from region names and group results by chromosome
      
      Args:
          results: DataFrame with correlation results containing 'name' column
      
      Returns:
          dict: {chromosome: filtered_results_df}
      """
      print("Extracting chromosome information from region names...")
      # Extract chromosome from region names (assuming format like "chr1:12345-67890")
      def extract_chr(name):
          if pd.isna(name):
              return None
          # Try different patterns
          patterns = [
              r'^chr(\w+):',           # chr1:12345-67890
              r'^(\w+):',              # 1:12345-67890  
              r'chr(\w+)_',            # chr1_12345_67890
              r'^(\w+)_'               # 1_12345_67890
          ]
          for pattern in patterns:
              match = re.search(pattern, str(name))
              if match:
                  return match.group(1)
          print(f"Warning: Could not extract chromosome from: {name}")
          return None
      # Extract chromosomes
      results['chromosome'] = results['name'].apply(extract_chr)
      # Remove rows where chromosome couldn't be extracted
      results_with_chr = results.dropna(subset=['chromosome'])
      print(f"Successfully extracted chromosome for {len(results_with_chr)}/{len(results)} regions")
      # Group by chromosome
      results_by_chromosome = {}
      for chr_name, group in results_with_chr.groupby('chromosome'):
          # Only include significant results
          significant_group = group[group['significant']]
          if len(significant_group) > 0:
              results_by_chromosome[chr_name] = significant_group
              print(f"Chromosome {chr_name}: {len(significant_group)} significant regions")
      return results_by_chromosome

def create_tumour_atlas_from_results(results, output_atlas_path, min_cpgs, regions_dir="/users/zetzioni/sharedscratch/loyfer_atlas/marker_regions"):
    """
      Create atlas file from tumor correlation results
      """
    results_by_chromosome = extract_chromosome_from_results(results)
    if not results_by_chromosome:
        print("No results found with extractable chromosome information!")
        return None
    all_atlas_regions = []
    print("\nProcessing chromosomes...")
    for chromosome, chr_results in results_by_chromosome.items():
        print(f"\nProcessing chromosome {chromosome}...")
        # Get significant region names for this chromosome
        significant_regions = chr_results['name'].values
        print(f"  Found {len(significant_regions)} significant regions")
        # Load reference regions for this chromosome
        regions_file = f"{regions_dir}/regions_chr{chromosome}_{min_cpgs}_500.bed.gz"
        try:
            reference_regions = pd.read_csv(regions_file, sep='\t', compression='gzip')
            print(f"  Loaded {len(reference_regions)} reference regions")
            # Filter to significant regions
            filtered_regions = reference_regions[
                  reference_regions['name'].isin(significant_regions)
              ].copy()
            print(f"  Matched {len(filtered_regions)} regions in reference")
            if len(filtered_regions) == 0:
                print(f"  No matches found for chromosome {chromosome}")
                continue
            # Add target column
            filtered_regions['target'] = 'OAC'
            # Ensure required columns
            required_columns = ['chr', 'start', 'end', 'startCpG', 'endCpG', 'name', 'direction', 'target']
            missing_cols = [col for col in required_columns if col not in filtered_regions.columns]
            if missing_cols:
                print(f"  Warning: Missing columns {missing_cols}")
                continue
            # Select columns in correct order
            atlas_regions = filtered_regions[required_columns].copy()
            all_atlas_regions.append(atlas_regions)
        except FileNotFoundError:
            print(f"  Warning: Could not find regions file {regions_file}")
            continue
        except Exception as e:
            print(f"  Error processing chromosome {chromosome}: {e}")
            continue
    if not all_atlas_regions:
        print("No atlas regions created!")
        return None
    # Combine all chromosomes
    print("\nCombining all chromosomes...")
    final_atlas = pd.concat(all_atlas_regions, ignore_index=True)
    # Sort by chromosome and position
    print("Sorting atlas...")
    def chromosome_sort_key(chr_str):
        """Convert chromosome to sortable format"""
        chr_clean = str(chr_str).replace('chr', '')
        if chr_clean.isdigit():
            return (0, int(chr_clean))
        elif chr_clean in ['X', 'Y']:
            return (1, ord(chr_clean))
        else:
            return (2, chr_clean)
    final_atlas['chr_sort'] = final_atlas['chr'].apply(chromosome_sort_key)
    final_atlas = final_atlas.sort_values(['chr_sort', 'start']).drop('chr_sort', axis=1)
    # Save atlas
    print(f"\nSaving atlas with {len(final_atlas)} regions to {output_atlas_path}")
    final_atlas.to_csv(output_atlas_path, sep='\t', index=False)
    # Print summary
    print(f"\nAtlas Summary:")
    print(f"Total regions: {len(final_atlas)}")
    chr_counts = final_atlas['chr'].value_counts()
    print(f"Regions by chromosome:")
    for chr_name in sorted(chr_counts.index, key=chromosome_sort_key):
        print(f"  {chr_name}: {chr_counts[chr_name]} regions")
    return final_atlas

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process pat files for UXM analysis')
    parser.add_argument('--min_cpgs', type=int, required=True, help='Minimum CpGs required')
    parser.add_argument('--pat_dir', required=True, help='Directory containing tumor pat files')
    parser.add_argument('--control_dir', help='Directory containing control pat files (not needed for --step1_only)')
    parser.add_argument("--output_atlas_path", required=True, help="Path to save atlas")
    parser.add_argument('--min_correlation', type=float, default=0.8, help='Minimum correlation with tumor purity')
    parser.add_argument('--max_control_signal', type=float, default=0.01, help='Maximum median signal allowed in control samples')
    parser.add_argument('--no_control_filter', action='store_true', help='Skip control filtering (not recommended)')
    parser.add_argument('--step1_only', action='store_true', help='Step 1 only: tumor correlation + coverage filtering (skip control filtering)')

    args = parser.parse_args()
    
    # Validate arguments
    if not args.step1_only and not args.control_dir:
        parser.error("--control_dir is required unless using --step1_only")

    tumor_purity_dict = {
        '069-009_ScrBsl_tumour':0.5171,
        '071-011_ScrBsl_tumour':0.2361,
        '071-014_ScrBsl_tumour':0.07926,
        '071-021_ScrBsl_tumour':0.4766,
        '071-022_ScrBsl_tumour':0.4108,
        '071-030_ScrBsl_tumour':0.0801,
        '071-043_ScrBsl_tumour':0.4607,
        '129-001_ScrBsl_tumour':0.6921
    }

    all_results = []
    
    # Process chromosomes one by one
    for chr_num in range(1, 23):
        print(f"\n{'='*60}")
        print(f"Processing chromosome {chr_num}")
        print(f"{'='*60}")
        
        # Load tumor data for this chromosome
        tumor_mv_file = f"{args.pat_dir}/l{args.min_cpgs}_chr{chr_num}_marker_values.parquet"
        tumor_cov_file = f"{args.pat_dir}/l{args.min_cpgs}_chr{chr_num}_coverage.parquet"
        
        try:
            tumor_mv = pd.read_parquet(tumor_mv_file)
            tumor_cov = pd.read_parquet(tumor_cov_file)
            print(f"Loaded tumor data: {tumor_mv.shape} regions")
        except FileNotFoundError:
            print(f"Tumor files not found for chr{chr_num}, skipping...")
            continue
        
        # Load control data for this chromosome (skip if step1_only)
        if not args.step1_only:
            control_mv_file = f"{args.control_dir}/l{args.min_cpgs}_chr{chr_num}_marker_values.parquet"
            control_cov_file = f"{args.control_dir}/l{args.min_cpgs}_chr{chr_num}_coverage.parquet"
            
            try:
                control_mv = pd.read_parquet(control_mv_file)
                control_cov = pd.read_parquet(control_cov_file)
                print(f"Loaded control data: {control_mv.shape} regions")
            except FileNotFoundError:
                print(f"Control files not found for chr{chr_num}, skipping...")
                continue
        else:
            print("Step 1 mode: Skipping control data loading")
            control_mv = None
            control_cov = None
        
        # Merge tumor and control data (or use tumor only for step1)
        if not args.step1_only:
            print("Merging tumor and control data...")
            common_regions = tumor_mv.index.intersection(control_mv.index)
            print(f"Found {len(common_regions)} common regions")
            
            if len(common_regions) == 0:
                print("No common regions found, skipping chromosome")
                continue
            
            # Create combined dataset
            combined_mv = tumor_mv.loc[common_regions].copy()
            combined_cov = tumor_cov.loc[common_regions].copy()
            
            # Add control columns
            control_sample_cols = [col for col in control_mv.columns if col not in ['name', 'direction']]
            print(f"Adding {len(control_sample_cols)} control columns...")
            
            for col in control_sample_cols:
                combined_mv[f"Control_{col}"] = control_mv.loc[common_regions, col].values
                combined_cov[f"Control_{col}"] = control_cov.loc[common_regions, col].values
        else:
            print("Step 1 mode: Using tumor data only")
            combined_mv = tumor_mv.copy()
            combined_cov = tumor_cov.copy()
        
        # Apply coverage filtering
        if not args.step1_only:
            print("Applying mixed coverage filtering...")
            filtered_mv, filtered_cov = apply_mixed_coverage_filter(
                combined_mv, combined_cov, 
                tumor_min_cov=10, 
                xtp_control_min_cov=10,  # X### and TP### controls need 10 reads
                gi_control_min_cov=5,    # GI controls only need 5 reads
                control_quorum=0.6
            )
        else:
            print("Applying tumor-only coverage filtering...")
            # Simple tumor coverage filter for step 1
            tumor_samples = [col for col in combined_mv.columns if not col.startswith('Control_') and col not in ['name', 'direction']]
            tumor_mask = (combined_cov[tumor_samples] >= 10).all(axis=1) if tumor_samples else pd.Series(True, index=combined_cov.index)
            print(f"  Tumor coverage ≥10 (all samples): {tumor_mask.sum()}/{len(tumor_mask)} ({100*tumor_mask.sum()/len(tumor_mask):.1f}%)")
            filtered_mv = combined_mv[tumor_mask].copy()
            filtered_cov = combined_cov[tumor_mask].copy()
        
        if len(filtered_mv) == 0:
            print("No regions passed coverage filtering, skipping chromosome")
            continue
        
        print(f"After filtering: {len(filtered_mv)} regions")
        
        # Run correlation analysis
        print("Running tumor purity correlation analysis...")
        results, sample_cols, purities = analyse_tumour_purity_correlation(
            filtered_mv,
            tumor_purity_dict,
            min_correlation=args.min_correlation,
            max_control_signal=args.max_control_signal,
            check_controls=not args.no_control_filter,
            step1_only=args.step1_only
        )
        
        # Add chromosome info
        results['chromosome'] = chr_num
        all_results.append(results)
        
        print(f"Chr {chr_num}: {results['significant'].sum()} significant regions found")
        
        # Clean up memory
        del tumor_mv, tumor_cov, control_mv, control_cov, combined_mv, combined_cov, filtered_mv, filtered_cov
    
    # Combine results from all chromosomes
    if all_results:
        print(f"\n{'='*60}")
        print("Combining results from all chromosomes...")
        print(f"{'='*60}")
        
        final_results = pd.concat(all_results, ignore_index=True)
        print(f"Total significant regions across all chromosomes: {final_results['significant'].sum()}")
        
        # Save combined results
        output_csv = f'{args.pat_dir}/l{args.min_cpgs}_tumor_purity_correlations_all_chr.csv'
        final_results.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")
        
        # Create atlas
        create_tumour_atlas_from_results(
            final_results,
            min_cpgs=args.min_cpgs,
            output_atlas_path=args.output_atlas_path,
        )
    else:
        print("No results found across any chromosomes!")
    
if __name__ == '__main__':
    main()