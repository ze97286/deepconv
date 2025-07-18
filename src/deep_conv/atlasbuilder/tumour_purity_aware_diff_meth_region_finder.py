import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import glob
import re

def filter_by_coverage(mv, cov, min_coverage=10):
      """
      Filter marker values and coverage dataframes using mixed coverage thresholds
      - High coverage controls (TP/X samples): >= 5 reads
      - Low coverage controls (GI samples): >= 3 reads
      - Tumor samples: >= min_coverage reads
      
      Args:
          mv: marker values dataframe (regions x samples)
          cov: coverage dataframe (regions x samples) 
          min_coverage: minimum coverage threshold for tumor samples
          
      Returns:
          filtered_mv, filtered_cov: filtered dataframes
      """
      # Get sample columns (assuming first few columns are metadata like 'name', 'direction')
      sample_cols = [col for col in cov.columns if col not in ['name', 'direction']]
      
      # Separate control and tumor samples
      # For control files: GI samples = low coverage, everything else = high coverage
      # For tumor files: no controls present
      low_cov_controls = [col for col in sample_cols if 'GI' in col]
      high_cov_controls = [col for col in sample_cols if 'GI' not in col and ('TP' in col or 'X' in col)]
      tumor_samples = [col for col in sample_cols if 'GI' not in col and 'TP' not in col and 'X' not in col]
      
      print(f"Sample classification:")
      print(f"  High coverage controls: {len(high_cov_controls)}")
      print(f"  Low coverage controls: {len(low_cov_controls)}")
      print(f"  Tumor samples: {len(tumor_samples)}")
      
      # Create coverage masks
      if high_cov_controls:
          high_cov_mask = (cov[high_cov_controls] >= 5).all(axis=1)
      else:
          high_cov_mask = True
      
      if low_cov_controls:
          low_cov_mask = (cov[low_cov_controls] >= 3).all(axis=1)
      else:
          low_cov_mask = True
      
      if tumor_samples:
          tumor_mask = (cov[tumor_samples] >= min_coverage).all(axis=1)
      else:
          tumor_mask = True
      
      # Combined mask
      coverage_mask = high_cov_mask & low_cov_mask & tumor_mask
      
      # Apply filter to both dataframes
      filtered_mv = mv[coverage_mask].copy()
      filtered_cov = cov[coverage_mask].copy()
      
      print(f"Original regions: {len(mv)}")
      print(f"After mixed coverage filter: {len(filtered_mv)}")
      print(f"Kept {len(filtered_mv)/len(mv)*100:.1f}% of regions")
      
      return filtered_mv, filtered_cov

def analyse_tumour_purity_correlation(filtered_mv, tumor_purity_dict, min_correlation=0.7,
                                    max_control_signal=0.01, check_controls=True):
      """
      Find regions where methylation signal correlates with tumor purity
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
      # Calculate correlation for each region
      correlations = []
      pvalues = []
      valid_counts = []
      print("Processing regions...")
      for i in range(len(filtered_mv)):
          if i % 100000 == 0:
              print(f"Processed {i}/{len(filtered_mv)} regions")
          signals = numeric_data[i, :]  # Get row i
          # Check for valid values
          valid_mask = ~np.isnan(signals) & np.isfinite(signals)
          valid_count = np.sum(valid_mask)
          valid_counts.append(valid_count)
          # Skip if too few valid values
          if valid_count < 3:
              correlations.append(np.nan)
              pvalues.append(np.nan)
              continue
          # Use only valid values for correlation
          valid_signals = signals[valid_mask]
          valid_purities = purities[valid_mask]
          # Check for constant values (correlation undefined)
          if np.std(valid_signals) == 0 or np.std(valid_purities) == 0:
              correlations.append(np.nan)
              pvalues.append(np.nan)
              continue
          # Calculate Pearson correlation
          try:
              r, p = stats.pearsonr(valid_purities, valid_signals)
              correlations.append(r)
              pvalues.append(p)
          except Exception as e:
              print(f"Error at region {i}: {e}")
              correlations.append(np.nan)
              pvalues.append(np.nan)
      # Create results dataframe
      results = pd.DataFrame({
          'region': filtered_mv.index,
          'name': filtered_mv['name'],
          'correlation': correlations,
          'pvalue': pvalues,
          'valid_samples': valid_counts,
          'significant': (np.array(pvalues) < 0.05) & (np.array(correlations) > min_correlation)  # Only positive correlations!
      })
      # Add mean signal across samples
      results['mean_signal'] = np.nanmean(numeric_data, axis=1)
      
      # Add control filtering if requested
      if check_controls and control_cols:
          print(f"\nApplying control filtering (max signal <= {max_control_signal})...")
          control_data = filtered_mv[control_cols].values
          print(f"  Control data shape: {control_data.shape}")
          
          # Debug: Check control data
          print(f"  Control data range: {np.nanmin(control_data):.4f} - {np.nanmax(control_data):.4f}")
          
          control_max = np.nanmax(control_data, axis=1)
          results['control_max'] = control_max
          
          # Debug: Show some examples
          print(f"\n  Examples of control max values:")
          for i in range(min(5, len(control_max))):
              print(f"    Region {i}: control_max = {control_max[i]:.4f}")
          
          # Update significant regions to include control filter
          control_filter = control_max <= max_control_signal
          results['significant'] = results['significant'] & control_filter
          
          # Store the correlation-only results before applying control filter  
          correlation_only = (np.array(pvalues) < 0.05) & (np.array(correlations) > min_correlation)
          print(f"Regions passing tumor-purity correlation: {correlation_only.sum()}")
          print(f"Regions passing control filter: {control_filter.sum()}")
          print(f"Regions passing BOTH filters: {results['significant'].sum()}")
          
          # Show control signal distribution
          print(f"\nControl signal distribution:")
          print(f"  Min: {np.nanmin(control_max):.4f}")
          print(f"  25th percentile: {np.nanpercentile(control_max, 25):.4f}")
          print(f"  50th percentile: {np.nanpercentile(control_max, 50):.4f}")
          print(f"  75th percentile: {np.nanpercentile(control_max, 75):.4f}")
          print(f"  Max: {np.nanmax(control_max):.4f}")
          
          # Show what we'd get with different thresholds
          print(f"\nRegions passing different control thresholds:")
          for thresh in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
              passing = (control_max <= thresh).sum()
              print(f"  <= {thresh}: {passing} regions ({100*passing/len(control_max):.1f}%)")
          
      # Sort by correlation (descending) - we want positive correlations at the top
      results = results.reindex(results['correlation'].sort_values(ascending=False).index)
      print(f"\nResults:")
      print(f"Regions with correlation > {min_correlation}: {((np.array(pvalues) < 0.05) & (np.array(correlations) > min_correlation)).sum()}")
      print(f"Regions with valid data: {(results['valid_samples'] >= 3).sum()}")
      print(f"Final significant regions (tumor-specific): {results['significant'].sum()}")
      
      # Report on negative correlations (biological nonsense for tumor markers)
      negative_high_corr = (np.array(pvalues) < 0.05) & (np.array(correlations) < -min_correlation)
      print(f"Regions with correlation < -{min_correlation} (negative - excluded): {negative_high_corr.sum()}")
      
      print(f"Top positive correlations: {results['correlation'].head(10).values}")
      print(f"Top negative correlations: {results['correlation'].tail(10).values}")
      return results, sample_cols, purities

def plot_top_correlations(filtered_mv, results, sample_cols, purities, output_dir, min_cpgs, n_plots=30):
    """Plot signal vs tumor purity for top positively correlated regions"""
    fig, axes = plt.subplots(10, 3, figsize=(15, 10))
    fig.suptitle('Top Tumor-Specific Regions (Positive Correlations Only)', fontsize=16)
    axes = axes.flatten()
    # Get top significant regions (now only positive correlations)
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
        ax.set_title(f"{region['name']}\nr={region['correlation']:.3f}, p={region['pvalue']:.1e}")
        # Add trend line if we have enough points
        if len(valid_signals) >= 2:
            try:
                z = np.polyfit(valid_purities, valid_signals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(valid_purities), max(valid_purities), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)
            except:
                print(f"Could not fit trend line for {region['name']}")
    # Hide empty subplots
    for idx in range(len(top_regions), n_plots):
        axes[idx].set_visible(False)
    plt.tight_layout()
    output_file = f'{output_dir}/l{min_cpgs}_tumour_purity_correlations.png'
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
    parser.add_argument('--control_dir', required=True, help='Directory containing control pat files')
    parser.add_argument("--output_atlas_path", required=True, help="Path to save atlas")
    parser.add_argument('--min_correlation', type=float, default=0.7, help='Minimum correlation with tumor purity')
    parser.add_argument('--max_control_signal', type=float, default=0.01, help='Maximum signal allowed in control samples')
    parser.add_argument('--no_control_filter', action='store_true', help='Skip control filtering (not recommended)')

    args = parser.parse_args()

    # Process tumor samples
    for i in range(1,23):
        mv = pd.read_parquet(f"{args.pat_dir}/l{args.min_cpgs}_chr{i}_marker_values.parquet")
        cov = pd.read_parquet(f"{args.pat_dir}/l{args.min_cpgs}_chr{i}_coverage.parquet")
        filtered_mv, filtered_cov = filter_by_coverage(mv, cov, min_coverage=10)
        filtered_mv.to_parquet(f"{args.pat_dir}/l{args.min_cpgs}_chr{i}_filtered_marker_values.parquet", index=False)
        filtered_cov.to_parquet(f"{args.pat_dir}/l{args.min_cpgs}_chr{i}_filtered_coverage.parquet", index=False)

    # Load tumor data
    tumor_files = glob.glob(f"{args.pat_dir}/*filtered_marker_values.parquet")
    tumor_cov_files = glob.glob(f"{args.pat_dir}/*filtered_coverage.parquet")
    print(f"Loading {len(tumor_files)} tumor marker files...")
    print(f"Loading {len(tumor_cov_files)} tumor coverage files...")
    
    tumor_mv = pd.read_parquet(tumor_files)
    tumor_cov = pd.read_parquet(tumor_cov_files)
    
    # Process control samples
    print("Processing control samples...")
    for i in range(1,23):
        control_mv = pd.read_parquet(f"{args.control_dir}/l{args.min_cpgs}_chr{i}_marker_values.parquet")
        control_cov = pd.read_parquet(f"{args.control_dir}/l{args.min_cpgs}_chr{i}_coverage.parquet")
        filtered_control_mv, filtered_control_cov = filter_by_coverage(control_mv, control_cov, min_coverage=10)
        filtered_control_mv.to_parquet(f"{args.control_dir}/l{args.min_cpgs}_chr{i}_filtered_marker_values.parquet", index=False)
        filtered_control_cov.to_parquet(f"{args.control_dir}/l{args.min_cpgs}_chr{i}_filtered_coverage.parquet", index=False)

    # Load control data
    control_files = glob.glob(f"{args.control_dir}/*filtered_marker_values.parquet")
    control_cov_files = glob.glob(f"{args.control_dir}/*filtered_coverage.parquet")
    print(f"Loading {len(control_files)} control marker files...")
    print(f"Loading {len(control_cov_files)} control coverage files...")
    
    control_mv = pd.read_parquet(control_files)
    control_cov = pd.read_parquet(control_cov_files)
    
    # Merge tumor and control data
    print("Merging tumor and control data...")
    # Since alignment is confirmed to be correct, use indices directly
    common_regions = tumor_mv.index.intersection(control_mv.index)
    print(f"Found {len(common_regions)} common regions between tumor and control samples")
    
    # Create combined dataset
    filtered_mv = tumor_mv.loc[common_regions].copy()
    filtered_cov = tumor_cov.loc[common_regions].copy()
    
    # Add control columns
    print("\nAdding control columns...")
    control_sample_cols = [col for col in control_mv.columns if col not in ['name', 'direction']]
    print(f"Control sample columns to add: {len(control_sample_cols)}")
    if len(control_sample_cols) > 0:
        print(f"First 5 control columns: {control_sample_cols[:5]}")
    
    for col in control_sample_cols:
        filtered_mv[f"Control_{col}"] = control_mv.loc[common_regions, col].values
        filtered_cov[f"Control_{col}"] = control_cov.loc[common_regions, col].values
    
    # Debug: Check if control values were added correctly
    control_cols_added = [col for col in filtered_mv.columns if col.startswith('Control_')]
    if control_cols_added:
        print(f"\nDebug - First control column values:")
        first_control = control_cols_added[0]
        print(f"  Column: {first_control}")
        print(f"  First 5 values: {filtered_mv[first_control].iloc[:5].values}")
        print(f"  Value range: {filtered_mv[first_control].min():.4f} - {filtered_mv[first_control].max():.4f}")
    
    print(f"Combined dataset shape: {filtered_mv.shape}")
    print(f"Control columns added: {len([col for col in filtered_mv.columns if col.startswith('Control_')])}")

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

    results, sample_cols, purities = analyse_tumour_purity_correlation(
        filtered_mv,
        tumor_purity_dict,
        min_correlation=args.min_correlation,
        max_control_signal=args.max_control_signal,
        check_controls=not args.no_control_filter
    )

    plot_top_correlations(filtered_mv, results, sample_cols, purities, args.pat_dir, args.min_cpgs)

    # Save results
    results.to_csv(f'{args.pat_dir}/l{args.min_cpgs}_tumor_purity_correlations.csv', index=False)

    # Extract highly correlated regions for next stage
    tumor_specific_regions = results[results['significant']]['name'].values
    print(f"\nFound {len(tumor_specific_regions)} tumor-specific regions")

    create_tumour_atlas_from_results(
        results,  
        min_cpgs=args.min_cpgs,
        output_atlas_path=args.output_atlas_path,
    )
    
if __name__ == '__main__':
    main()