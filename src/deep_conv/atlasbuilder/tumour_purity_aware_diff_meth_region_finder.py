import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import glob
import re

def filter_by_coverage(mv, cov, min_coverage=10):
      """
      Filter marker values and coverage dataframes to keep only rows 
      where all samples have coverage > min_coverage
      
      Args:
          mv: marker values dataframe (regions x samples)
          cov: coverage dataframe (regions x samples) 
          min_coverage: minimum coverage threshold
          
      Returns:
          filtered_mv, filtered_cov: filtered dataframes
      """
      # Get sample columns (assuming first few columns are metadata like 'name', 'direction')
      sample_cols = [col for col in cov.columns if col not in ['name', 'direction']]
      # Create mask: True for rows where ALL samples have coverage > min_coverage
      coverage_mask = (cov[sample_cols] > min_coverage).all(axis=1)
      # Apply filter to both dataframes
      filtered_mv = mv[coverage_mask].copy()
      filtered_cov = cov[coverage_mask].copy()
      print(f"Original regions: {len(mv)}")
      print(f"After coverage filter (all samples > {min_coverage}): {len(filtered_mv)}")
      print(f"Kept {len(filtered_mv)/len(mv)*100:.1f}% of regions")
      return filtered_mv, filtered_cov

def analyse_tumour_purity_correlation(filtered_mv, tumor_purity_dict, min_correlation=0.7):
      """
      Find regions where methylation signal correlates with tumor purity
      """
      # Get sample columns that have purity info
      sample_cols = [col for col in filtered_mv.columns if col in tumor_purity_dict]
      purities = np.array([tumor_purity_dict[col] for col in sample_cols])
      print(f"Analyzing {len(sample_cols)} samples with purity data")
      print(f"Tumor purities: {purities}")
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
          'significant': (np.array(pvalues) < 0.05) & (np.abs(np.array(correlations)) > min_correlation)
      })
      # Add mean signal across samples
      results['mean_signal'] = np.nanmean(numeric_data, axis=1)
      # Sort by absolute correlation (descending)
      results = results.reindex(results['correlation'].abs().sort_values(ascending=False).index)
      print(f"\nResults:")
      print(f"Regions with |correlation| > {min_correlation}: {results['significant'].sum()}")
      print(f"Regions with valid data: {(results['valid_samples'] >= 3).sum()}")
      print(f"Top correlations: {results['correlation'].head(10).values}")
      return results, sample_cols, purities

def plot_top_correlations(filtered_mv, results, sample_cols, purities, n_plots=6):
    """Plot signal vs tumor purity for top correlated regions"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    # Get top significant regions
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
    plt.savefig('/users/zetzioni/sharedscratch/loyfer_atlas/cna_corrected_pats/tumour_purity_correlations.png', dpi=300, bbox_inches='tight')

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

# python -m deep_conv.atlasbuilder.tumour_purity_aware_diff_meth_region_finder \
# --min_cpgs 3 \
# --pat_dir /users/zetzioni/sharedscratch/loyfer_atlas/cna_corrected_pats \
# --output_atlas_path /users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_tumour_content_correlated_regions_l3.bed
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process pat files for UXM analysis')
    parser.add_argument('--min_cpgs', type=int, required=True, help='Minimum CpGs required')
    parser.add_argument('--pat_dir', required=True, help='Directory containing pat files')
    parser.add_argument("--output_atlas_path", required=True, help="Path to save atlas")

    args = parser.parse_args()

    for i in range(1,23):
        mv = pd.read_parquet(f"{args.pat_dir}/chr{i}_marker_values.parquet")
        cov = pd.read_parquet(f"{args.pat_dir}/chr{i}_coverage.parquet")
        filtered_mv, filtered_cov = filter_by_coverage(mv, cov, min_coverage=10)
        filtered_mv.to_parquet(f"{args.pat_dir}/chr{i}_filtered_marker_values.parquet", index=False)
        filtered_cov.to_parquet(f"{args.pat_dir}/chr{i}_filtered_coverage.parquet", index=False)

    filtered_mv = pd.read_parquet(glob.glob(f"{args.pat_dir}/*filtered_marker_values.parquet"))
    filtered_cov = pd.read_parquet(glob.glob(f"{args.pat_dir}/*filtered_coverage.parquet"))

    tumor_purity_dict = {
        'OAC_069-009_ScrBsl_tumour_cna_corrected':0.5171,
        'OAC_071-011_ScrBsl_tumour_cna_corrected':0.2361,
        'OAC_071-014_ScrBsl_tumour_cna_corrected':0.07926,
        'OAC_071-021_ScrBsl_tumour_cna_corrected':0.4766,
        'OAC_071-022_ScrBsl_tumour_cna_corrected':0.4108,
        'OAC_071-030_ScrBsl_tumour_cna_corrected':0.0801,
        'OAC_071-043_ScrBsl_tumour_cna_corrected':0.4607,
        'OAC_129-001_ScrBsl_tumour_cna_corrected':0.6921
    }

    results, sample_cols, purities = analyse_tumour_purity_correlation(
        filtered_mv,
        tumor_purity_dict,
        min_correlation=0.7
    )

    plot_top_correlations(filtered_mv, results, sample_cols, purities)

    # Save results
    results.to_csv(f'{args.pat_dir}/tumor_purity_correlations.csv', index=False)

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