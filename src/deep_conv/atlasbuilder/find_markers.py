import argparse 
import glob 
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from deep_conv.deconvolution.preprocess_pats import *


def calculate_snr_significance_vectorized(values_matrix, cell_types, n_permutations=1000):
    """
    Calculate p-values for SNR using permutation test, vectorized for all rows
    
    Args:
        values_matrix: numpy array of shape (n_regions, n_cell_types)
        cell_types: list of cell type names
        n_permutations: number of permutations for the test
    
    Returns:
        p_values array of shape (n_regions, n_cell_types)
    """
    n_regions, n_cell_types = values_matrix.shape
    observed_snrs = np.zeros((n_regions, n_cell_types))
    
    # Calculate observed SNRs for each target
    for i, target in enumerate(cell_types):
        other_indices = [j for j, c in enumerate(cell_types) if c != target]
        max_background = values_matrix[:, other_indices].max(axis=1)
        observed_snrs[:, i] = values_matrix[:, i] / (max_background + 1e-10)
    
    # Permutation test
    random_snrs = np.zeros((n_regions, n_permutations, n_cell_types))
    for p in range(n_permutations):
        # Shuffle each row independently
        shuffled_values = np.array([np.random.permutation(row) for row in values_matrix])
        for i, target in enumerate(cell_types):
            other_indices = [j for j, c in enumerate(cell_types) if c != target]
            max_background = shuffled_values[:, other_indices].max(axis=1)
            random_snrs[:, p, i] = shuffled_values[:, i] / (max_background + 1e-10)
    
    # Calculate p-values
    p_values = np.mean(random_snrs >= observed_snrs[:, np.newaxis, :], axis=1)
    return p_values


   

def process_batch(args):
    """
    Process a batch of regions to evaluate markers

    Args:
        args: tuple of (batch_df, pats_path, wgbs_tools_exec_path, min_coverage, snr_threshold, significance_threshold)
    """
    temp_atlas, pats_path, wgbs_tools_exec_path, min_coverage, snr_threshold, significance_threshold = args
   
    batch_df = pd.read_csv(temp_atlas,sep="\t")
    # Get marker values and coverage
    marker_props, coverage = pats_to_homog(temp_atlas, pats_path, wgbs_tools_exec_path)

    # Clean cell type names
    col_mapping = {col.split('_')[0]: col for col in marker_props.columns 
                    if col not in ['name', 'direction']}
    cell_types = list(col_mapping.keys())
    
    # Filter out rows with NAs or zero coverage
    valid_rows = ~marker_props.iloc[:, 2:].isna().any(axis=1)
    marker_props = marker_props[valid_rows]
    coverage = coverage[valid_rows]
    batch_df = batch_df[valid_rows].reset_index(drop=True)
    
    if len(batch_df) == 0:
        return None
        
    # Filter rows with insufficient coverage
    sufficient_coverage = (coverage.iloc[:, 2:] >= min_coverage).all(axis=1)
    marker_props = marker_props[sufficient_coverage]
    coverage = coverage[sufficient_coverage]
    batch_df = batch_df[sufficient_coverage].reset_index(drop=True)
    
    if len(batch_df) == 0:
        return None
        
    # Extract just the values matrix for cell types
    values_matrix = marker_props.iloc[:, 2:].values
    
    # Calculate SNR for each cell type (vectorized)
    snrs = np.zeros((len(values_matrix), len(cell_types)))
    for i, target in enumerate(cell_types):
        other_indices = [j for j, c in enumerate(cell_types) if c != target]
        max_background = values_matrix[:, other_indices].max(axis=1)
        snrs[:, i] = values_matrix[:, i] / (max_background + 1e-10)
    
    p_values = calculate_snr_significance_vectorized(values_matrix, cell_types)
    
    # Find best target for each row
    best_snrs = snrs.max(axis=1)
    best_targets_idx = snrs.argmax(axis=1)
    best_targets = np.array(cell_types)[best_targets_idx]
    best_pvalues = p_values[np.arange(len(p_values)), best_targets_idx]
    
    # Filter by SNR threshold and significance
    good_markers = (best_snrs > snr_threshold) & (best_pvalues < significance_threshold)
    
    if not good_markers.any():
        return None
        
    # Create result DataFrame
    result_df = batch_df[good_markers].copy()
    result_df['target'] = best_targets[good_markers]
    result_df['snr'] = best_snrs[good_markers]
    result_df['pvalue'] = best_pvalues[good_markers]
    
    # Add values for all cell types
    for cell in cell_types:
        result_df[cell] = marker_props[col_mapping[cell]][good_markers].values
        result_df[f'{cell}_coverage'] = coverage[col_mapping[cell]][good_markers].values
    
    return result_df
       

def process_all_regions(marker_regions, pats_path, wgbs_tools_exec_path, min_coverage=10, 
                      snr_threshold=1.0, significance_threshold=0.05, threads=32):
   """
   Process all region files in parallel and combine results
   """
   print(f"Processing {len(marker_regions)} region files...")
   
   # Prepare args for each file
   process_args = []
   for region_file in marker_regions:
       args = (region_file, pats_path, wgbs_tools_exec_path, min_coverage, 
              snr_threshold, significance_threshold)
       process_args.append(args)
   
   # Process in parallel
   with Pool(threads) as pool:
       results = list(tqdm(
           pool.imap(process_batch, process_args),
           total=len(process_args),
           desc="Processing region files"
       ))
   
   # Filter None results and combine
   valid_results = [df for df in results if df is not None]
   if not valid_results:
       print("No valid markers found.")
       return None
   
   # Combine all results
   print("Combining results...")
   combined_df = pd.concat(valid_results, ignore_index=True)
   print(f"Found {len(combined_df)} valid markers.")
   
   return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chr', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--pats_path', type=str, required=True)
    parser.add_argument('--wgbs_tools_exec_path', type=str, required=True)
    parser.add_argument('--snr_threshold', type=float, default=1.0)
    parser.add_argument('--min_coverage', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)

    args = parser.parse_args()
    marker_regions = glob.glob(f"{args.output_dir}/{args.chr}_region_*.l4.bed")
    
    result_df = process_all_regions(
       marker_regions,
       args.pats_path,
       args.wgbs_tools_exec_path,
       args.min_coverage,
       args.snr_threshold,
       args.significance_threshold,
       args.threads
    )
    
    if result_df is not None:
        # Save results
        output_file = f"{args.output_dir}/{args.chr}_markers.csv.gz"
        result_df.to_csv(output_file, index=False, compression='gzip')
        print(f"Results saved to {output_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total markers: {len(result_df)}")
        print("\nMarkers per target:")
        print(result_df['target'].value_counts())
        print("\nAverage SNR per target:")
        print(result_df.groupby('target')['snr'].mean())