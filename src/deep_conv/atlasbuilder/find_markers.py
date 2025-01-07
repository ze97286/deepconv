import argparse 
import glob 
import numpy as np
from tqdm import tqdm
from deep_conv.deconvolution.preprocess_pats import *


def evaluate_marker_quality(values, target_idx, min_signal, min_snr, significance_threshold):
    """
    Evaluate marker quality considering signal strength, SNR and statistical significance
    """
    target_value = values[target_idx]
    other_values = values[np.arange(len(values)) != target_idx]
    
    # Basic metrics
    max_background = other_values.max()
    snr = target_value / (max_background + 1e-10)
    
    # Calculate significance (p-value)
    p_value = np.mean(other_values >= target_value)
    
    # Additional quality metrics
    num_nonzero_background = (other_values > 0).sum()
    background_mean = other_values.mean()
    background_std = other_values.std()
    
    # Quality criteria
    is_good_marker = (
        (target_value > min_signal) &  # Minimum absolute signal
        (snr > min_snr) &  # Minimum SNR
        (p_value < significance_threshold)  # Statistical significance
    )
    
    return is_good_marker, {
        'snr': snr,
        'target_value': target_value,
        'max_background': max_background,
        'p_value': p_value,
        'num_nonzero_background': num_nonzero_background,
        'background_mean': background_mean,
        'background_std': background_std
    }


def process_batch(temp_atlas, pats_path, wgbs_tools_exec_path, min_cpgs, min_coverage, 
                  snr_threshold, significance_threshold, min_signal_threshold, threads):
    batch_df = pd.read_csv(temp_atlas,sep="\t")
    try:
        marker_props, coverage = pats_to_homog(temp_atlas, pats_path, wgbs_tools_exec_path, r_len=min_cpgs, threads = threads)
    except:
        print("failed to process batch for",temp_atlas)
        return None
    col_mapping = {col.split('_')[0]: col for col in marker_props.columns if col not in ['name', 'direction']}
    cell_types = list(col_mapping.keys())
    # Filter out rows with NAs or insufficient coverage
    valid_rows = ~marker_props.iloc[:, 2:].isna().any(axis=1)
    marker_props = marker_props[valid_rows]
    coverage = coverage[valid_rows]
    batch_df = batch_df[valid_rows].reset_index(drop=True)
    if len(batch_df) == 0:
        return None
    coverage.index = marker_props.index
    batch_df.index = marker_props.index
    sufficient_coverage = (coverage.iloc[:, 2:] >= min_coverage).all(axis=1)
    marker_props = marker_props[sufficient_coverage].reset_index(drop=True)
    coverage = coverage[sufficient_coverage].reset_index(drop=True)
    batch_df = batch_df[sufficient_coverage].reset_index(drop=True)
    if len(batch_df) == 0:
        return None
    values_matrix = marker_props.iloc[:, 2:].values
    best_targets_idx = values_matrix.argmax(axis=1)
    # Evaluate each region
    good_markers = []
    for i in range(len(values_matrix)):
        is_good_marker, metrics = evaluate_marker_quality(values_matrix[i], best_targets_idx[i], 
                                                        min_signal=min_signal_threshold, 
                                                        min_snr=snr_threshold,
                                                        significance_threshold=significance_threshold)
        if is_good_marker:  
            result = batch_df.iloc[i].copy()
            result['target'] = cell_types[best_targets_idx[i]]
            result['snr'] = metrics['snr']
            result['pvalue'] = metrics['p_value']  
            result['target_value'] = metrics['target_value']
            result['num_nonzero_background'] = metrics['num_nonzero_background']
            result['background_mean'] = metrics['background_mean']
            result['background_std'] = metrics['background_std']
            # Add all cell type values and coverage
            for cell in cell_types:
                result[cell] = marker_props[col_mapping[cell]].iloc[i]
                result[f'{cell}_coverage'] = coverage[col_mapping[cell]].iloc[i]
            print("found good marker for", cell_types[best_targets_idx[i]], "with SNR", metrics['snr'], "value",metrics['target_value'])
            good_markers.append(result)
    if not good_markers:
        return None
    return pd.DataFrame(good_markers)
    
        
def process_all_regions(marker_regions, 
                        pats_path, 
                        wgbs_tools_exec_path, 
                        min_cpgs, 
                        min_coverage, 
                        snr_threshold, 
                        significance_threshold, 
                        min_signal_threshold, 
                        threads):
   print(f"Processing {len(marker_regions)} region files...")
   valid_results = []
   for region_file in tqdm(marker_regions):
        df = process_batch(temp_atlas=region_file, 
                           pats_path=pats_path, 
                           wgbs_tools_exec_path=wgbs_tools_exec_path, 
                           min_cpgs=min_cpgs,
                           min_coverage= min_coverage,
                           snr_threshold=snr_threshold,
                           significance_threshold=significance_threshold,
                           min_signal_threshold=min_signal_threshold,
                           threads=threads)
        if df is not None:
            print("found good markers",df.head())
            valid_results.append(df)
   if not valid_results:
       print("No valid markers found.")
       return None
   print("Combining results...")
   combined_df = pd.concat(valid_results, ignore_index=True)
   print(f"Found {len(combined_df)} valid markers.")
   return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chr', type=str, required=True)
    parser.add_argument('--min_cpgs', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--pats_path', type=str, required=True)
    parser.add_argument('--wgbs_tools_exec_path', type=str, required=True)
    parser.add_argument('--snr_threshold', type=float, default=2.0)
    parser.add_argument('--significance_threshold', type=float, default=0.05)
    parser.add_argument('--min_signal_threshold', type=float, default=0.5)
    parser.add_argument('--min_coverage', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)

    args = parser.parse_args()
    marker_regions = sorted(glob.glob(f"{args.output_dir}/by_chr/{args.chr}_region_*.l4.bed"))
    result_df = process_all_regions(
        marker_regions=marker_regions,
        pats_path=args.pats_path,
        wgbs_tools_exec_path=args.wgbs_tools_exec_path,
        min_cpgs=args.min_cpgs,
        min_coverage=args.min_coverage,
        snr_threshold=args.snr_threshold,
        significance_threshold=args.significance_threshold,
        min_signal_threshold=args.min_signal_threshold,
        threads=args.threads,
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