import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict
import gzip
from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import os
import time

@dataclass
class Region:
    start_cpg: int
    end_cpg: int
    index: int  


class RegionCounter:
    def __init__(self, regions_df: pd.DataFrame, min_cpgs: int):
        self.patterns_counted = 0
        self.min_cpgs = min_cpgs
        self.th1 = round(1 - (min_cpgs - 1) / min_cpgs, 3) + 0.001
        self.th2 = round((min_cpgs - 1) / min_cpgs, 3)
        self.regions = []
        self.first_cpg = regions_df['startCpG'].min() - 20
        self.last_cpg = regions_df['endCpG'].max()
        self.counts = defaultdict(lambda: {'u': 0, 'x': 0, 'm': 0})

        # Sort regions by end_cpg and store in numpy arrays
        for idx, row in regions_df.iterrows():
            self.regions.append(Region(
                start_cpg=row['startCpG'],
                end_cpg=row['endCpG'],
                index=idx
            ))
        self.regions.sort(key=lambda r: r.end_cpg)
        
        # Create arrays for efficient searching
        self.end_positions = np.array([r.end_cpg for r in self.regions])
        self.start_positions = np.array([r.start_cpg for r in self.regions])
        
        # Create start position index by sorting regions by start_cpg
        self.start_sorted_indices = np.argsort(self.start_positions)
    
    def find_overlapping_regions(self, pat_start: int, pattern: str) -> List[Tuple[Region, int, int]]:
        valid_cpgs = sum(1 for c in pattern if c in 'CT')
        if valid_cpgs < self.min_cpgs:
            return []
            
        pat_end = pat_start + len(pattern) - 1
        overlaps = []
        
        # Find first region that ends after pattern start
        left = np.searchsorted(self.end_positions, pat_start, side='right')
        
        # Find last region that starts before pattern end
        right = np.searchsorted(self.start_positions[self.start_sorted_indices], pat_end, side='right')
        
        # Only check regions in this window
        candidate_indices = set(range(left, len(self.regions))) & set(self.start_sorted_indices[:right])
        
        for i in candidate_indices:
            region = self.regions[i]
            # Calculate overlap
            overlap_start = max(pat_start, region.start_cpg)
            overlap_end = min(pat_end + 1, region.end_cpg)
            
            if overlap_start >= overlap_end:
                continue
                
            pattern_offset = overlap_start - pat_start
            overlap_pat = pattern[pattern_offset:pattern_offset + (overlap_end - overlap_start)]
            
            valid_overlap_cpgs = sum(1 for c in overlap_pat if c in 'CT')
            
            if valid_overlap_cpgs >= self.min_cpgs:
                overlaps.append((region, pattern_offset, overlap_end - overlap_start))
                
        return overlaps
    def process_pattern(self, pattern: str, start_cpg: int, count: int):
        if len(pattern) < self.min_cpgs:
            return
        overlaps = self.find_overlapping_regions(start_cpg, pattern)
        if overlaps:
            self.patterns_counted += 1
        for region, offset, overlap_len in overlaps:
            overlap_pat = pattern[offset:offset + overlap_len]
            meth_count = overlap_pat.count('C')
            valid_cpgs = sum(1 for c in overlap_pat if c in 'CT')
            meth_ratio = meth_count / valid_cpgs
            if meth_ratio < self.th1:
                self.counts[region.index]['u'] += count
            elif meth_ratio > self.th2:
                self.counts[region.index]['m'] += count
            else:
                self.counts[region.index]['x'] += count


def create_empty_results(regions_df: pd.DataFrame, cell_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_uxm = []
    results_coverage = []
    for idx in range(len(regions_df)):
        results_uxm.append({
            'name': regions_df.iloc[idx]['name'],
            'direction': regions_df.iloc[idx]['direction'],
            'value': np.nan,
            'cell_type': cell_type
        })
        results_coverage.append({
            'name': regions_df.iloc[idx]['name'],
            'direction': regions_df.iloc[idx]['direction'],
            'value': 0,
            'cell_type': cell_type
        })
    return pd.DataFrame(results_uxm), pd.DataFrame(results_coverage), cell_type


def process_pat_file(regions_df: pd.DataFrame, pat_file: str, min_cpgs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counter = RegionCounter(regions_df, min_cpgs)
    pat_file = str(pat_file)
    cell_type = Path(pat_file).stem.replace('.pat', '')
    column_names = ['chr', 'start', 'pattern', 'count']
    total_lines = sum(1 for _ in (gzip.open(pat_file, 'rt') if pat_file.endswith('.gz') else open(pat_file)))
    processed_lines = 0
    
    with tqdm(total=total_lines, desc=f"Processing {cell_type}") as pbar:
        for chunk in pd.read_csv(pat_file, sep='\t', names=column_names, chunksize=1_000_000):
            relevant_chunk = chunk[
                (chunk['start'] < counter.last_cpg) & 
                (chunk['start'] + chunk['pattern'].str.len() > counter.first_cpg)
            ]
            if len(relevant_chunk) == 0:
                if chunk['start'].min() >= counter.last_cpg:
                    pbar.update(total_lines - processed_lines)
                    break  # Past all our regions
                processed_lines += len(chunk)
                pbar.update(len(chunk))
                continue
            for _, row in relevant_chunk.iterrows():
                counter.process_pattern(row['pattern'], row['start'], row['count'])
            chunk_size = len(chunk)
            processed_lines += chunk_size
            pbar.update(chunk_size)
    results_uxm = []
    results_coverage = []
    for idx in range(len(regions_df)):
        counts = counter.counts[idx]
        total = sum(counts.values())
        if total > 0:
            results_uxm.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': counts['u'] / total,
                'cell_type': cell_type
            })
            results_coverage.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': total,
                'cell_type': cell_type
            })
        else:
            results_uxm.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': np.nan,
                'cell_type': cell_type
            })
            results_coverage.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': 0,
                'cell_type': cell_type
            })
    return pd.DataFrame(results_uxm), pd.DataFrame(results_coverage), cell_type


def process_pat_file_with_name(regions_df, pat_file, min_cpgs):
    return {'file': pat_file.name, 'result': process_pat_file(pat_file=pat_file, min_cpgs=min_cpgs, regions_df=regions_df)}


def create_marker_matrices(atlas_path: str, pat_dir: str, min_cpgs: int, threads=4) -> tuple[pd.DataFrame, pd.DataFrame]:
   """
   Create marker values matrix and coverage matrix from atlas markers and pat files.
   Args:
       atlas_path: Path to atlas file containing markers
       pat_dir: Directory containing pat files
       min_cpgs: Minimum CpGs required for overlap
   
   Returns:
       tuple of (marker_matrix, coverage_matrix) where:
       - Rows are markers from atlas
       - First columns are name, direction
       - Remaining columns are values/coverage for each pat file
   """
   # Read atlas
   print(f"Loading markers from {atlas_path}...")
   markers_df = pd.read_csv(atlas_path, sep='\t')
   # Get pat files
   pat_files = sorted(list(Path(pat_dir).glob('*.pat.gz')))
   print(f"Found {len(pat_files)} pat files in {pat_dir}")
   with mp.Pool(threads) as pool:
        process_func = partial(process_pat_file_with_name, markers_df, min_cpgs=min_cpgs)
        results = list(tqdm(
            pool.imap(process_func, pat_files),
            total=len(pat_files),
            desc="Processing pat files"
        ))
    # Sort results by filename
   results = sorted(results, key=lambda x: x['file'])  # or key=lambda x: x[0] if using tuples
    # If you need just the results in order:
   results = [r['result'] for r in results]
   # Create base matrix with name and direction
   base_df = markers_df[['name', 'direction']]
   # Build matrices
   marker_matrix = base_df.copy()
   coverage_matrix = base_df.copy()
   for uxm_df, coverage_df, cell_type in results:
       # Add columns for this pat file's values
       marker_matrix[cell_type] = pd.merge(
           base_df, 
           uxm_df[['name', 'direction', 'value']], 
           on=['name', 'direction'], 
           how='left'
       )['value']
       coverage_matrix[cell_type] = pd.merge(
           base_df, 
           coverage_df[['name', 'direction', 'value']], 
           on=['name', 'direction'], 
           how='left'
       )['value'].fillna(0)
   return marker_matrix, coverage_matrix


def get_ground_truth(pat_dir, names, cell_types):
    dfs = []
    for n in names:
        dfs.append(pd.read_csv(str(pat_dir)+f"/{n}_true_concentrations.csv"))
    df=pd.concat(dfs, ignore_index=True)
    df.columns=cell_types

    return df[sorted(cell_types)]


def evaluate_marker_quality(values, target_idx, min_signal, min_snr, significance_threshold):
    """
    Evaluate marker quality with additional metrics while keeping core functionality
    
    Args:
        values: Array of UXM proportions for all cell types
        target_idx: Index of target cell type
        min_signal: Minimum required signal
        min_snr: Minimum required SNR
        significance_threshold: P-value threshold
    """
    target_value = values[target_idx]
    other_values = values[np.arange(len(values)) != target_idx]
    
    # Core metrics (as in original)
    max_background = other_values.max()
    median_background = np.median(other_values)
    mean_background = other_values.mean()
    background_std = other_values.std()
    
    # SNR calculations (expanded)
    snr = target_value / (max_background + 1e-10)
    snr_vs_median = target_value / (median_background + 1e-10)
    snr_vs_mean = target_value / (mean_background + 1e-10)
    
    # Statistical significance
    p_value = np.mean(other_values >= target_value)
    
    # Additional metrics that might be useful for model training
    metrics = {
        'snr': snr,
        'snr_vs_median': snr_vs_median,
        'snr_vs_mean': snr_vs_mean,
        'target_value': target_value,
        'max_background': max_background,
        'median_background': median_background,
        'mean_background': mean_background,
        'background_std': background_std,
        'p_value': p_value,
        'background_range': np.ptp(other_values),  # Peak-to-peak range
        'background_quartile_ratio': np.percentile(other_values, 75) / (np.percentile(other_values, 25) + 1e-10),
        'signal_to_noise_area': target_value - mean_background - background_std,
        'relative_signal_strength': (target_value - mean_background) / (max_background - mean_background + 1e-10)
    }
    
    # Core quality criteria (as in original)
    is_good_marker = (
        (target_value > min_signal) &  # Minimum absolute signal
        ((snr > min_snr) | (snr_vs_median > min_snr) | (snr_vs_mean > min_snr)) &  # Minimum SNR
        (p_value < significance_threshold)  # Statistical significance
    )
    
    return is_good_marker, metrics


def find_good_markers(chr, batch_df, cell_types, marker_props, col_mapping, coverage, 
                     values_matrix, best_targets_idx, min_signal_threshold, 
                     snr_threshold, significance_threshold, output_dir, batch_id):
    """Find good markers efficiently using bulk operations"""
    # Find all good markers with their metrics
    good_indices = []
    good_metrics = []
    
    for i in range(len(values_matrix)):
        is_good_marker, metrics = evaluate_marker_quality(
            values_matrix[i],
            best_targets_idx[i],
            min_signal=min_signal_threshold,
            min_snr=snr_threshold,
            significance_threshold=significance_threshold
        )
        if is_good_marker:
            good_indices.append(i)
            good_metrics.append(metrics)
            
    if not good_indices:
        return None
        
    # Get all column names we'll need
    metric_columns = list(good_metrics[0].keys())
    
    # Create the base result from batch_df
    result_df = batch_df.iloc[good_indices].copy()
    
    # Add target column
    result_df['target'] = [cell_types[idx] for idx in best_targets_idx[good_indices]]
    
    # Add metrics columns efficiently
    for metric in metric_columns:
        result_df[metric] = [m[metric] for m in good_metrics]
    
    # Add cell type values and coverage efficiently
    for cell in cell_types:
        result_df[cell] = marker_props[col_mapping[cell]].iloc[good_indices].values
        result_df[f'{cell}_coverage'] = coverage[col_mapping[cell]].iloc[good_indices].values
    
    # Save results
    grouped = result_df.groupby('target')
    for target, group in grouped:
        filename = f"{chr}_{target}_markers_{batch_id}.parquet"
        filepath = os.path.join(output_dir, filename)
        group.to_parquet(filepath, index=False)
        print(f"saved {len(group)} markers for chromosome {chr}/{target}")
    
    return result_df


def process_with_params(chr, pat_dir, regions, min_cpgs, min_coverage, snr_threshold, significance_threshold, min_signal_threshold, output_dir, threads, batch_size=500_000):
    print(f"Loading regions from {regions}...")
    t0 = time.time()
    batch_id=0
    for batch in pd.read_csv(regions, sep='\t', chunksize=batch_size):
        batch_id+=1
        output_file = f'{output_dir}/{chr}_raw_markers_{batch_id}.l{min_cpgs}.bed.gz'
        if os.path.exists(output_file):
            print(f"Skipping batch {batch_id} as it was already processed")
            continue
        t_batch = time.time()
        regions_df = batch.reset_index(drop=True) 
        print(f"Loaded {len(regions_df)} regions")
        pat_files = list(Path(pat_dir).glob('*.pat.gz'))
        if not pat_files:
            raise ValueError(f"No .pat.gz files found in {pat_dir}")
        with mp.Pool(threads) as pool:
            process_func = partial(process_pat_file, regions_df, min_cpgs=min_cpgs)
            results = list(tqdm(
                pool.imap(process_func, pat_files),
                total=len(pat_files),
                desc="Overall progress"
            ))
        print("\nBuilding final matrices...")
        # Separate UXM and coverage results
        uxm_dfs = []
        coverage_dfs = []
        cell_types = []
        for uxm_df, coverage_df, cell_type in results:
            uxm_dfs.append(uxm_df)
            coverage_dfs.append(coverage_df)
            cell_types.append(cell_type)
        # Create final matrices
        # First, create the base DataFrame with name and direction
        base_df = regions_df[['name', 'direction']]
        # Create UXM matrix
        uxm_matrix = base_df.copy()
        for df, cell_type in zip(uxm_dfs, cell_types):
            # First merge base_df with current results
            merged = pd.merge(base_df, 
                            df[['name', 'direction', 'value']], 
                            on=['name', 'direction'], 
                            how='left')
            # Then assign to new column
            uxm_matrix[f"{cell_type}_merged"] = merged['value']
        # Create coverage matrix                           
        coverage_matrix = base_df.copy()
        for df, cell_type in zip(coverage_dfs, cell_types):
            merged = pd.merge(base_df, 
                            df[['name', 'direction', 'value']], 
                            on=['name', 'direction'], 
                            how='left')
            coverage_matrix[f"{cell_type}_merged"] = merged['value']
        marker_props, coverage = uxm_matrix, coverage_matrix
        col_mapping = {col.split('_')[0]: col for col in marker_props.columns if col not in ['name', 'direction']}
        cell_types = list(col_mapping.keys())
        valid_rows = ~marker_props.iloc[:, 2:].isna().any(axis=1)
        marker_props = marker_props[valid_rows]
        coverage = coverage[valid_rows]
        batch_df = regions_df
        batch_df = batch_df[valid_rows].reset_index(drop=True)
        if len(batch_df) == 0:
            print("finished batch with no coverage",batch_id,"in",time.time()-t_batch)    
            continue 
        coverage.index = marker_props.index
        batch_df.index = marker_props.index
        sufficient_coverage = (coverage.iloc[:, 2:] >= min_coverage).all(axis=1)
        marker_props = marker_props[sufficient_coverage].reset_index(drop=True)
        coverage = coverage[sufficient_coverage].reset_index(drop=True)
        batch_df = batch_df[sufficient_coverage].reset_index(drop=True)
        if len(batch_df) == 0:
            print("finished batch with insufficient coverage",batch_id,"in",time.time()-t_batch)
            continue
        marker_props.to_csv(f'{output_dir}/{chr}_raw_markers_{batch_id}.l{min_cpgs}.bed.gz', sep='\t', index=False, compression='gzip')
        coverage.to_csv(f'{output_dir}/{chr}_raw_coverage_{batch_id}.l{min_cpgs}.bed.gz', sep='\t', index=False, compression='gzip')
        values_matrix = marker_props.iloc[:, 2:].values
        best_targets_idx = values_matrix.argmax(axis=1)
        find_good_markers(chr, batch_df, cell_types, marker_props, col_mapping, coverage, values_matrix, best_targets_idx, min_signal_threshold, snr_threshold, significance_threshold, output_dir, batch_id)
        print("finished batch",batch_id,"in",time.time()-t_batch)

    print("finished",chr, "in",time.time()-t0)


# python -m deep_conv.atlasbuilder.find_marker_candidates \
# --chr 21 \
# --pat_dir /users/zetzioni/sharedscratch/atlas/pat_by_cell_type \
# --regions /users/zetzioni/sharedscratch/atlas/marker_regions/regions_chr21_4_1000.bed.gz \
# --min_coverage 10 \
# --snr_threshold 2.5 \
# --significance_threshold 0.05 \
# --min_signal_threshold 0.5 \
# --output_dir /users/zetzioni/sharedscratch/atlas/marker_regions/ \
# --threads 15 

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process pat files for UXM analysis')
    parser.add_argument('--chr', required=True, help='chromosome to run for')
    parser.add_argument('--pat_dir', required=True, help='Directory containing pat files')
    parser.add_argument('--regions', required=True, help='Path to regions BED file')
    parser.add_argument('--min_cpgs', type=int, required=True, help='Minimum CpGs required')
    parser.add_argument('--min_coverage', type=int, required=True, help='Minimum coverage per region required')
    parser.add_argument('--snr_threshold', type=float, default=2.0)
    parser.add_argument('--significance_threshold', type=float, default=0.05)
    parser.add_argument('--min_signal_threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', required=True, help='Path to output marker and coverage files')
    parser.add_argument('--threads', type=int, default=mp.cpu_count(), help='Number of threads')
    parser.add_argument('--batch_size', type=int, default=100_000, help='Batch size')
    args = parser.parse_args()

    process_with_params(args.chr, args.pat_dir, args.regions, args.min_cpgs, args.min_coverage, args.snr_threshold, args.significance_threshold, args.min_signal_threshold, args.output_dir, args.threads, batch_size=args.batch_size)
    # process_with_params(chr="chr2", pat_dir="/users/zetzioni/sharedscratch/atlas/pat_by_cell_type", regions="/users/zetzioni/sharedscratch/atlas/marker_regions/regions_chr2_4_1000.bed.gz", min_cpgs=4, min_coverage=10, snr_threshold=1,significance_threshold=0.05, min_signal_threshold=0.1,output_dir="/users/zetzioni/sharedscratch/atlas/marker_regions", threads=15, batch_size=10_000)


if __name__ == '__main__':
    main()

