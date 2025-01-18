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
        self.first_cpg = regions_df['startCpG'].min() - 20  # Look back 20 CpGs
        self.last_cpg = regions_df['endCpG'].max()
        for idx, row in regions_df.iterrows():
            self.regions.append(Region(
                start_cpg=row['startCpG'],
                end_cpg=row['endCpG'],
                index=idx
            ))
        self.regions.sort(key=lambda r: r.start_cpg)
        # Pre-compute end positions array for binary search
        self.end_positions = np.array([r.end_cpg for r in self.regions])
        # Initialize counts dictionary exactly as in original
        self.counts = defaultdict(lambda: {'u': 0, 'x': 0, 'm': 0})
    def find_overlapping_regions(self, pat_start: int, pattern: str) -> List[Tuple[Region, int, int]]:
        """Find overlapping regions with optimized but equivalent binary search"""
        valid_cpgs = sum(1 for c in pattern if c in 'CT')
        if valid_cpgs < self.min_cpgs:
            return []
            
        pat_end = pat_start + len(pattern) - 1
        overlaps = []
        
        # Binary search for first potential region
        left = np.searchsorted(self.end_positions, pat_start, side='right')
        # Check all potential overlapping regions
        for region in self.regions[left:]:
            if region.start_cpg > pat_end:
                break
                
            # Calculate overlap with half-open intervals
            overlap_start = max(pat_start, region.start_cpg)
            overlap_end = min(pat_end + 1, region.end_cpg)
            
            # Extract overlapping portion
            pattern_offset = overlap_start - pat_start
            overlap_pat = pattern[pattern_offset:pattern_offset + (overlap_end - overlap_start)]
            
            # Count valid CpGs in overlap
            valid_overlap_cpgs = sum(1 for c in overlap_pat if c in 'CT')
            
            if valid_overlap_cpgs >= self.min_cpgs:
                overlaps.append((region, pattern_offset, overlap_end - overlap_start))
                
        return overlaps
    def process_pattern(self, pattern: str, start_cpg: int, count: int):
        """Process pattern"""
        if len(pattern) < self.min_cpgs:
            return
        # Find overlapping regions
        overlaps = self.find_overlapping_regions(start_cpg, pattern)
        if overlaps:
            self.patterns_counted += 1
        for region, offset, overlap_len in overlaps:
            # Extract overlapping portion
            overlap_pat = pattern[offset:offset + overlap_len]
            # Calculate methylation ratio
            meth_count = overlap_pat.count('C')
            valid_cpgs = sum(1 for c in overlap_pat if c in 'CT')
            meth_ratio = meth_count / valid_cpgs
            # Update appropriate counter
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
    """Process a single pat file and return UXM proportions and coverage"""
    t_start = time.time()
    counter = RegionCounter(regions_df, min_cpgs)
    pat_file = str(pat_file)
    cell_type = Path(pat_file).stem.replace('.pat', '')
    column_names = ['chr', 'start', 'pattern', 'count']
    profiling_stats = defaultdict(float)
    
    # Count total lines for progress bar
    t0 = time.time()
    total_lines = sum(1 for _ in (gzip.open(pat_file, 'rt') if pat_file.endswith('.gz') else open(pat_file)))
    profiling_stats['count_lines'] = time.time() - t0
    
    processed_lines = 0
    with tqdm(total=total_lines, desc=f"Processing {cell_type}") as pbar:
        for chunk in pd.read_csv(pat_file, sep='\t', names=column_names, chunksize=1_000_000):
            t0 = time.time()
            # A pattern can overlap if:
            # - starts before last_cpg (starts before end of last region)
            # - extends past first_cpg (pattern end > first region start)
            relevant_chunk = chunk[
                (chunk['start'] < counter.last_cpg) & 
                (chunk['start'] + chunk['pattern'].str.len() > counter.first_cpg)
            ]
            profiling_stats['chunk_filtering'] += time.time() - t0
            
            if len(relevant_chunk) == 0:
                if chunk['start'].min() >= counter.last_cpg:
                    pbar.update(total_lines - processed_lines)
                    break  # Past all our regions
                processed_lines += len(chunk)
                pbar.update(len(chunk))
                continue
                
            t0 = time.time()
            for _, row in relevant_chunk.iterrows():
                counter.process_pattern(row['pattern'], row['start'], row['count'])
            profiling_stats['pattern_processing'] += time.time() - t0
            
            chunk_size = len(chunk)
            processed_lines += chunk_size
            pbar.update(chunk_size)
            
    # Convert counts to proportions and create output DataFrames
    t0 = time.time()
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
    profiling_stats['results_creation'] = time.time() - t0
    
    profiling_stats['total_time'] = time.time() - t_start
    print(f"\nProfiling for {cell_type}:")
    for key, value in profiling_stats.items():
        print(f"{key}: {value:.2f}s")
    
    return pd.DataFrame(results_uxm), pd.DataFrame(results_coverage), cell_type


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
    """Modified to include extended metrics in output"""
    good_markers = []
    
    for i in range(len(values_matrix)):
        is_good_marker, metrics = evaluate_marker_quality(
            values_matrix[i], 
            best_targets_idx[i],
            min_signal=min_signal_threshold,
            min_snr=snr_threshold,
            significance_threshold=significance_threshold
        )
        
        if is_good_marker:
            result = batch_df.iloc[i].copy()
            result['target'] = cell_types[best_targets_idx[i]]
            
            # Add all metrics to results
            for metric_name, metric_value in metrics.items():
                result[metric_name] = metric_value
            
            # Add cell type specific values and coverage (as in original)
            for cell in cell_types:
                result[cell] = marker_props[col_mapping[cell]].iloc[i]
                result[f'{cell}_coverage'] = coverage[col_mapping[cell]].iloc[i]
                
            good_markers.append(result)
    
    if not good_markers:
        return None
        
    result_df = pd.DataFrame(good_markers)
    if result_df is not None:
        grouped = result_df.groupby('target')
        for target, group in grouped:
            filename = f"{chr}_{target}_markers_{batch_id}.parquet"
            filepath = os.path.join(output_dir, filename)
            group.to_parquet(filepath, index=False)
            print(f"saved {len(group)} markers for chromosome {chr}/{target}")
            

def process_with_params(chr, pat_dir, regions, min_cpgs, min_coverage, snr_threshold, significance_threshold, min_signal_threshold, output_dir, threads, batch_size=500_000):
    print(f"Loading regions from {regions}...")
    t0 = time.time()
    batch_id = 0
    profiling_stats = defaultdict(float)
    for batch in pd.read_csv(regions, sep='\t', chunksize=batch_size):
        batch_id += 1
        t_batch = time.time()
        output_file = f'{output_dir}/{chr}_raw_markers_{batch_id}.l{min_cpgs}.bed.gz'
        if os.path.exists(output_file):
            print(f"Skipping batch {batch_id} as it was already processed")
            continue
        regions_df = batch.reset_index(drop=True)
        print(f"Processing batch {batch_id} with {len(regions_df)} regions")
        # Process pat files
        t1 = time.time()
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
        profiling_stats['pat_processing'] += time.time() - t1
        print("\nBuilding final matrices...")
        t1 = time.time()
        # Separate UXM and coverage results
        uxm_dfs = []
        coverage_dfs = []
        cell_types = []
        for uxm_df, coverage_df, cell_type in results:
            uxm_dfs.append(uxm_df)
            coverage_dfs.append(coverage_df)
            cell_types.append(cell_type)
        # Create final matrices
        base_df = regions_df[['name', 'direction']]
        uxm_matrix = base_df.copy()
        for df, cell_type in zip(uxm_dfs, cell_types):
            merged = pd.merge(base_df, 
                            df[['name', 'direction', 'value']], 
                            on=['name', 'direction'], 
                            how='left')
            uxm_matrix[f"{cell_type}_merged"] = merged['value']
        coverage_matrix = base_df.copy()
        for df, cell_type in zip(coverage_dfs, cell_types):
            merged = pd.merge(base_df, 
                            df[['name', 'direction', 'value']], 
                            on=['name', 'direction'], 
                            how='left')
            coverage_matrix[f"{cell_type}_merged"] = merged['value']
        profiling_stats['matrix_building'] += time.time() - t1
        t1 = time.time()
        marker_props, coverage = uxm_matrix, coverage_matrix
        col_mapping = {col.split('_')[0]: col for col in marker_props.columns if col not in ['name', 'direction']}
        cell_types = list(col_mapping.keys())
        valid_rows = ~marker_props.iloc[:, 2:].isna().any(axis=1)
        marker_props = marker_props[valid_rows]
        coverage = coverage[valid_rows]
        batch_df = regions_df[valid_rows].reset_index(drop=True)
        if len(batch_df) == 0:
            print(f"finished batch with no coverage {batch_id} in {time.time()-t_batch}")    
            continue 
        coverage.index = marker_props.index
        batch_df.index = marker_props.index
        sufficient_coverage = (coverage.iloc[:, 2:] >= min_coverage).all(axis=1)
        marker_props = marker_props[sufficient_coverage].reset_index(drop=True)
        coverage = coverage[sufficient_coverage].reset_index(drop=True)
        batch_df = batch_df[sufficient_coverage].reset_index(drop=True)
        profiling_stats['filtering'] += time.time() - t1
        if len(batch_df) == 0:
            print(f"finished batch with insufficient coverage {batch_id} in {time.time()-t_batch}")
            continue
        t1 = time.time()
        marker_props.to_csv(output_file, sep='\t', index=False, compression='gzip')
        coverage.to_csv(f'{output_dir}/{chr}_raw_coverage_{batch_id}.l{min_cpgs}.bed.gz', sep='\t', index=False, compression='gzip')
        profiling_stats['file_writing'] += time.time() - t1
        t1 = time.time()
        values_matrix = marker_props.iloc[:, 2:].values
        best_targets_idx = values_matrix.argmax(axis=1)
        find_good_markers(chr, batch_df, cell_types, marker_props, col_mapping, coverage, values_matrix, best_targets_idx, min_signal_threshold, snr_threshold, significance_threshold, output_dir, batch_id)
        profiling_stats['marker_evaluation'] += time.time() - t1
        batch_time = time.time() - t_batch
        print(f"Batch {batch_id} profiling:")
        print(f"{'Operation':<20} {'Time (s)':<10} {'Percentage':>10}")
        print("-" * 45)
        for operation, duration in profiling_stats.items():
            percentage = (duration / batch_time) * 100
            print(f"{operation:<20} {duration:>10.2f}s {percentage:>10.1f}%")
        print(f"\nfinished batch {batch_id} in {batch_time:.2f}s")
    total_time = time.time() - t0
    print(f"\nTotal chromosome processing time: {total_time:.2f}s")
    print("\nOverall profiling:")
    for operation, duration in profiling_stats.items():
        percentage = (duration / total_time) * 100
        print(f"{operation:<20} {duration:>10.2f}s {percentage:>10.1f}%")


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

