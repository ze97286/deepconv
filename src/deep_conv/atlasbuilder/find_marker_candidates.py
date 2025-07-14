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
import numba
import gc
import h5py
import glob

@dataclass
class Region:
    start_cpg: int
    end_cpg: int
    index: int  


# 1. Optimized function for counting valid CpGs
@numba.jit(nopython=True)
def count_valid_cpgs(pattern):
    count = 0
    for c in pattern:
        if c == 'C' or c == 'T':
            count += 1
    return count


# 2. Optimized function for counting valid CpGs in a slice
@numba.jit(nopython=True)
def count_valid_cpgs_slice(pattern, start, length):
    count = 0
    end = min(start + length, len(pattern))
    for i in range(start, end):
        if pattern[i] == 'C' or pattern[i] == 'T':
            count += 1
    return count


# 3. Fast filter for relevant chunks
@numba.jit(nopython=True)
def fast_filter(starts, pattern_lens, first_cpg, last_cpg):
    mask = np.zeros(len(starts), dtype=np.bool_)
    for i in range(len(starts)):
        if starts[i] < last_cpg and starts[i] + pattern_lens[i] > first_cpg:
            mask[i] = True
    return mask


# 4. Process multiple patterns in batch for better memory usage
def process_patterns_batch(patterns, starts, counts, counter):
    for pattern, start, count in zip(patterns, starts, counts):
        counter.process_pattern(pattern, start, count)


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
        # 5. Use optimized function for valid CpGs count
        valid_cpgs = count_valid_cpgs(pattern)
        if valid_cpgs < self.min_cpgs:
            return []
            
        pat_end = pat_start + len(pattern) - 1
        overlaps = []
        
        # 6. Optimized region finding with direct array indexing
        left = np.searchsorted(self.end_positions, pat_start, side='right')
        
        # 7. Direct iteration over region indices
        for i in range(left, len(self.regions)):
            region = self.regions[i]
            # Early termination when no more overlaps possible
            if region.start_cpg > pat_end:
                break
                
            # Calculate overlap
            overlap_start = max(pat_start, region.start_cpg)
            overlap_end = min(pat_end + 1, region.end_cpg)
            
            if overlap_start >= overlap_end:
                continue
                
            pattern_offset = overlap_start - pat_start
            overlap_len = overlap_end - overlap_start
            
            # 8. Use optimized function for slice valid CpGs count
            valid_overlap_cpgs = count_valid_cpgs_slice(pattern, pattern_offset, overlap_len)
            
            if valid_overlap_cpgs >= self.min_cpgs:
                overlaps.append((region, pattern_offset, overlap_len))
                
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
            # 9. Use optimized function for valid CpGs count
            valid_cpgs = count_valid_cpgs(overlap_pat)
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
    
    # 10. Improved progress tracking based on file size
    file_size = os.path.getsize(pat_file)
    processed_bytes = 0
    
    with tqdm(total=file_size, desc=f"Processing {cell_type}", unit='B', unit_scale=True) as pbar:
        file_handle = gzip.open(pat_file, 'rt') if pat_file.endswith('.gz') else open(pat_file)
        
        for chunk in pd.read_csv(pat_file, sep='\t', names=column_names, chunksize=1_000_000):
            # 11. Optimize chunk filtering using numba
            pattern_lens = np.array([len(p) for p in chunk['pattern']], dtype=np.int32)
            starts = chunk['start'].values.astype(np.int32)
            mask = fast_filter(starts, pattern_lens, counter.first_cpg, counter.last_cpg)
            relevant_chunk = chunk[mask]
            
            if len(relevant_chunk) == 0:
                if chunk['start'].min() >= counter.last_cpg:
                    # Early termination when past all regions
                    pbar.update(file_size - processed_bytes)
                    break
                chunk_bytes = chunk.memory_usage(deep=True).sum()
                processed_bytes += chunk_bytes
                pbar.update(chunk_bytes)
                continue
                
            # 12. Process patterns in batch
            process_patterns_batch(
                relevant_chunk['pattern'].values,
                relevant_chunk['start'].values,
                relevant_chunk['count'].values,
                counter
            )
            
            # Update progress based on memory usage
            chunk_bytes = chunk.memory_usage(deep=True).sum()
            processed_bytes += chunk_bytes
            pbar.update(chunk_bytes)
        
        file_handle.close()
    
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
    
    # 13. Memory cleanup
    gc.collect()
    
    return pd.DataFrame(results_uxm), pd.DataFrame(results_coverage), cell_type


def process_pat_file_with_name(regions_df, pat_file, min_cpgs):
    return {'file': pat_file.name, 'result': process_pat_file(pat_file=pat_file, min_cpgs=min_cpgs, regions_df=regions_df)}


# 14. Optimized DataFrame merge function using vectorized operations
def efficient_merge(base_df, value_df, key_cols=['name', 'direction'], value_col='value'):
    # Create lookup dictionary using vectorized operations
    lookup = {}
    
    # Vectorized key creation for value_df
    if len(key_cols) == 2:
        # Optimized for the common case of ['name', 'direction']
        for name, direction, value in zip(value_df[key_cols[0]], value_df[key_cols[1]], value_df[value_col]):
            lookup[(name, direction)] = value
    else:
        # General case for arbitrary key columns
        for i in range(len(value_df)):
            key = tuple(value_df.iloc[i][k] for k in key_cols)
            lookup[key] = value_df.iloc[i][value_col]
    
    # Vectorized lookup for base_df
    if len(key_cols) == 2:
        # Optimized for the common case
        result = [lookup.get((name, direction), np.nan) 
                 for name, direction in zip(base_df[key_cols[0]], base_df[key_cols[1]])]
    else:
        # General case
        result = [lookup.get(tuple(base_df.iloc[i][k] for k in key_cols), np.nan) 
                 for i in range(len(base_df))]
    
    return result


def create_marker_matrices(atlas_path: str, pat_dir: str, min_cpgs: int, threads=4) -> tuple[pd.DataFrame, pd.DataFrame]:
   """
   Create marker values matrix and coverage matrix from atlas markers and pat files.
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
   results = sorted(results, key=lambda x: x['file'])
   # If you need just the results in order:
   results = [r['result'] for r in results]
   # Create base matrix with name and direction
   base_df = markers_df[['name', 'direction']]
   
   # Build matrices efficiently with progress tracking
   print(f"Building marker and coverage matrices for {len(results)} samples...")
   
   # Prepare data for efficient creation
   marker_data = {'name': base_df['name'], 'direction': base_df['direction']}
   coverage_data = {'name': base_df['name'], 'direction': base_df['direction']}
   
   # Process with progress bar
   for uxm_df, coverage_df, cell_type in tqdm(results, desc="Merging sample data"):
       # 15. Use efficient merge instead of pandas merge
       marker_data[cell_type] = efficient_merge(
           base_df, 
           uxm_df[['name', 'direction', 'value']]
       )
       coverage_data[cell_type] = efficient_merge(
           base_df, 
           coverage_df[['name', 'direction', 'value']]
       )
   
   # Create matrices all at once to avoid fragmentation
   print("Creating final DataFrames...")
   marker_matrix = pd.DataFrame(marker_data)
   coverage_matrix = pd.DataFrame(coverage_data)
   
   # 16. Memory cleanup
   gc.collect()
   
   return marker_matrix, coverage_matrix

def create_marker_matrices_h5(atlas_path: str, pat_dir: str, min_cpgs: int, threads=4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create marker values matrix and coverage matrix from atlas markers and pat files.
    Handles both regular pat.gz files and HDF5 batch files.
    """
    # Read atlas
    print(f"Loading markers from {atlas_path}...")
    markers_df = pd.read_csv(atlas_path, sep='\t')
    
    # Check if this is an HDF5 directory
    h5_files = sorted(glob.glob(os.path.join(pat_dir, 'batch_*.h5')))
    
    if h5_files:
        # Extract all samples from HDF5 to temporary pat.gz files
        print(f"Found {len(h5_files)} HDF5 batch files. Extracting samples...")
        temp_dir = os.path.join(pat_dir, 'temp_pats')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract all samples
        for h5_file in tqdm(h5_files, desc="Extracting from HDF5"):
            with h5py.File(h5_file, 'r') as f:
                for sample_id in f.keys():
                    # Extract to pat.gz file
                    pat_file = os.path.join(temp_dir, f"{sample_id}.pat.gz")
                    
                    chroms = f[sample_id]['chromosomes'][:]
                    positions = f[sample_id]['positions'][:]
                    patterns = f[sample_id]['patterns'][:]
                    counts = f[sample_id]['counts'][:]
                    
                    with gzip.open(pat_file, 'wt') as out:
                        for chrom, pos, pattern, count in zip(chroms, positions, patterns, counts):
                            chrom = chrom.decode() if isinstance(chrom, bytes) else chrom
                            pattern = pattern.decode() if isinstance(pattern, bytes) else pattern
                            out.write(f"{chrom}\t{pos}\t{pattern}\t{count}\n")
        
        # Use temp directory for processing
        pat_files = sorted(list(Path(temp_dir).glob('*.pat.gz')))
    else:
        # Regular pat.gz files
        pat_files = sorted(list(Path(pat_dir).glob('*.pat.gz')))
    
    print(f"Found {len(pat_files)} pat files to process")
    
    # Use existing parallel processing
    with mp.Pool(threads) as pool:
        process_func = partial(process_pat_file_with_name, markers_df, min_cpgs=min_cpgs)
        results = list(tqdm(
            pool.imap(process_func, pat_files),
            total=len(pat_files),
            desc="Processing pat files"
        ))
    
    # Clean up temp files if we created them
    if h5_files and os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    
    # Sort results by filename
    results = sorted(results, key=lambda x: x['file'])
    # If you need just the results in order:
    results = [r['result'] for r in results]
    
    # Create base matrix with name and direction
    base_df = markers_df[['name', 'direction']]
    
    # Build matrices efficiently with progress tracking
    print(f"Building marker and coverage matrices for {len(results)} samples...")
    
    # Prepare data for efficient concatenation
    marker_data = {'name': base_df['name'], 'direction': base_df['direction']}
    coverage_data = {'name': base_df['name'], 'direction': base_df['direction']}
    
    # Process with progress bar
    for i, (uxm_df, coverage_df, cell_type) in enumerate(tqdm(results, desc="Merging sample data")):
        # Use efficient merge instead of pandas merge
        marker_data[cell_type] = efficient_merge(
            base_df, 
            uxm_df[['name', 'direction', 'value']]
        )
        coverage_data[cell_type] = efficient_merge(
            base_df, 
            coverage_df[['name', 'direction', 'value']]
        )
    
    # Create matrices all at once to avoid fragmentation
    print("Creating final DataFrames...")
    marker_matrix = pd.DataFrame(marker_data)
    coverage_matrix = pd.DataFrame(coverage_data)
    
    # Memory cleanup
    gc.collect()
    
    return marker_matrix, coverage_matrix


def get_ground_truth(pat_dir, names):
    dfs = []
    for n in names:
        dfs.append(pd.read_csv(str(pat_dir)+f"/{n}_true_concentrations.csv"))
    df=pd.concat(dfs, ignore_index=True)
    return df

def get_ground_truth_h5(pat_dir, names):
    # Check if this is an HDF5 directory
    h5_files = sorted(glob.glob(os.path.join(pat_dir, 'batch_*.h5')))
    
    if h5_files:
        # Extract concentration files from HDF5
        temp_dir = os.path.join(pat_dir, 'temp_concentrations')
        os.makedirs(temp_dir, exist_ok=True)
        
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                for sample_id in f.keys():
                    if sample_id in names:
                        # Extract concentration file
                        conc_file = os.path.join(temp_dir, f"{sample_id}_true_concentrations.csv")
                        cell_types = [ct.decode() if isinstance(ct, bytes) else ct 
                                     for ct in f[sample_id]['cell_types'][:]]
                        concentrations = f[sample_id]['concentrations'][:]
                        
                        with open(conc_file, 'w') as out:
                            for ct, conc in zip(cell_types, concentrations):
                                out.write(f"{ct},{conc}\n")
        
        # Use temp directory
        use_dir = temp_dir
    else:
        use_dir = pat_dir
    
    # Read files and reshape to expected format
    dfs = []
    for n in names:
        # Read the CSV without headers
        df = pd.read_csv(f"{use_dir}/{n}_true_concentrations.csv", header=None, names=['cell_type', 'concentration'])
        # Pivot to get cell types as columns
        df_pivot = df.set_index('cell_type').T
        df_pivot.index = [n]  # Set sample name as index
        dfs.append(df_pivot)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean up temp files if created
    if h5_files and os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    
    return df
def evaluate_marker_quality(values, target_idx, min_signal, min_snr, significance_threshold):
    """
    Evaluate marker quality with additional metrics while keeping core functionality
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
    
    # 17. Use context manager for better resource handling
    with pd.read_csv(regions, sep='\t', chunksize=batch_size) as reader:
        for batch in reader:
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
            # Create final matrices efficiently to avoid fragmentation
            print("Building final UXM and coverage matrices...")
            # First, create the base DataFrame with name and direction
            base_df = regions_df[['name', 'direction']]
            
            # Prepare data for efficient matrix creation
            uxm_data = {'name': base_df['name'], 'direction': base_df['direction']}
            coverage_data = {'name': base_df['name'], 'direction': base_df['direction']}
            
            for df, cell_type in zip(uxm_dfs, cell_types):
                # 18. Use efficient merge instead of pandas merge
                uxm_data[f"{cell_type}_merged"] = efficient_merge(base_df, df)
                
            for df, cell_type in zip(coverage_dfs, cell_types):
                # 19. Use efficient merge instead of pandas merge
                coverage_data[f"{cell_type}_merged"] = efficient_merge(base_df, df)
            
            # Create matrices all at once to avoid fragmentation
            uxm_matrix = pd.DataFrame(uxm_data)
            coverage_matrix = pd.DataFrame(coverage_data)
                
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
            
            # 20. Memory cleanup after batch processing
            gc.collect()
            
            print("finished batch",batch_id,"in",time.time()-t_batch)

    print("finished",chr, "in",time.time()-t0)


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


if __name__ == '__main__':
    main()
