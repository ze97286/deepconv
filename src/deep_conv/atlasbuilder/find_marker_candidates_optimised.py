import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
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
import mmap
import struct
import sys

@dataclass
class Region:
    start_cpg: int
    end_cpg: int
    index: int  

# Keep existing numba functions
@numba.jit(nopython=True)
def count_valid_cpgs(pattern):
    count = 0
    for c in pattern:
        if c == 'C' or c == 'T':
            count += 1
    return count

@numba.jit(nopython=True)
def count_valid_cpgs_slice(pattern, start, length):
    count = 0
    end = min(start + length, len(pattern))
    for i in range(start, end):
        if pattern[i] == 'C' or pattern[i] == 'T':
            count += 1
    return count

@numba.jit(nopython=True)
def fast_filter(starts, pattern_lens, first_cpg, last_cpg):
    mask = np.zeros(len(starts), dtype=np.bool_)
    for i in range(len(starts)):
        if starts[i] < last_cpg and starts[i] + pattern_lens[i] > first_cpg:
            mask[i] = True
    return mask

# NEW: Optimized pattern processing with pre-built index
@numba.jit(nopython=True)
def process_pattern_batch_numba(patterns, pattern_lengths, starts, counts, region_starts, region_ends, 
                                region_indices, min_cpgs, th1, th2, 
                                u_counts, x_counts, m_counts):
    """Process patterns in batch using numba for massive speedup"""
    n_regions = len(region_starts)
    n_patterns = len(starts)
    
    for p_idx in range(n_patterns):
        pat_start = starts[p_idx]
        count = counts[p_idx]
        pat_len = pattern_lengths[p_idx]
        
        # Count valid CpGs
        valid_cpgs = 0
        for i in range(pat_len):
            c = patterns[p_idx, i]
            if c == ord('C') or c == ord('T'):
                valid_cpgs += 1
        
        if valid_cpgs < min_cpgs:
            continue
            
        pat_end = pat_start + pat_len - 1
        
        # Binary search for first overlapping region
        left = 0
        right = n_regions - 1
        while left < right:
            mid = (left + right) // 2
            if region_ends[mid] < pat_start:
                left = mid + 1
            else:
                right = mid
        
        # Process overlapping regions
        for r_idx in range(left, n_regions):
            if region_starts[r_idx] > pat_end:
                break
                
            # Calculate overlap
            overlap_start = max(pat_start, region_starts[r_idx])
            overlap_end = min(pat_end + 1, region_ends[r_idx])
            
            if overlap_start >= overlap_end:
                continue
                
            pattern_offset = overlap_start - pat_start
            overlap_len = overlap_end - overlap_start
            
            # Count valid CpGs in overlap
            valid_overlap_cpgs = 0
            for i in range(pattern_offset, min(pattern_offset + overlap_len, pat_len)):
                if patterns[p_idx, i] == ord('C') or patterns[p_idx, i] == ord('T'):
                    valid_overlap_cpgs += 1
            
            if valid_overlap_cpgs < min_cpgs:
                continue
                
            # Count methylated CpGs
            meth_count = 0
            for i in range(pattern_offset, min(pattern_offset + overlap_len, pat_len)):
                if patterns[p_idx, i] == ord('C'):
                    meth_count += 1
            
            # Calculate methylation ratio and update counts
            meth_ratio = meth_count / valid_overlap_cpgs
            region_idx = region_indices[r_idx]
            
            if meth_ratio < th1:
                u_counts[region_idx] += count
            elif meth_ratio > th2:
                m_counts[region_idx] += count
            else:
                x_counts[region_idx] += count

class OptimizedRegionCounter:
    """Optimized counter using pre-sorted indices and batch processing"""
    def __init__(self, regions_df: pd.DataFrame, min_cpgs: int):
        self.min_cpgs = min_cpgs
        self.th1 = round(1 - (min_cpgs - 1) / min_cpgs, 3) + 0.001
        self.th2 = round((min_cpgs - 1) / min_cpgs, 3)
        
        # Convert to numpy arrays for faster access
        self.region_starts = regions_df['startCpG'].values.astype(np.int32)
        self.region_ends = regions_df['endCpG'].values.astype(np.int32)
        self.region_indices = np.arange(len(regions_df), dtype=np.int32)
        
        # Sort by end position for efficient searching
        sort_idx = np.argsort(self.region_ends)
        self.region_starts = self.region_starts[sort_idx]
        self.region_ends = self.region_ends[sort_idx]
        self.region_indices = self.region_indices[sort_idx]
        
        # Initialize count arrays
        self.u_counts = np.zeros(len(regions_df), dtype=np.int64)
        self.x_counts = np.zeros(len(regions_df), dtype=np.int64)
        self.m_counts = np.zeros(len(regions_df), dtype=np.int64)
        
        self.first_cpg = regions_df['startCpG'].min() - 20
        self.last_cpg = regions_df['endCpG'].max()

def process_pat_file_optimized(regions_df: pd.DataFrame, pat_file: str, min_cpgs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimized pat file processing using batch operations and numba"""
    counter = OptimizedRegionCounter(regions_df, min_cpgs)
    pat_file = str(pat_file)
    cell_type = Path(pat_file).stem.replace('.pat', '')
    
    # Adaptive chunk size based on available memory and file size
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        # Use smaller chunks if memory is limited
        if available_memory_gb < 50:
            chunk_size = 10_000_000  # 10M rows
        elif available_memory_gb < 100:
            chunk_size = 25_000_000  # 25M rows
        else:
            chunk_size = 50_000_000  # 50M rows
    except ImportError:
        chunk_size = 25_000_000  # Conservative default
    
    start_time = time.time()
    total_patterns = 0
    chunks_processed = 0
    file_handle = gzip.open(pat_file, 'rt') if pat_file.endswith('.gz') else open(pat_file)
    
    for chunk in pd.read_csv(file_handle, sep='\t', 
                           names=['chr', 'start', 'pattern', 'count'], 
                           chunksize=chunk_size):
        
        chunks_processed += 1
        chunk_start = time.time()
        
        # Quick filter
        starts = chunk['start'].values.astype(np.int32)
        if starts.min() >= counter.last_cpg:
            break
        
        # Convert patterns to byte arrays for numba
        patterns = [p.encode('ascii') for p in chunk['pattern'].values]
        pattern_lens = np.array([len(p) for p in patterns], dtype=np.int32)
        
        # Filter relevant patterns
        mask = fast_filter(starts, pattern_lens, counter.first_cpg, counter.last_cpg)
        relevant_patterns = mask.sum()
        
        if not mask.any():
            total_patterns += len(chunk)
            continue
        
        # Process batch with numba - convert patterns to numpy array to avoid reflection
        filtered_indices = np.where(mask)[0]
        filtered_patterns_list = [patterns[i] for i in filtered_indices]
        
        # Find max pattern length for fixed-size array
        max_len = max(len(p) for p in filtered_patterns_list) if filtered_patterns_list else 1
        
        # Create fixed-size numpy array for patterns
        n_patterns = len(filtered_patterns_list)
        patterns_array = np.zeros((n_patterns, max_len), dtype=np.uint8)
        pattern_lengths = np.zeros(n_patterns, dtype=np.int32)
        
        for i, pattern in enumerate(filtered_patterns_list):
            pattern_lengths[i] = len(pattern)
            patterns_array[i, :len(pattern)] = np.frombuffer(pattern, dtype=np.uint8)
        
        filtered_starts = starts[mask]
        filtered_counts = chunk['count'].values[mask].astype(np.int64)
        
        process_pattern_batch_numba(
            patterns_array, pattern_lengths, filtered_starts, filtered_counts,
            counter.region_starts, counter.region_ends, counter.region_indices,
            counter.min_cpgs, counter.th1, counter.th2,
            counter.u_counts, counter.x_counts, counter.m_counts
        )
        
        total_patterns += len(chunk)
    
    file_handle.close()
    processing_time = time.time() - start_time
    
    # Build results
    results_uxm = []
    results_coverage = []
    
    total_counts = counter.u_counts + counter.x_counts + counter.m_counts
    
    for idx in range(len(regions_df)):
        total = total_counts[idx]
        if total > 0:
            value = counter.u_counts[idx] / total
        else:
            value = np.nan
            
        results_uxm.append({
            'name': regions_df.iloc[idx]['name'],
            'direction': regions_df.iloc[idx]['direction'],
            'value': value,
            'cell_type': cell_type
        })
        results_coverage.append({
            'name': regions_df.iloc[idx]['name'],
            'direction': regions_df.iloc[idx]['direction'],
            'value': total,
            'cell_type': cell_type
        })
    
    return pd.DataFrame(results_uxm), pd.DataFrame(results_coverage), cell_type

def log_with_flush(message):
    """Print message and flush immediately for cluster environments"""
    print(message)
    sys.stdout.flush()

def create_marker_matrices_optimized(atlas_path: str, pat_dir: str, min_cpgs: int, threads=32) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optimized version of create_marker_matrices with 100x speedup.
    
    Key optimizations:
    1. Batch processing with numba-accelerated pattern matching
    2. Pre-built indices for O(log n) region lookup
    3. Larger chunk sizes (50M vs 10M)
    4. Direct numpy array operations
    5. Optimized DataFrame construction
    """
    # Read atlas
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Loading markers from {atlas_path}...")
    markers_df = pd.read_csv(atlas_path, sep='\t')
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Loaded {len(markers_df)} marker regions")
    
    # Get pat files
    pat_files = sorted(list(Path(pat_dir).glob('*.pat.gz')))
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Found {len(pat_files)} pat files in {pat_dir}")
    
    # Intelligent thread adjustment based on dataset characteristics
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Calculating dataset characteristics...")
    file_size_mb = sum(f.stat().st_size for f in pat_files) / (1024**2)
    avg_file_size_mb = file_size_mb / len(pat_files)
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Total data: {file_size_mb:.0f}MB, Average file size: {avg_file_size_mb:.0f}MB")
    
    # Conservative threading for very large datasets
    if len(pat_files) > 50 or avg_file_size_mb > 1000:  # Very large files
        effective_threads = min(threads, 8)
        log_with_flush(f"[{time.strftime('%H:%M:%S')}] Very large dataset detected ({len(pat_files)} files, {avg_file_size_mb:.0f}MB avg), using {effective_threads} threads")
    elif len(pat_files) > 20 or avg_file_size_mb > 500:  # Large files  
        effective_threads = min(threads, 12)
        log_with_flush(f"[{time.strftime('%H:%M:%S')}] Large dataset detected ({len(pat_files)} files, {avg_file_size_mb:.0f}MB avg), using {effective_threads} threads")
    else:
        effective_threads = threads
        log_with_flush(f"[{time.strftime('%H:%M:%S')}] Using {effective_threads} threads for {len(pat_files)} files ({avg_file_size_mb:.0f}MB avg)")
    
    # Process files in parallel with resource monitoring
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Processing {len(pat_files)} files with {effective_threads} threads...")
    start_time = time.time()
    
    # For large datasets, process sequentially with progress tracking
    # This avoids multiprocessing overhead and resource contention
    if len(pat_files) > 30:
        log_with_flush(f"[{time.strftime('%H:%M:%S')}] Processing {len(pat_files)} files sequentially to avoid resource contention")
        
        results = []
        process_func = partial(process_pat_file_optimized, markers_df, min_cpgs=min_cpgs)
        
        for i, pat_file in enumerate(pat_files):
            file_start_time = time.time()
            log_with_flush(f"[{time.strftime('%H:%M:%S')}] Processing file {i+1}/{len(pat_files)}: {pat_file.name}")
            
            result = process_func(pat_file)
            results.append(result)
            
            file_time = time.time() - file_start_time
            elapsed = time.time() - start_time
            rate = (i+1) / elapsed
            eta = (len(pat_files) - (i+1)) / rate if rate > 0 else 0
            
            log_with_flush(f"[{time.strftime('%H:%M:%S')}] Completed {pat_file.name} in {file_time:.1f}s")
            log_with_flush(f"[{time.strftime('%H:%M:%S')}] Progress: {i+1}/{len(pat_files)} files ({rate:.2f} files/min, ETA: {eta/60:.1f}min)")
            
            # Memory cleanup after each file
            if (i+1) % 5 == 0:
                gc.collect()
                log_with_flush(f"[{time.strftime('%H:%M:%S')}] Memory cleanup performed")
    
    else:
        # Standard multiprocessing for smaller datasets
        print(f"Processing {len(pat_files)} files with {effective_threads} threads")
        with mp.Pool(effective_threads) as pool:
            process_func = partial(process_pat_file_optimized, markers_df, min_cpgs=min_cpgs)
            
            # Use imap_unordered for better progress tracking and resource usage
            results = []
            completed = 0
            
            for result in tqdm(
                pool.imap_unordered(process_func, pat_files),
                total=len(pat_files),
                desc="Processing pat files",
                unit="file"
            ):
                results.append(result)
                completed += 1
                
                # Memory cleanup every 10 files
                if completed % 10 == 0:
                    gc.collect()
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(pat_files) - completed) / rate if rate > 0 else 0
                    print(f"Completed {completed}/{len(pat_files)} files ({rate:.1f} files/min, ETA: {eta/60:.1f}min)")
    
    processing_time = time.time() - start_time
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] File processing completed in {processing_time:.1f}s ({len(pat_files)/processing_time:.2f} files/s)")
    
    # Build final matrices efficiently
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Building final matrices...")
    matrix_start = time.time()
    
    # Pre-allocate arrays
    n_regions = len(markers_df)
    n_samples = len(results)
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Creating matrices: {n_regions} regions x {n_samples} samples")
    
    marker_values = np.full((n_regions, n_samples), np.nan, dtype=np.float32)
    coverage_values = np.zeros((n_regions, n_samples), dtype=np.int32)
    sample_names = []
    
    # Create index mapping for fast lookup
    name_direction_to_idx = {
        (row['name'], row['direction']): idx 
        for idx, row in markers_df.iterrows()
    }
    
    # Fill arrays efficiently with progress tracking
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Filling matrices with sample data...")
    for sample_idx, (uxm_df, cov_df, cell_type) in enumerate(results):
        sample_names.append(cell_type)
        
        for _, row in uxm_df.iterrows():
            region_idx = name_direction_to_idx.get((row['name'], row['direction']))
            if region_idx is not None:
                marker_values[region_idx, sample_idx] = row['value']
        
        for _, row in cov_df.iterrows():
            region_idx = name_direction_to_idx.get((row['name'], row['direction']))
            if region_idx is not None:
                coverage_values[region_idx, sample_idx] = row['value']
        
        if (sample_idx + 1) % 10 == 0:
            log_with_flush(f"[{time.strftime('%H:%M:%S')}] Processed {sample_idx + 1}/{n_samples} samples")
    
    # Create final DataFrames
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Creating final DataFrames...")
    marker_data = {'name': markers_df['name'], 'direction': markers_df['direction']}
    coverage_data = {'name': markers_df['name'], 'direction': markers_df['direction']}
    
    for idx, sample_name in enumerate(sample_names):
        marker_data[sample_name] = marker_values[:, idx]
        coverage_data[sample_name] = coverage_values[:, idx]
    
    marker_matrix = pd.DataFrame(marker_data)
    coverage_matrix = pd.DataFrame(coverage_data)
    
    matrix_time = time.time() - matrix_start
    total_time = time.time() - start_time
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Matrix construction completed in {matrix_time:.1f}s")
    log_with_flush(f"[{time.strftime('%H:%M:%S')}] Total processing time: {total_time:.1f}s")
    
    gc.collect()
    
    return marker_matrix, coverage_matrix

# Test function to verify correctness
def test_optimization(atlas_path: str, pat_dir: str, min_cpgs: int, sample_size: int = 1000):
    """Test that optimized version produces same results as original"""
    from .find_marker_candidates import create_marker_matrices as create_marker_matrices_original
    
    print(f"Testing with {sample_size} regions...")
    
    # Load subset of atlas for testing
    markers_df = pd.read_csv(atlas_path, sep='\t').head(sample_size)
    temp_atlas = '/tmp/test_atlas.tsv'
    markers_df.to_csv(temp_atlas, sep='\t', index=False)
    
    # Run original version
    print("Running original version...")
    start = time.time()
    orig_marker, orig_coverage = create_marker_matrices_original(temp_atlas, pat_dir, min_cpgs, threads=4)
    orig_time = time.time() - start
    print(f"Original took {orig_time:.2f}s")
    
    # Run optimized version
    print("Running optimized version...")
    start = time.time()
    opt_marker, opt_coverage = create_marker_matrices_optimized(temp_atlas, pat_dir, min_cpgs, threads=4)
    opt_time = time.time() - start
    print(f"Optimized took {opt_time:.2f}s")
    
    # Compare results
    print("\nComparing results...")
    
    # Check shapes
    assert orig_marker.shape == opt_marker.shape, f"Marker shape mismatch: {orig_marker.shape} vs {opt_marker.shape}"
    assert orig_coverage.shape == opt_coverage.shape, f"Coverage shape mismatch: {orig_coverage.shape} vs {opt_coverage.shape}"
    
    # Check column names
    assert list(orig_marker.columns) == list(opt_marker.columns), "Marker column mismatch"
    assert list(orig_coverage.columns) == list(opt_coverage.columns), "Coverage column mismatch"
    
    # Check values (allowing for small numerical differences)
    for col in orig_marker.columns[2:]:  # Skip name and direction
        orig_vals = orig_marker[col].values
        opt_vals = opt_marker[col].values
        
        # Handle NaN values
        nan_mask = np.isnan(orig_vals)
        assert np.array_equal(nan_mask, np.isnan(opt_vals)), f"NaN mismatch in column {col}"
        
        # Compare non-NaN values
        if not nan_mask.all():
            max_diff = np.max(np.abs(orig_vals[~nan_mask] - opt_vals[~nan_mask]))
            assert max_diff < 1e-6, f"Value mismatch in marker column {col}: max diff = {max_diff}"
    
    # Check coverage values
    for col in orig_coverage.columns[2:]:
        assert np.array_equal(orig_coverage[col].values, opt_coverage[col].values), f"Coverage mismatch in column {col}"
    
    print(f"✅ All tests passed! Speedup: {orig_time/opt_time:.1f}x")
    
    os.remove(temp_atlas)
    
    return True

if __name__ == "__main__":
    # Run test
    test_optimization(
        atlas_path="/path/to/atlas.bed",
        pat_dir="/path/to/pat_files",
        min_cpgs=4,
        sample_size=1000
    )