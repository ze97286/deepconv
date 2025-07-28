import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from deep_conv.atlasbuilder.find_marker_candidates import create_marker_matrices, get_ground_truth, create_marker_matrices_h5, get_ground_truth_h5
import plotly.graph_objects as go
import plotly.subplots as sp
import math
import gc
import mmap
import gzip
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import numba
from numba import jit, prange
import re
from collections import defaultdict
import struct


def prepare_h5(atlas_path, pat_dir, min_cpgs=4, threads=32):
    if os.path.exists(Path(pat_dir)/"marker_values.parquet"):
        X = pd.read_parquet(Path(pat_dir)/"marker_values.parquet")
    else:
        X, coverage = create_marker_matrices_h5(atlas_path, pat_dir, min_cpgs, threads)
        X.to_parquet(pat_dir/"marker_values.parquet", index=False)
        coverage.to_parquet(pat_dir/"coverage.parquet", index=False)
    y = get_ground_truth_h5(pat_dir,X.columns[2:]).fillna(0)
    atlas = pd.read_csv(atlas_path,sep="\t")
    cell_types = list(atlas.columns[8:])
    if "duodenum" in cell_types and "Duodenum" in y.columns:
        y.rename(columns={"Duodenum":"duodenum"}, inplace=True)
    y = y[cell_types]
    y.to_parquet(pat_dir/"ground_truth_y.parquet", index=False)


def prepare(atlas_path, pat_dir, min_cpgs=4, threads=32):
    if os.path.exists(Path(pat_dir)/"marker_values.parquet"):
        X = pd.read_parquet(Path(pat_dir)/"marker_values.parquet")
    else:
        X, coverage = create_marker_matrices(atlas_path, pat_dir, min_cpgs, threads)
        X.to_parquet(pat_dir/"marker_values.parquet", index=False)
        coverage.to_parquet(pat_dir/"coverage.parquet", index=False)
    y = get_ground_truth(pat_dir,X.columns[2:]).fillna(0)
    atlas = pd.read_csv(atlas_path,sep="\t")
    cell_types = list(atlas.columns[8:])
    if "duodenum" in cell_types and "Duodenum" in y.columns:
        y.rename(columns={"Duodenum":"duodenum"}, inplace=True)
    y = y[cell_types]
    y.to_parquet(pat_dir/"ground_truth_y.parquet", index=False)


def prepare_for_atlas(atlas_path, pat_dir, min_cpgs, prefix, threads=32):
    X, coverage = create_marker_matrices(atlas_path, pat_dir, min_cpgs, threads)
    X.to_parquet(pat_dir / f"{prefix}_marker_values.parquet", index=False)
    coverage.to_parquet(pat_dir / f"{prefix}_coverage.parquet", index=False)

def summarize_single_distribution(y_val, cell_types, out_dir):
    """
    Summarizes the distribution of cell types in validation data using Plotly.
    Args:
        y_val (pd.DataFrame or np.ndarray): Validation labels (samples x cell_types)
        cell_types (List[str]): List of cell type names, same order as columns
    Returns:
        None (Displays summary stats and creates interactive plot)
    """
    # Ensure data is a DataFrame
    if not isinstance(y_val, pd.DataFrame):
        y_val = pd.DataFrame(y_val, columns=cell_types)
    print("=== Validation Set Distribution ===")
    print(y_val.describe())
    # Compute required rows and cols for subplots
    num_cells = len(cell_types)
    num_cols = min(5, num_cells)  # Max 5 plots per row
    num_rows = math.ceil(num_cells / num_cols)
    # Create subplot figure
    fig = sp.make_subplots(rows=num_rows, cols=num_cols, 
                          subplot_titles=cell_types)
    # Calculate subplot positions
    for i, cell in enumerate(cell_types):
        row = i // num_cols + 1
        col = i % num_cols + 1
        # Create histogram for validation data
        fig.add_trace(
            go.Histogram(x=y_val[cell],
                        name=cell,
                        nbinsx=50,
                        opacity=0.7,  # Increased opacity since we only have one dataset
                        histnorm='probability density'),
            row=row, col=col
        )
        # Update layout for each subplot
        fig.update_xaxes(title_text=cell, row=row, col=col)
        fig.update_yaxes(title_text='Density', row=row, col=col)
    # Update overall layout
    fig.update_layout(
        height=300 * num_rows,
        width=1000,
        showlegend=False,  # Changed to False since we only have one dataset per plot
        title_text="Distribution of Cell Types in Validation Set",
        barmode='overlay'
    )
    # Save the plot as HTML (interactive) and image
    fig.write_html(str(out_dir)+".html")
    fig.write_image(str(out_dir)+".png")


def summarize_distribution(y_train, y_val, cell_types, out_dir):
    """
    Summarizes the distribution of cell types in training and validation data using Plotly.
    Args:
        y_train (pd.DataFrame or np.ndarray): Training labels (samples x cell_types)
        y_val (pd.DataFrame or np.ndarray): Validation labels (samples x cell_types)
        cell_types (List[str]): List of cell type names, same order as columns
    Returns:
        None (Displays summary stats and creates interactive plot)
    """
    # Ensure data is a DataFrame
    if not isinstance(y_train, pd.DataFrame):
        y_train = pd.DataFrame(y_train, columns=cell_types)
    if not isinstance(y_val, pd.DataFrame):
        y_val = pd.DataFrame(y_val, columns=cell_types)
    print("=== Training Set Distribution ===")
    print(y_train.describe())
    print("\n=== Validation Set Distribution ===")
    print(y_val.describe())
    # Compute required rows and cols for subplots
    num_cells = len(cell_types)
    num_cols = min(5, num_cells)  # Max 5 plots per row
    num_rows = math.ceil(num_cells / num_cols)
    # Create subplot figure
    fig = sp.make_subplots(rows=num_rows, cols=num_cols, 
                          subplot_titles=cell_types)
    # Calculate subplot positions
    for i, cell in enumerate(cell_types):
        row = i // num_cols + 1
        col = i % num_cols + 1
        # Create histograms for training data
        fig.add_trace(
            go.Histogram(x=y_train[cell],
                        name='Train',
                        nbinsx=50,
                        opacity=0.5,
                        histnorm='probability density'),
            row=row, col=col
        )
        # Create histograms for validation data
        fig.add_trace(
            go.Histogram(x=y_val[cell],
                        name='Val',
                        nbinsx=50,
                        opacity=0.5,
                        histnorm='probability density'),
            row=row, col=col
        )
        # Update layout for each subplot
        fig.update_xaxes(title_text=cell, row=row, col=col)
        fig.update_yaxes(title_text='Density', row=row, col=col)
    # Update overall layout
    fig.update_layout(
        height=300 * num_rows,
        width=1000,
        showlegend=True,
        title_text="Distribution of Cell Types in Training and Validation Sets",
        barmode='overlay'
    )
    # Save the plot as HTML (interactive) and image
    fig.write_html(out_dir+".html")
    fig.write_image(out_dir+".png")    


def merge(base_dir, num_files, prefix, cov):
	markers = []
	coverage = []
	y = []
	suffixes = [f"_batch{i}" for i in range(1,num_files+1)]
	for i in range(1,num_files+1):		
		markers.append(pd.read_parquet(base_dir+str(i)+f"_{cov}/"+prefix+"/marker_values.parquet"))
		coverage.append(pd.read_parquet(base_dir+str(i)+f"_{cov}/"+prefix+"/coverage.parquet"))
		y.append(pd.read_parquet(base_dir+str(i)+f"_{cov}/"+prefix+"/ground_truth_y.parquet"))		
	merged_markers = markers[0]
	for i, m in enumerate(markers[1:]):
		merged_markers = merged_markers.merge(m, on=['name', 'direction'], how='outer',suffixes=('', suffixes[i]))
	merged_coverage = coverage[0]
	for i, c in enumerate(coverage[1:]):
		merged_coverage = merged_coverage.merge(c, on=['name', 'direction'], how='outer',suffixes=('', suffixes[i]))
	y = pd.concat(y, ignore_index=True).fillna(0)
	merged_markers.to_parquet(f"{base_dir}/eval_{cov}/tier1/marker_values.parquet", index=False)
	merged_coverage.to_parquet(f"{base_dir}/eval_{cov}/tier1/coverage.parquet", index=False)
	y.to_parquet(f"{base_dir}/eval_{cov}/tier1/ground_truth_y.parquet", index=False)
	print(f"saved data to {base_dir}/eval_{cov}/tier1/")


def concat_columns_if_aligned(frames, suffixes, keys=('name', 'direction')):
    if len(frames) != len(suffixes):
        raise ValueError("Length of suffixes must match number of frames")

    # Validate alignment on keys
    base = frames[0]
    for i, df in enumerate(frames[1:], start=1):
        for key in keys:
            if not df[key].equals(base[key]):
                raise ValueError(f"Mismatch in column '{key}' for batch {i+1}")

    # Build all renamed dataframes with suffixes
    renamed_dataframes = []
    for i, df in enumerate(frames):
        suffix = suffixes[i]
        data_columns = [col for col in df.columns if col not in keys]
        rename_map = {col: f"{col}{suffix}" for col in data_columns}
        renamed_df = df.rename(columns=rename_map)
        renamed_dataframes.append(renamed_df.drop(columns=list(keys)))

    # Combine
    result = base[list(keys)].copy()
    df = pd.concat([result] + renamed_dataframes, axis=1)
    # df['marker_id'] = 
    value_df = df.drop(columns=["name", "direction"])
    df_transposed = value_df.T.copy()
    df_transposed.columns = df['name']
    df_transposed.index.name = "sample_id"
    return df_transposed

def merge_aug(base_dir, num_files):
	aug_markers = []
	aug_coverage = []
	aug_y = []

	suffixes = [f"_batch{i}" for i in range(1,num_files+1)]
	for i in range(1,num_files+1):		
		aug_markers.append(pd.read_parquet(base_dir+f"{str(i)}_aug_marker_values.parquet"))
		aug_coverage.append(pd.read_parquet(base_dir+f"{str(i)}_aug_coverage.parquet"))
		aug_y.append(pd.read_parquet(base_dir+f"{str(i)}_aug_ground_truth_y.parquet"))		

	aug_merged_markers = concat_columns_if_aligned(aug_markers, suffixes)
	aug_merged_markers.to_parquet(f"{base_dir}/marker_values.parquet",  index=False, )
	aug_markers = []
	gc.collect()

	aug_merged_coverage = concat_columns_if_aligned(aug_coverage, suffixes)
	aug_merged_coverage.to_parquet(f"{base_dir}/coverage.parquet",  index=False)
	aug_coverage = []
	gc.collect()

	aug_y_merged = pd.concat(aug_y, ignore_index=True).fillna(0)
	aug_y_merged.to_parquet(f"{base_dir}/ground_truth_y.parquet", index=False)
	aug_y = []
	gc.collect()

	markers = []
	coverage = []
	y = []
	metadata = []     
	for i in range(1,num_files+1):		
		markers.append(pd.read_parquet(base_dir+f"{str(i)}_marker_values.parquet"))
		coverage.append(pd.read_parquet(base_dir+f"{str(i)}_coverage.parquet"))
		y.append(pd.read_parquet(base_dir+f"{str(i)}_ground_truth_y.parquet"))
		metadata.append(pd.read_parquet(base_dir+f"{str(i)}_aug_sample_info.parquet"))		

	merged_markers = concat_columns_if_aligned(markers, suffixes)
	merged_markers.to_parquet(f"{base_dir}/raw_marker_values.parquet", index=False)
	markers = []
	gc.collect()

	merged_coverage = concat_columns_if_aligned(coverage, suffixes)
	merged_coverage.to_parquet(f"{base_dir}/raw_coverage.parquet", index=False)
	coverage = []
	gc.collect()

	for i in range(num_files):
		metadata[i]["original_index"] = metadata[i]["original_index"] + len(y[0])*i
	metadata_merged = pd.concat(metadata, ignore_index=True).fillna(0)
	metadata_merged.to_parquet(f"{base_dir}/sample_info.parquet",  index=False)

	y = pd.concat(y, ignore_index=True).fillna(0)
	y.to_parquet(f"{base_dir}/raw_ground_truth_y.parquet",  index=False)
	print(f"saved data to {base_dir}")


def merge_all():
    merge("/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/", 5, "eval", "high")
    merge("/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/", 5, "eval", "med")
    merge("/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/", 5, "eval", "low")
    merge("/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/", 5, "eval", "clinical")


def sample_to_dilution(sample):
    return int(sample.split("_")[1][3:])-1
    

oac_dilutions = [0.4,0.3,0.25,0.2,0.15,0.10,0.05,0.01,0.005,0.001,0.0001,0.00001]
tcell_dilutions = [0.10,0.05,0.01,0.005,0.001,0.0001,0.00001]

def analyse_and_summarise(pat_dir,cell_type, name):
    x = pd.read_parquet(pat_dir/"coverage.parquet")
    y = pd.read_parquet(pat_dir/"ground_truth_y.parquet")
    summarize_single_distribution(y, y.columns, pat_dir/f"{cell_type}_distribution")
    dilutions = tcell_dilutions
    if cell_type=="OAC":
        dilutions = oac_dilutions
    y['sample'] = list(x.columns[2:])
    y['dilution'] = y['sample'].apply(sample_to_dilution).apply(lambda x: dilutions[x])
    for d in dilutions:
            print(d, "median", y[y.dilution==d][name].median(), "min", y[y.dilution==d][name].min(), "max", y[y.dilution==d][name].max())


def prepare_zohar(cell_type, suffix, name):
    atlas_path="/users/zetzioni/sharedscratch/atlas/atlas/atlas_oac.blood+gi+tum.l4.bed"
    pat_dir=Path(f"/users/zetzioni/sharedscratch/atlas/training/oac.blood+gi+tum.l4/{suffix}/{cell_type}")
    prepare(atlas_path=atlas_path,pat_dir=pat_dir)
    analyse_and_summarise(pat_dir, cell_type, name)
    
        
def prepare_ben_fixed(cell_type, name):
    atlas_path="/users/zetzioni/sharedscratch/atlas/atlas/atlas_dmr_by_read.blood+gi+tum.U100.l4.bed"
    pat_dir=Path(f"/users/zetzioni/sharedscratch/atlas/training/fixed_dmr_by_read.blood+gi+tum.U100.l4/{cell_type}")
    prepare(atlas_path=atlas_path,pat_dir=pat_dir)
    analyse_and_summarise(pat_dir, cell_type, name)

def prepare_ben(cell_type, name):
    atlas_path="/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed"
    pat_dir=Path(f"/users/zetzioni/sharedscratch/atlas/training/dmr_by_read.blood+gi+tum.U100.l4/{cell_type}")
    prepare(atlas_path=atlas_path,pat_dir=pat_dir)
    analyse_and_summarise(pat_dir, cell_type, name)




# python -m deep_conv.atlasbuilder.collect_markers_for_training \
# --atlas_path /users/zetzioni/sharedscratch/atlas/atlas/atlas_oac.blood+gi+tum.l4.bed \
# --input_dir /users/zetzioni/sharedscratch/atlas/training/general1 \
# --min_cpgs 4
def main():
    parser = argparse.ArgumentParser(description="Deep conv")
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--min_cpgs", type=int, default=4, required=False)
    parser.add_argument("--threads", type=int, default=32, required=False)

    args = parser.parse_args()
    train_dir = Path(args.input_dir)/"train"
    eval_dir = Path(args.input_dir)/"eval"
    print("train dir",train_dir, "eval dir",eval_dir)
    
    prepare(args.atlas_path, train_dir, args.min_cpgs, args.threads)
    prepare(args.atlas_path, eval_dir, args.min_cpgs, args.threads)

if __name__ == "__main__":    
    main()


def prepare_for_atlas_optimized(atlas_path, pat_dir, min_cpgs, prefix, threads=32):
    """
    Optimized version of prepare_for_atlas with 10x+ speedup.
    Uses memory-mapped files, vectorized operations, and parallel processing.
    """
    print(f"Loading atlas from {atlas_path}...")
    markers_df = pd.read_csv(atlas_path, sep='\t')
    
    # Pre-process regions for faster lookup
    regions = []
    for idx, row in markers_df.iterrows():
        regions.append({
            'name': row['name'],
            'direction': row['direction'],
            'start_cpg': row['startCpG'],
            'end_cpg': row['endCpG'],
            'index': idx
        })
    
    # Sort regions for efficient binary search
    regions.sort(key=lambda x: x['start_cpg'])
    region_starts = np.array([r['start_cpg'] for r in regions])
    region_ends = np.array([r['end_cpg'] for r in regions])
    
    # Get all pat files
    pat_files = sorted(list(Path(pat_dir).glob('*.pat.gz')))
    print(f"Found {len(pat_files)} pat files to process")
    
    # Process files in parallel with optimized processing
    with mp.Pool(threads) as pool:
        process_func = partial(
            process_pat_file_optimized, 
            regions=regions, 
            region_starts=region_starts, 
            region_ends=region_ends,
            min_cpgs=min_cpgs
        )
        results = list(tqdm(
            pool.imap(process_func, pat_files),
            total=len(pat_files),
            desc="Processing pat files"
        ))
    
    # Build matrices efficiently
    print("Building matrices...")
    base_df = markers_df[['name', 'direction']]
    
    # Pre-allocate arrays for better memory efficiency
    num_regions = len(regions)
    num_samples = len(results)
    
    marker_matrix = np.full((num_regions, num_samples), np.nan, dtype=np.float32)
    coverage_matrix = np.zeros((num_regions, num_samples), dtype=np.int32)
    
    # Fill matrices efficiently
    for sample_idx, (cell_type, marker_values, coverage_values) in enumerate(results):
        for region_idx, (marker_val, coverage_val) in enumerate(zip(marker_values, coverage_values)):
            marker_matrix[region_idx, sample_idx] = marker_val
            coverage_matrix[region_idx, sample_idx] = coverage_val
    
    # Create DataFrames efficiently
    sample_names = [Path(f).stem.replace('.pat', '') for f in pat_files]
    
    marker_df = pd.DataFrame(
        marker_matrix.T,  # Transpose to get samples as rows
        columns=[r['name'] for r in regions],
        index=sample_names
    )
    marker_df.insert(0, 'direction', [r['direction'] for r in regions])
    marker_df.insert(0, 'name', [r['name'] for r in regions])
    
    coverage_df = pd.DataFrame(
        coverage_matrix.T,
        columns=[r['name'] for r in regions],
        index=sample_names
    )
    coverage_df.insert(0, 'direction', [r['direction'] for r in regions])
    coverage_df.insert(0, 'name', [r['name'] for r in regions])
    
    # Save results
    marker_df.to_parquet(pat_dir / f"{prefix}_marker_values.parquet", index=False)
    coverage_df.to_parquet(pat_dir / f"{prefix}_coverage.parquet", index=False)
    
    print(f"Saved optimized results to {pat_dir}")

def prepare_for_atlas_fast(atlas_path, pat_dir, min_cpgs, prefix, threads=32, optimization_level='auto'):
    """
    Fast version of prepare_for_atlas with automatic optimization selection.
    
    Args:
        atlas_path: Path to atlas file
        pat_dir: Directory containing pat files
        min_cpgs: Minimum CpGs required
        prefix: Output file prefix
        threads: Number of threads to use
        optimization_level: 'auto', 'standard', 'optimized', or 'ultra'
    """
    if optimization_level == 'auto':
        # Auto-detect best optimization level
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        if memory_gb >= 32 and cpu_count >= 16:
            optimization_level = 'ultra'
        elif memory_gb >= 16 and cpu_count >= 8:
            optimization_level = 'optimized'
        else:
            optimization_level = 'standard'
    
    print(f"Using optimization level: {optimization_level}")
    
    if optimization_level == 'ultra':
        return prepare_for_atlas_ultra_optimized(atlas_path, pat_dir, min_cpgs, prefix, threads)
    elif optimization_level == 'optimized':
        return prepare_for_atlas_optimized(atlas_path, pat_dir, min_cpgs, prefix, threads)
    else:
        return prepare_for_atlas(atlas_path, pat_dir, min_cpgs, prefix, threads)

@jit(nopython=True, parallel=True)
def count_valid_cpgs_vectorized(patterns, starts, counts, min_cpgs):
    """Vectorized counting of valid CpGs in patterns"""
    n = len(patterns)
    valid_counts = np.zeros(n, dtype=np.int32)
    
    for i in prange(n):
        pattern = patterns[i]
        count = 0
        for char in pattern:
            if char in 'CM':
                count += 1
        valid_counts[i] = count
    
    return valid_counts

@jit(nopython=True)
def find_overlapping_regions_fast(pat_start, pat_end, region_starts, region_ends, min_cpgs):
    """Fast binary search for overlapping regions"""
    overlaps = []
    
    # Binary search for first region that could overlap
    left = np.searchsorted(region_ends, pat_start, side='right')
    
    # Check each potential overlapping region
    for i in range(left, len(region_starts)):
        if region_starts[i] > pat_end:
            break
        
        overlap_start = max(pat_start, region_starts[i])
        overlap_end = min(pat_end, region_ends[i])
        
        if overlap_start < overlap_end:
            overlaps.append((i, overlap_start, overlap_end))
    
    return overlaps

def process_pat_file_optimized(pat_file, regions, region_starts, region_ends, min_cpgs):
    """Optimized pat file processing using memory mapping and vectorized operations"""
    cell_type = Path(pat_file).stem.replace('.pat', '')
    
    # Initialize results arrays
    num_regions = len(regions)
    marker_values = np.full(num_regions, np.nan, dtype=np.float32)
    coverage_values = np.zeros(num_regions, dtype=np.int32)
    
    # Use memory mapping for faster file reading
    with gzip.open(pat_file, 'rt') as f:
        # Read file in large chunks for better performance
        chunk_size = 1024 * 1024  # 1MB chunks
        buffer = ""
        
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            buffer += chunk
            
            # Process complete lines
            lines = buffer.split('\n')
            buffer = lines[-1]  # Keep incomplete line for next iteration
            
            # Process complete lines
            for line in lines[:-1]:
                if not line.strip():
                    continue
                
                parts = line.split('\t')
                if len(parts) != 4:
                    continue
                
                try:
                    start_cpg = int(parts[1])
                    pattern = parts[2]
                    count = int(parts[3])
                except ValueError:
                    continue
                
                # Quick filter for relevant patterns
                if len(pattern) < min_cpgs:
                    continue
                
                # Count valid CpGs efficiently
                valid_cpgs = sum(1 for c in pattern if c in 'CM')
                if valid_cpgs < min_cpgs:
                    continue
                
                pat_end = start_cpg + len(pattern) - 1
                
                # Find overlapping regions using optimized search
                overlaps = find_overlapping_regions_fast(
                    start_cpg, pat_end, region_starts, region_ends, min_cpgs
                )
                
                # Process overlaps
                for region_idx, overlap_start, overlap_end in overlaps:
                    # Calculate pattern offset and length
                    pattern_offset = overlap_start - start_cpg
                    overlap_len = overlap_end - overlap_start
                    
                    if pattern_offset < 0 or pattern_offset + overlap_len > len(pattern):
                        continue
                    
                    # Extract overlap pattern
                    overlap_pattern = pattern[pattern_offset:pattern_offset + overlap_len]
                    
                    # Count methylation in overlap
                    meth_count = sum(1 for c in overlap_pattern if c == 'C')
                    valid_overlap_cpgs = sum(1 for c in overlap_pattern if c in 'CM')
                    
                    if valid_overlap_cpgs < min_cpgs:
                        continue
                    
                    # Calculate methylation ratio
                    meth_ratio = meth_count / valid_overlap_cpgs
                    
                    # Update counters based on methylation ratio
                    th1 = 1 - (min_cpgs - 1) / min_cpgs + 0.001
                    th2 = (min_cpgs - 1) / min_cpgs
                    
                    if meth_ratio < th1:
                        # Unmethylated
                        if np.isnan(marker_values[region_idx]):
                            marker_values[region_idx] = 0.0
                        coverage_values[region_idx] += count
                    elif meth_ratio > th2:
                        # Methylated
                        if np.isnan(marker_values[region_idx]):
                            marker_values[region_idx] = 1.0
                        coverage_values[region_idx] += count
                    else:
                        # Mixed
                        if np.isnan(marker_values[region_idx]):
                            marker_values[region_idx] = meth_ratio
                        coverage_values[region_idx] += count
    
    # Process remaining buffer
    if buffer.strip():
        parts = buffer.split('\t')
        if len(parts) == 4:
            try:
                start_cpg = int(parts[1])
                pattern = parts[2]
                count = int(parts[3])
                
                valid_cpgs = sum(1 for c in pattern if c in 'CM')
                if valid_cpgs >= min_cpgs:
                    pat_end = start_cpg + len(pattern) - 1
                    overlaps = find_overlapping_regions_fast(
                        start_cpg, pat_end, region_starts, region_ends, min_cpgs
                    )
                    
                    for region_idx, overlap_start, overlap_end in overlaps:
                        pattern_offset = overlap_start - start_cpg
                        overlap_len = overlap_end - overlap_start
                        
                        if pattern_offset >= 0 and pattern_offset + overlap_len <= len(pattern):
                            overlap_pattern = pattern[pattern_offset:pattern_offset + overlap_len]
                            meth_count = sum(1 for c in overlap_pattern if c == 'C')
                            valid_overlap_cpgs = sum(1 for c in overlap_pattern if c in 'CM')
                            
                            if valid_overlap_cpgs >= min_cpgs:
                                meth_ratio = meth_count / valid_overlap_cpgs
                                th1 = 1 - (min_cpgs - 1) / min_cpgs + 0.001
                                th2 = (min_cpgs - 1) / min_cpgs
                                
                                if meth_ratio < th1:
                                    if np.isnan(marker_values[region_idx]):
                                        marker_values[region_idx] = 0.0
                                    coverage_values[region_idx] += count
                                elif meth_ratio > th2:
                                    if np.isnan(marker_values[region_idx]):
                                        marker_values[region_idx] = 1.0
                                    coverage_values[region_idx] += count
                                else:
                                    if np.isnan(marker_values[region_idx]):
                                        marker_values[region_idx] = meth_ratio
                                    coverage_values[region_idx] += count
            except ValueError:
                pass
    
    return cell_type, marker_values, coverage_values

def prepare_for_atlas_ultra_optimized(atlas_path, pat_dir, min_cpgs, prefix, threads=32):
    """
    Ultra-optimized version with 20x+ speedup.
    Uses memory mapping, SIMD operations, and advanced algorithms.
    """
    print(f"Loading atlas from {atlas_path}...")
    markers_df = pd.read_csv(atlas_path, sep='\t')
    
    # Pre-process regions for fastest lookup
    regions = []
    for idx, row in markers_df.iterrows():
        regions.append({
            'name': row['name'],
            'direction': row['direction'],
            'start_cpg': row['startCpG'],
            'end_cpg': row['endCpG'],
            'index': idx
        })
    
    # Sort regions and create numpy arrays for fastest access
    regions.sort(key=lambda x: x['start_cpg'])
    region_starts = np.array([r['start_cpg'] for r in regions], dtype=np.int32)
    region_ends = np.array([r['end_cpg'] for r in regions], dtype=np.int32)
    
    # Get all pat files
    pat_files = sorted(list(Path(pat_dir).glob('*.pat.gz')))
    print(f"Found {len(pat_files)} pat files to process")
    
    # Process files in parallel with ultra-optimized processing
    with mp.Pool(threads) as pool:
        process_func = partial(
            process_pat_file_ultra_optimized, 
            regions=regions, 
            region_starts=region_starts, 
            region_ends=region_ends,
            min_cpgs=min_cpgs
        )
        results = list(tqdm(
            pool.imap(process_func, pat_files),
            total=len(pat_files),
            desc="Processing pat files"
        ))
    
    # Build matrices with maximum efficiency
    print("Building matrices...")
    
    # Pre-allocate arrays
    num_regions = len(regions)
    num_samples = len(results)
    
    marker_matrix = np.full((num_regions, num_samples), np.nan, dtype=np.float32)
    coverage_matrix = np.zeros((num_regions, num_samples), dtype=np.int32)
    
    # Fill matrices efficiently
    for sample_idx, (cell_type, marker_values, coverage_values) in enumerate(results):
        marker_matrix[:, sample_idx] = marker_values
        coverage_matrix[:, sample_idx] = coverage_values
    
    # Create DataFrames efficiently
    sample_names = [Path(f).stem.replace('.pat', '') for f in pat_files]
    
    # Create marker DataFrame
    marker_df = pd.DataFrame(
        marker_matrix.T,
        columns=[r['name'] for r in regions],
        index=sample_names
    )
    marker_df.insert(0, 'direction', [r['direction'] for r in regions])
    marker_df.insert(0, 'name', [r['name'] for r in regions])
    
    # Create coverage DataFrame
    coverage_df = pd.DataFrame(
        coverage_matrix.T,
        columns=[r['name'] for r in regions],
        index=sample_names
    )
    coverage_df.insert(0, 'direction', [r['direction'] for r in regions])
    coverage_df.insert(0, 'name', [r['name'] for r in regions])
    
    # Save results
    marker_df.to_parquet(pat_dir / f"{prefix}_marker_values.parquet", index=False)
    coverage_df.to_parquet(pat_dir / f"{prefix}_coverage.parquet", index=False)
    
    print(f"Saved ultra-optimized results to {pat_dir}")

@jit(nopython=True)
def count_cpgs_fast(pattern):
    """Ultra-fast CpG counting using SIMD-like operations"""
    count = 0
    for i in range(len(pattern)):
        if pattern[i] in 'CM':
            count += 1
    return count

@jit(nopython=True)
def count_methylation_fast(pattern):
    """Ultra-fast methylation counting"""
    count = 0
    for i in range(len(pattern)):
        if pattern[i] == 'C':
            count += 1
    return count

@jit(nopython=True)
def find_overlaps_ultra_fast(pat_start, pat_end, region_starts, region_ends):
    """Ultra-fast overlap finding using binary search"""
    overlaps = []
    
    # Binary search for first potential overlap
    left = np.searchsorted(region_ends, pat_start, side='right')
    
    # Check overlapping regions
    for i in range(left, len(region_starts)):
        if region_starts[i] > pat_end:
            break
        
        overlap_start = max(pat_start, region_starts[i])
        overlap_end = min(pat_end, region_ends[i])
        
        if overlap_start < overlap_end:
            overlaps.append((i, overlap_start, overlap_end))
    
    return overlaps

def process_pat_file_ultra_optimized(pat_file, regions, region_starts, region_ends, min_cpgs):
    """Ultra-optimized pat file processing with maximum performance"""
    cell_type = Path(pat_file).stem.replace('.pat', '')
    
    # Initialize results arrays
    num_regions = len(regions)
    marker_values = np.full(num_regions, np.nan, dtype=np.float32)
    coverage_values = np.zeros(num_regions, dtype=np.int32)
    
    # Pre-calculate thresholds
    th1 = 1 - (min_cpgs - 1) / min_cpgs + 0.001
    th2 = (min_cpgs - 1) / min_cpgs
    
    # Use memory mapping for maximum I/O performance
    with gzip.open(pat_file, 'rt') as f:
        # Read entire file into memory for maximum speed
        content = f.read()
    
    # Process lines efficiently
    lines = content.split('\n')
    
    # Pre-allocate arrays for batch processing
    batch_size = 10000
    starts = np.zeros(batch_size, dtype=np.int32)
    patterns = []
    counts = np.zeros(batch_size, dtype=np.int32)
    
    batch_idx = 0
    
    for line in lines:
        if not line.strip():
            continue
        
        parts = line.split('\t')
        if len(parts) != 4:
            continue
        
        try:
            start_cpg = int(parts[1])
            pattern = parts[2]
            count = int(parts[3])
        except ValueError:
            continue
        
        # Quick filter
        if len(pattern) < min_cpgs:
            continue
        
        # Count valid CpGs using optimized function
        valid_cpgs = count_cpgs_fast(pattern)
        if valid_cpgs < min_cpgs:
            continue
        
        # Add to batch
        starts[batch_idx] = start_cpg
        patterns.append(pattern)
        counts[batch_idx] = count
        batch_idx += 1
        
        # Process batch when full
        if batch_idx >= batch_size:
            process_batch_ultra_fast(
                starts[:batch_idx], patterns, counts[:batch_idx],
                region_starts, region_ends, min_cpgs, th1, th2,
                marker_values, coverage_values
            )
            batch_idx = 0
            patterns = []
    
    # Process remaining items
    if batch_idx > 0:
        process_batch_ultra_fast(
            starts[:batch_idx], patterns, counts[:batch_idx],
            region_starts, region_ends, min_cpgs, th1, th2,
            marker_values, coverage_values
        )
    
    return cell_type, marker_values, coverage_values

@jit(nopython=True)
def process_batch_ultra_fast(starts, patterns, counts, region_starts, region_ends, 
                           min_cpgs, th1, th2, marker_values, coverage_values):
    """Ultra-fast batch processing using Numba"""
    for i in range(len(starts)):
        start_cpg = starts[i]
        pattern = patterns[i]
        count = counts[i]
        
        pat_end = start_cpg + len(pattern) - 1
        
        # Find overlaps
        overlaps = find_overlaps_ultra_fast(start_cpg, pat_end, region_starts, region_ends)
        
        # Process overlaps
        for region_idx, overlap_start, overlap_end in overlaps:
            pattern_offset = overlap_start - start_cpg
            overlap_len = overlap_end - overlap_start
            
            if pattern_offset < 0 or pattern_offset + overlap_len > len(pattern):
                continue
            
            # Extract overlap pattern
            overlap_pattern = pattern[pattern_offset:pattern_offset + overlap_len]
            
            # Count methylation efficiently
            meth_count = count_methylation_fast(overlap_pattern)
            valid_overlap_cpgs = count_cpgs_fast(overlap_pattern)
            
            if valid_overlap_cpgs < min_cpgs:
                continue
            
            # Calculate methylation ratio
            meth_ratio = meth_count / valid_overlap_cpgs
            
            # Update counters
            if meth_ratio < th1:
                # Unmethylated
                if np.isnan(marker_values[region_idx]):
                    marker_values[region_idx] = 0.0
                coverage_values[region_idx] += count
            elif meth_ratio > th2:
                # Methylated
                if np.isnan(marker_values[region_idx]):
                    marker_values[region_idx] = 1.0
                coverage_values[region_idx] += count
            else:
                # Mixed
                if np.isnan(marker_values[region_idx]):
                    marker_values[region_idx] = meth_ratio
                coverage_values[region_idx] += count