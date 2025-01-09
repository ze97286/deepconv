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
        self.counts = defaultdict(lambda: {'u': 0, 'x': 0, 'm': 0})
    def find_overlapping_regions(self, pat_start: int, pat_length: int) -> List[Tuple[Region, int, int]]:
        pat_end = pat_start + pat_length - 1
        overlaps = []
        # Binary search for first potential region
        left, right = 0, len(self.regions) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.regions[mid].end_cpg > pat_start:  # Changed from >= to >
                right = mid - 1
            else:
                left = mid + 1
        # Check all potential overlapping regions
        for region in self.regions[left:]:
            if region.start_cpg > pat_end:
                break
            # Calculate overlap with half-open intervals
            overlap_start = max(pat_start, region.start_cpg)
            overlap_end = min(pat_end + 1, region.end_cpg)  # +1 because pat_end is inclusive
            overlap_length = overlap_end - overlap_start  # Remove +1 since we're using half-open interval
            if overlap_length >= self.min_cpgs:
                overlaps.append((region, overlap_start - pat_start, overlap_length))
        return overlaps
    def process_pattern(self, pattern: str, start_cpg: int, count: int):
        """Process a single pattern and update counts for overlapping regions"""
        if len(pattern) < self.min_cpgs:
            return
        # Find overlapping regions
        overlaps = self.find_overlapping_regions(start_cpg, len(pattern))
        if overlaps:  # If we found any overlaps, increment counter
            self.patterns_counted += 1
        for region, offset, overlap_len in overlaps:
            # Extract overlapping portion
            overlap_pat = pattern[offset:offset + overlap_len]
            # Calculate methylation ratio
            meth_count = overlap_pat.count('C')
            meth_ratio = meth_count / len(overlap_pat)
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
    counter = RegionCounter(regions_df, min_cpgs)
    pat_file = str(pat_file)
    cell_type = Path(pat_file).stem.replace('.pat', '')
    print(f"\nProcessing {cell_type}...")
    column_names = ['chr', 'start', 'pattern', 'count']
    processed_lines = 0
    total_lines = 0
    # Count total lines for progress bar
    with gzip.open(pat_file, 'rt') if pat_file.endswith('.gz') else open(pat_file) as f:
       for line in f:
           if total_lines == 0:  # Check first line
               start = int(line.split('\t')[1])
               if start > counter.last_cpg:
                   print(f"Skipping {cell_type} - all positions after our regions")
                   return create_empty_results(regions_df, cell_type)
           total_lines += 1
    with tqdm(total=total_lines, desc=f"Processing {cell_type}") as pbar:
       for chunk in pd.read_csv(pat_file, sep='\t', names=column_names, chunksize=1_000_000):
           # Filter chunk to only patterns that could overlap our regions
           relevant_chunk = chunk[chunk['start'].between(counter.first_cpg, counter.last_cpg)]
           if len(relevant_chunk) == 0:
               if chunk['start'].min() > counter.last_cpg:
                   pbar.update(total_lines - processed_lines)
                   break  # We've passed our regions
               processed_lines += len(chunk)
               pbar.update(len(chunk))
               continue  # Skip this chunk
           # Process relevant rows
           for _, row in relevant_chunk.iterrows():
               counter.process_pattern(row['pattern'], row['start'], row['count'])
           # Update progress for all rows in chunk, not just relevant ones
           chunk_size = len(chunk)
           processed_lines += chunk_size
           pbar.update(chunk_size)
           # Check if we've passed our regions
           if chunk['start'].max() > counter.last_cpg:
               pbar.update(total_lines - processed_lines)
               break
    print(f"Finished processing {cell_type} ({processed_lines}/{total_lines} patterns)")
    # Convert counts to proportions and create output DataFrames
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
                'cell_type': cell_type  # Add cell_type column
            })
            results_coverage.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': total,
                'cell_type': cell_type  # Add cell_type column
            })
        else:
            results_uxm.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': np.nan,
                'cell_type': cell_type  # Add cell_type column
            })
            results_coverage.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': 0,
                'cell_type': cell_type  # Add cell_type column
            })
    print(f"Counted {counter.patterns_counted} patterns for this pat file")

    return pd.DataFrame(results_uxm), pd.DataFrame(results_coverage), cell_type


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


def find_good_markers(chr, batch_df, cell_types, marker_props, col_mapping, coverage, values_matrix, best_targets_idx, min_signal_threshold, snr_threshold, significance_threshold, output_dir, batch_id):
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
            if cell_types[best_targets_idx[i]] == "CD4-T-cells" or cell_types[best_targets_idx[i]] == "CD8-T-cells":
                print("found good marker for", cell_types[best_targets_idx[i]], "with SNR", metrics['snr'], "value",metrics['target_value'])
            good_markers.append(result)
    if not good_markers:
        return None
    result_df =  pd.DataFrame(good_markers)
    if result_df is None:
        print("couldn't find any markers")
        return None
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
    batch_id=0
    for regions_df in pd.read_csv(regions, sep='\t', chunksize=batch_size):
        t_batch = time.time()
        batch_id+=1
        print(f"Loaded {len(regions_df)} regions")
        pat_files = list(Path(pat_dir).glob('*.pat.gz'))
        print(f"Found {len(pat_files)} pat files in {pat_dir}")
        if not pat_files:
            raise ValueError(f"No .pat.gz files found in {pat_dir}")
        # Process pat files in parallel with progress tracking
        print(f"\nProcessing pat files using {threads} threads...")
        with mp.Pool(threads) as pool:
            process_func = partial(process_pat_file, regions_df, min_cpgs=min_cpgs)
            results = list(tqdm(
                pool.imap(process_func, pat_files),
                total=len(pat_files),
                desc="Overall progress"
            ))
        print("\nBuilding final matrices...")
        debug_region = regions_df.iloc[0]
        print(f"\nTracking region: {debug_region['name']}")
        print(f"CpG range: {debug_region['startCpG']}-{debug_region['endCpG']}")
        # Separate UXM and coverage results
        uxm_dfs = []
        coverage_dfs = []
        cell_types = []
        for uxm_df, coverage_df, cell_type in results:
            uxm_dfs.append(uxm_df)
            coverage_dfs.append(coverage_df)
            cell_types.append(cell_type)
        print(f"UXM values for tracked region:")
        for df, cell_type in zip(uxm_dfs, cell_types):
            matching_row = df[df['name'] == debug_region['name']]
            if not matching_row.empty:
                print(f"{cell_type}: {matching_row['value'].iloc[0]}")
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
            # First merge base_df with current results
            merged = pd.merge(base_df, 
                            df[['name', 'direction', 'value']], 
                            on=['name', 'direction'], 
                            how='left')
            # Then assign to new column
            coverage_matrix[f"{cell_type}_merged"] = merged['value']
        marker_props, coverage = uxm_matrix, coverage_matrix
        col_mapping = {col.split('_')[0]: col for col in marker_props.columns if col not in ['name', 'direction']}
        cell_types = list(col_mapping.keys())
        valid_rows = ~marker_props.iloc[:, 2:].isna().any(axis=1)
        marker_props = marker_props[valid_rows]
        coverage = coverage[valid_rows]
        batch_df = regions_df
        batch_df = batch_df[valid_rows].reset_index(drop=True)
        print(f"Batch {batch_id}: After matrix creation: {len(marker_props)} rows")
        if len(batch_df) == 0:
            print(f"Batch {batch_id}: No valid rows after NaN filtering")
            print("finished batch",batch_id,"in",time.time()-t_batch)    
            continue 
        coverage.index = marker_props.index
        batch_df.index = marker_props.index
        sufficient_coverage = (coverage.iloc[:, 2:] >= min_coverage).all(axis=1)
        marker_props = marker_props[sufficient_coverage].reset_index(drop=True)
        coverage = coverage[sufficient_coverage].reset_index(drop=True)
        batch_df = batch_df[sufficient_coverage].reset_index(drop=True)
        if len(batch_df) == 0:
            print("finished batch",batch_id,"in",time.time()-t_batch)
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

