#!/usr/bin/env python3
"""
Stochastic PAT File Dilution Script

This script dilutes PAT files to simulate different coverage levels for testing
model performance. It uses proper stochastic sampling to preserve biological
realism while reducing statistical power.

Usage:
    python dilute_pat_files.py --input_dir path/to/pat/files --output_dir path/to/output --target_coverage 0.5
"""

import pandas as pd
import numpy as np
import gzip
import os
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import random
from collections import defaultdict, Counter
import multiprocessing as mp
from functools import partial


def expand_pat_entry(pattern, count):
    """
    Expand a PAT entry into individual read observations
    
    Args:
        pattern: methylation pattern string (e.g., "CCTTT")
        count: number of reads with this pattern
    
    Returns:
        List of individual pattern observations
    """
    return [pattern] * count


def stochastic_subsample(patterns, target_fraction):
    """
    Stochastically subsample patterns to target coverage fraction
    
    Args:
        patterns: List of individual pattern observations
        target_fraction: Target coverage as fraction (0.0-1.0)
    
    Returns:
        Subsampled list of patterns
    """
    if target_fraction >= 1.0:
        return patterns
    
    n_total = len(patterns)
    n_target = max(1, int(n_total * target_fraction))  # At least 1 read
    
    # Randomly subsample without replacement
    return random.sample(patterns, n_target)


def aggregate_patterns(patterns):
    """
    Aggregate subsampled patterns back into count format
    
    Args:
        patterns: List of pattern observations
    
    Returns:
        Dictionary mapping pattern -> count
    """
    return Counter(patterns)


def process_pat_chunk(chunk_data, target_fraction, random_seed=None):
    """
    Process a chunk of PAT data with stochastic dilution
    
    Args:
        chunk_data: List of tuples (chr, start, pattern, count)
        target_fraction: Target coverage fraction
        random_seed: Random seed for reproducibility
    
    Returns:
        List of diluted PAT entries
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    diluted_entries = []
    
    for chr_name, start, pattern, count in chunk_data:
        # Expand to individual observations
        individual_patterns = expand_pat_entry(pattern, count)
        
        # Stochastically subsample
        subsampled_patterns = stochastic_subsample(individual_patterns, target_fraction)
        
        # Aggregate back to counts
        pattern_counts = aggregate_patterns(subsampled_patterns)
        
        # Create new PAT entries
        for diluted_pattern, diluted_count in pattern_counts.items():
            diluted_entries.append((chr_name, start, diluted_pattern, diluted_count))
    
    return diluted_entries


def dilute_pat_file(input_file, output_file, target_fraction, chunk_size=100000, random_seed=None):
    """
    Dilute a single PAT file to target coverage fraction
    
    Args:
        input_file: Path to input PAT file
        output_file: Path to output diluted PAT file
        target_fraction: Target coverage as fraction (0.0-1.0)
        chunk_size: Number of entries to process per chunk
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Determine if input is gzipped
    is_gzipped = str(input_file).endswith('.gz')
    
    # Open files
    if is_gzipped:
        input_handle = gzip.open(input_file, 'rt')
    else:
        input_handle = open(input_file, 'r')
    
    # Always output as gzipped
    output_handle = gzip.open(output_file, 'wt')
    
    try:
        # Process in chunks for memory efficiency
        chunk_data = []
        total_processed = 0
        total_original_reads = 0
        total_diluted_reads = 0
        
        # Get file size for progress bar
        file_size = os.path.getsize(input_file)
        
        with tqdm(total=file_size, unit='B', unit_scale=True, 
                 desc=f"Diluting {Path(input_file).name}") as pbar:
            
            for line in input_handle:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 4:
                    continue
                
                chr_name, start, pattern, count = parts
                start = int(start)
                count = int(count)
                
                chunk_data.append((chr_name, start, pattern, count))
                total_original_reads += count
                
                # Update progress bar
                pbar.update(len(line.encode('utf-8')))
                
                # Process chunk when it reaches target size
                if len(chunk_data) >= chunk_size:
                    # Generate unique random seed for this chunk
                    chunk_seed = random.randint(0, 2**32-1) if random_seed is not None else None
                    
                    diluted_entries = process_pat_chunk(chunk_data, target_fraction, chunk_seed)
                    
                    # Write diluted entries
                    for chr_name, start, pattern, count in diluted_entries:
                        output_handle.write(f"{chr_name}\t{start}\t{pattern}\t{count}\n")
                        total_diluted_reads += count
                    
                    total_processed += len(chunk_data)
                    chunk_data = []
            
            # Process remaining data
            if chunk_data:
                chunk_seed = random.randint(0, 2**32-1) if random_seed is not None else None
                diluted_entries = process_pat_chunk(chunk_data, target_fraction, chunk_seed)
                
                for chr_name, start, pattern, count in diluted_entries:
                    output_handle.write(f"{chr_name}\t{start}\t{pattern}\t{count}\n")
                    total_diluted_reads += count
                
                total_processed += len(chunk_data)
    
    finally:
        input_handle.close()
        output_handle.close()
    
    # Calculate actual dilution achieved
    actual_fraction = total_diluted_reads / total_original_reads if total_original_reads > 0 else 0
    
    print(f"  Original reads: {total_original_reads:,}")
    print(f"  Diluted reads: {total_diluted_reads:,}")
    print(f"  Target fraction: {target_fraction:.3f}")
    print(f"  Actual fraction: {actual_fraction:.3f}")
    
    return {
        'input_file': str(input_file),
        'output_file': str(output_file),
        'original_reads': total_original_reads,
        'diluted_reads': total_diluted_reads,
        'target_fraction': target_fraction,
        'actual_fraction': actual_fraction
    }


def dilute_pat_directory(input_dir, output_dir, target_fraction, parallel_jobs=4, random_seed=42):
    """
    Dilute all PAT files in a directory
    
    Args:
        input_dir: Directory containing input PAT files
        output_dir: Directory for output diluted PAT files
        target_fraction: Target coverage fraction (0.0-1.0)
        parallel_jobs: Number of parallel processes
        random_seed: Random seed for reproducibility
    """
    print(f"Diluting PAT files from {input_dir} to {target_fraction:.1%} coverage")
    print(f"Output directory: {output_dir}")
    
    # Create output directory recursively
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find PAT files
    pat_files = []
    for pattern in ['*.pat.gz', '*.pat']:
        pat_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    if not pat_files:
        raise ValueError(f"No PAT files found in {input_dir}")
    
    print(f"Found {len(pat_files)} PAT files to process")
    
    # Prepare arguments for parallel processing
    process_args = []
    for input_file in pat_files:
        input_path = Path(input_file)
        output_filename = f"{input_path.stem}_diluted_{target_fraction:.3f}.pat.gz"
        output_file = os.path.join(output_dir, output_filename)
        
        # Generate unique seed for each file
        file_seed = random_seed + hash(input_path.name) % 10000 if random_seed is not None else None
        
        process_args.append((input_file, output_file, target_fraction, 100000, file_seed))
    
    # Process files in parallel
    if parallel_jobs > 1:
        with mp.Pool(parallel_jobs) as pool:
            results = pool.starmap(dilute_pat_file, process_args)
    else:
        results = [dilute_pat_file(*args) for args in process_args]
    
    # Summary statistics
    total_original = sum(r['original_reads'] for r in results)
    total_diluted = sum(r['diluted_reads'] for r in results)
    overall_fraction = total_diluted / total_original if total_original > 0 else 0
    
    print(f"\n" + "="*60)
    print("DILUTION SUMMARY")
    print("="*60)
    print(f"Files processed: {len(results)}")
    print(f"Total original reads: {total_original:,}")
    print(f"Total diluted reads: {total_diluted:,}")
    print(f"Target coverage fraction: {target_fraction:.3f}")
    print(f"Actual coverage fraction: {overall_fraction:.3f}")
    print(f"Dilution accuracy: {(overall_fraction/target_fraction)*100:.1f}%")
    
    return results


def validate_dilution_quality(original_file, diluted_file, n_samples=10000):
    """
    Validate that dilution preserves methylation patterns correctly
    
    Args:
        original_file: Path to original PAT file
        diluted_file: Path to diluted PAT file
        n_samples: Number of positions to sample for validation
    """
    print(f"Validating dilution quality...")
    
    # Sample positions from both files
    original_patterns = defaultdict(list)
    diluted_patterns = defaultdict(list)
    
    # Read original file
    with gzip.open(original_file, 'rt') if str(original_file).endswith('.gz') else open(original_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) == 4:
                chr_name, start, pattern, count = parts
                pos_key = (chr_name, int(start))
                original_patterns[pos_key].extend([pattern] * int(count))
    
    # Read diluted file  
    with gzip.open(diluted_file, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                chr_name, start, pattern, count = parts
                pos_key = (chr_name, int(start))
                if pos_key in original_patterns:  # Only check sampled positions
                    diluted_patterns[pos_key].extend([pattern] * int(count))
    
    # Compare methylation frequencies
    correlation_scores = []
    for pos_key in original_patterns:
        if pos_key in diluted_patterns:
            orig_counts = Counter(original_patterns[pos_key])
            dilut_counts = Counter(diluted_patterns[pos_key])
            
            # Calculate methylation rate for common patterns
            common_patterns = set(orig_counts.keys()) & set(dilut_counts.keys())
            if common_patterns:
                orig_meth_rates = [orig_counts[p] / sum(orig_counts.values()) for p in common_patterns]
                dilut_meth_rates = [dilut_counts[p] / sum(dilut_counts.values()) for p in common_patterns]
                
                if len(orig_meth_rates) > 1:
                    corr = np.corrcoef(orig_meth_rates, dilut_meth_rates)[0, 1]
                    if not np.isnan(corr):
                        correlation_scores.append(corr)
    
    if correlation_scores:
        mean_correlation = np.mean(correlation_scores)
        print(f"Pattern preservation correlation: {mean_correlation:.3f}")
        print(f"Positions validated: {len(correlation_scores)}")
        return mean_correlation
    else:
        print("Insufficient data for validation")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Stochastically dilute PAT files to simulate different coverage levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dilute to 50% coverage
    python dilute_pat_files.py --input_dir /path/to/pats --output_dir /path/to/diluted --target_coverage 0.5
    
    # Dilute to 25% coverage with 8 parallel jobs
    python dilute_pat_files.py --input_dir /path/to/pats --output_dir /path/to/diluted --target_coverage 0.25 --jobs 8
    
    # Multiple coverage levels for systematic study
    for cov in 0.1 0.2 0.3 0.5 0.7; do
        python dilute_pat_files.py --input_dir /path/to/pats --output_dir /path/to/diluted_${cov} --target_coverage ${cov}
    done
        """
    )
    
    parser.add_argument('--input_dir', required=True,
                      help='Directory containing input PAT files (*.pat.gz or *.pat)')
    parser.add_argument('--output_dir', required=True,
                      help='Directory for output diluted PAT files')
    parser.add_argument('--target_coverage', type=float, required=True,
                      help='Target coverage fraction (0.0-1.0, e.g., 0.5 for 50%%)')
    parser.add_argument('--jobs', type=int, default=4,
                      help='Number of parallel jobs (default: 4)')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--validate', action='store_true',
                      help='Run validation on first processed file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.target_coverage <= 1.0):
        raise ValueError("target_coverage must be between 0.0 and 1.0")
    
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    # Run dilution
    results = dilute_pat_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_fraction=args.target_coverage,
        parallel_jobs=args.jobs,
        random_seed=args.random_seed
    )
    
    # Optional validation
    if args.validate and results:
        first_result = results[0]
        validate_dilution_quality(
            first_result['input_file'],
            first_result['output_file']
        )
    
    print(f"\n✅ Dilution complete! Output files in: {args.output_dir}")


if __name__ == "__main__":
    main()