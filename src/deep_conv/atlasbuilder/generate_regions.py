import argparse 
import time
import glob
import os
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

def generate_regions(cpg_file, chromosome, output_dir, min_cpgs=4, max_distance=1000):
    """
    Generate regions for a single chromosome.
    
    Args:
        cpg_file: Path to CpG.bed.gz
        chromosome: Chromosome to process
        min_cpgs: Minimum number of CpGs in a region (default: 4)
        max_distance: Maximum distance between first and last CpG (default: 1000)
        
    Returns:
        DataFrame with columns: chr, start, end, startCpG, endCpG, name, direction, target
    """
    out_file = f"{output_dir}/regions_{chromosome}_{min_cpgs}_{max_distance}.bed.gz"
    # Read CpG positions for this chromosome
    my_file = Path(out_file)
    if my_file.is_file():
        return pd.read_csv(out_file, sep="\t")
    cpgs = pd.read_csv(cpg_file, sep='\t', names=['chr', 'pos', 'index'])
    chr_cpgs = cpgs[cpgs['chr'] == chromosome]
    # Convert to numpy for faster access
    positions = chr_cpgs['pos'].values
    indices = chr_cpgs['index'].values
    n_cpgs = len(positions)
    regions = []
    # For each starting CpG
    for i in range(n_cpgs):
        # Find all valid ending positions
        j = i + min_cpgs - 1  # minimum end position
        while j < n_cpgs and positions[j] - positions[i] <= max_distance:
            regions.append({
                'chr': chromosome,
                'start': positions[i],
                'end': positions[j],
                'startCpG': indices[i],
                'endCpG': indices[j],
                'name': f"{chromosome}:{positions[i]}-{positions[j]}",
                'direction': 'U',
                'target': 'test'
            })
            j += 1
    # Convert to DataFrame
    regions_df = pd.DataFrame(regions)
    if not regions_df.empty:
        regions_df = regions_df.sort_values(['start', 'end'])
    regions_df.to_csv(out_file, index=False, compression="gzip", sep="\t")
    return regions_df


def process_chunk(args):
    """
    Process a chunk of regions to find non-overlapping batches
    Writes batches to files as they are created
    """
    chr, chunk_df, chunk_id, output_dir = args  
    n_processed = 0
    n_batches = 0
    
    # Sort by length first, then by start position
    sorted_chunk = chunk_df.assign(length=lambda x: x['end'] - x['start'])\
                          .sort_values(['length', 'start'])
    
    remaining_regions = set(range(len(sorted_chunk)))
    last_report_time = time.time()
    report_interval = 5  # seconds
    
    while remaining_regions:
        current_time = time.time()
        if current_time - last_report_time >= report_interval:
            print(f"Chunk {chunk_id:2d}: "
                  f"Processed {n_processed:7d}/{len(sorted_chunk)} regions "
                  f"({n_processed/len(sorted_chunk)*100:5.1f}%), "
                  f"Batches: {n_batches:4d}, "
                  f"Avg batch size: {n_processed/max(n_batches, 1):6.1f}")
            last_report_time = current_time
        
        # Start new batch
        current_batch = []
        current_end = 0
        
        # Try to fill this batch with as many non-overlapping regions as possible
        for idx in sorted(remaining_regions,
                         key=lambda x: (sorted_chunk.iloc[x]['start'],
                                      sorted_chunk.iloc[x]['end'])):
            region = sorted_chunk.iloc[idx]
            if region['start'] >= current_end:
                current_batch.append(region)
                current_end = region['end']
                remaining_regions.remove(idx)
                n_processed += 1
        
        if current_batch:
            # Save batch directly to file
            batch_df = pd.DataFrame(current_batch)
            os.makedirs(Path(output_dir) / "by_chr", exist_ok=True)
            batch_file = Path(output_dir) / "by_chr" /f"{chr}_region_{chunk_id}_{n_batches}.l4.bed"
            batch_df.to_csv(batch_file, sep='\t', index=False)
            n_batches += 1
    
    print(f"Chunk {chunk_id:2d} COMPLETE: "
          f"Total regions: {len(sorted_chunk):7d}, "
          f"Batches: {n_batches:4d}, "
          f"Final avg batch size: {len(sorted_chunk)/n_batches:6.1f}")
    
    return n_batches  


def find_non_overlapping_batches(chr, regions_df, output_dir, threads=32):
    """
    Find minimal number of non-overlapping batches using parallel processing
    """
    total_regions = len(regions_df)
    print(f"Starting to process {total_regions} regions...")
    # Split into roughly equal chunks
    chunk_size = len(regions_df) // threads
    chunks = []
    print(f"Splitting into {threads} chunks...")
    for i in range(0, len(regions_df), chunk_size):
        chunk = regions_df.iloc[i:i + chunk_size]
        chunks.append(chunk)
    print(f"Created {len(chunks)} chunks with sizes: {[len(chunk) for chunk in chunks]}")
    # Process chunks in parallel
    chunk_args = [(chr, chunk, i, output_dir) for i, chunk in enumerate(chunks)]
    with Pool(min(threads, len(chunks))) as pool:
        chunk_results = list(tqdm(
            pool.imap(process_chunk, chunk_args),
            total=len(chunks),
            desc="Processing chunks"
        ))
    # Summarize results
    total_batches = sum(chunk_results)  # Now just sum of n_batches returned from each chunk
    # Get actual file counts and sizes
    region_files = glob.glob(f"{output_dir}/{chr}_region_*.l4.bed")
    total_regions = sum(len(pd.read_csv(f, sep='\t')) for f in region_files)
    print(f"\nFinal Statistics:")
    print(f"Total batches created: {total_batches}")
    print(f"Total regions in batches: {total_regions}")
    print(f"Average batch size: {total_regions/total_batches:.1f}")
    print(f"Coverage: {total_regions/len(regions_df)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpg_file', type=str, required=True)
    parser.add_argument('--chr', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--min_cpgs', type=int, default=4)
    parser.add_argument('--max_distance', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    regions_df = generate_regions(
        cpg_file=args.cpg_file, 
        chromosome=args.chr, 
        output_dir=args.output_dir, 
        min_cpgs=args.min_cpgs, 
        max_distance=args.max_distance
    )
    find_non_overlapping_batches(args.chr, regions_df, args.output_dir, args.threads)
    