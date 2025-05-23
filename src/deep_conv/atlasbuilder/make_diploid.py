import pandas as pd
import numpy as np
import gzip
from typing import Dict, Tuple
from tqdm import tqdm
from pathlib import Path

def load_cpg_positions(cpg_bed_path: str) -> pd.DataFrame:
    """
    Load CpG positions and create a mapping from (chr, cpg_index) to genomic position.
    BED file format: chr, position, cpg_index
    """
    print("Loading CpG positions...")
    
    # Read BED file with correct column names
    cpg_df = pd.read_csv(cpg_bed_path, sep='\t', compression='gzip', 
                         header=None, names=['chr', 'position', 'cpg_idx'])
    
    # Create multi-index for fast lookup
    cpg_mapping = cpg_df.set_index(['chr', 'cpg_idx'])
    
    return cpg_mapping

def parse_hatchet_cn(hatchet_file: str) -> pd.DataFrame:
    """
    Parse HATCHET best.bbc.ucn file to extract copy number segments.
    Simply uses the RD (read depth) column which already contains the aggregate CN information.
    """
    print("Parsing HATCHET copy number data...")
    
    # Read HATCHET file
    cn_df = pd.read_csv(hatchet_file, sep='\t', comment='#', header=None)
    
    # Extract only what we need: chr, start, end, RD
    # RD (column 4) is the read depth ratio - this is all we need!
    cn_segments = cn_df.iloc[:, [0, 1, 2, 4]].copy()
    cn_segments.columns = ['chr', 'start', 'end', 'rd']
    
    # Convert RD to copy number: CN = RD * 2
    cn_segments['cn'] = cn_segments['rd'] * 2
    
    # Keep only the columns we need
    cn_segments = cn_segments[['chr', 'start', 'end', 'cn']]
    
    # Sort by chromosome and start position
    cn_segments = cn_segments.sort_values(['chr', 'start'])
    
    print(f"Parsed {len(cn_segments)} segments")
    print(f"CN range: {cn_segments['cn'].min():.2f} - {cn_segments['cn'].max():.2f}")
    
    # Debug: Check for gaps in coverage
    for chrom in ['chr1', 'chr2', 'chr3']:
        chr_segs = cn_segments[cn_segments['chr'] == chrom]
        if len(chr_segs) > 0:
            print(f"\n{chrom} coverage: {len(chr_segs)} segments")
            print(f"  First segment: {chr_segs.iloc[0]['start']:,} - {chr_segs.iloc[0]['end']:,}")
            print(f"  Last segment: {chr_segs.iloc[-1]['start']:,} - {chr_segs.iloc[-1]['end']:,}")
            
            # Check for gaps
            gaps = []
            for i in range(len(chr_segs) - 1):
                if chr_segs.iloc[i]['end'] < chr_segs.iloc[i+1]['start']:
                    gap_size = chr_segs.iloc[i+1]['start'] - chr_segs.iloc[i]['end']
                    if gap_size > 1000:  # Only report significant gaps
                        gaps.append((chr_segs.iloc[i]['end'], chr_segs.iloc[i+1]['start'], gap_size))
            
            if gaps:
                print(f"  Found {len(gaps)} gaps in coverage")
                for i, (end, start, size) in enumerate(gaps[:3]):  # Show first 3 gaps
                    print(f"    Gap {i+1}: {end:,} - {start:,} ({size:,} bp)")
                if len(gaps) > 3:
                    print(f"    ... and {len(gaps)-3} more gaps")
    
    return cn_segments

def assign_cn_to_positions_vectorized(positions_df: pd.DataFrame, cn_segments: pd.DataFrame) -> pd.Series:
    """
    Vectorized assignment of copy numbers to positions.
    """
    # Initialize with default diploid
    cn_values = pd.Series(2.0, index=positions_df.index)
    
    # Group by chromosome for efficiency
    for chrom in positions_df['chr'].unique():
        # Get data for this chromosome
        chr_mask = positions_df['chr'] == chrom
        chr_positions = positions_df.loc[chr_mask, 'position'].values
        chr_segments = cn_segments[cn_segments['chr'] == chrom]
        
        if len(chr_segments) == 0:
            continue
        
        # Vectorized interval assignment
        for _, segment in chr_segments.iterrows():
            segment_mask = (chr_positions >= segment['start']) & (chr_positions < segment['end'])
            if segment_mask.any():
                cn_values.loc[chr_mask].iloc[segment_mask] = segment['cn']
    
    return cn_values

def probabilistic_round_vectorized(values: pd.Series, rng: np.random.Generator) -> pd.Series:
    """
    Vectorized probabilistic rounding.
    """
    if len(values) == 0:
        return values
    
    integer_parts = values.astype(int)
    fractional_parts = values - integer_parts
    
    # Generate random values for all elements at once
    random_values = rng.random(len(values))
    
    # Add 1 where random value is less than fractional part
    return integer_parts + (random_values < fractional_parts).astype(int)

def get_chromosome_sizes_hg38():
    """
    Return chromosome sizes for hg38.
    """
    return {
        'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559,
        'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
        'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
        'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
        'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
        'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
        'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
        'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415
    }

def fill_cn_gaps(cn_segments: pd.DataFrame, fill_gaps: bool = True) -> pd.DataFrame:
    """
    Fill gaps between CN segments with interpolated values.
    """
    if not fill_gaps:
        return cn_segments
    
    print("Filling gaps in CN segments...")
    chromosome_sizes = get_chromosome_sizes_hg38()
    
    filled_segments = []
    total_original_coverage = 0
    total_filled_coverage = 0
    
    for chrom in cn_segments['chr'].unique():
        if chrom not in chromosome_sizes:
            continue
            
        chr_size = chromosome_sizes[chrom]
        chr_segs = cn_segments[cn_segments['chr'] == chrom].sort_values('start').reset_index(drop=True)
        
        if len(chr_segs) == 0:
            continue
        
        # Calculate original coverage
        for _, seg in chr_segs.iterrows():
            total_original_coverage += seg['end'] - seg['start']
        
        # Fill from chromosome start to first segment
        if chr_segs.iloc[0]['start'] > 1:
            filled_segments.append({
                'chr': chrom,
                'start': 1,
                'end': chr_segs.iloc[0]['start'],
                'cn': chr_segs.iloc[0]['cn'],
                'filled': True
            })
            total_filled_coverage += chr_segs.iloc[0]['start'] - 1
        
        # Add segments and fill gaps
        for i in range(len(chr_segs)):
            # Add the original segment
            seg = chr_segs.iloc[i].to_dict()
            seg['filled'] = False
            filled_segments.append(seg)
            
            # Fill gap to next segment
            if i < len(chr_segs) - 1:
                gap_start = chr_segs.iloc[i]['end']
                gap_end = chr_segs.iloc[i+1]['start']
                
                if gap_end > gap_start:
                    # Use average of surrounding segments
                    avg_cn = (chr_segs.iloc[i]['cn'] + chr_segs.iloc[i+1]['cn']) / 2
                    filled_segments.append({
                        'chr': chrom,
                        'start': gap_start,
                        'end': gap_end,
                        'cn': avg_cn,
                        'filled': True
                    })
                    total_filled_coverage += gap_end - gap_start
        
        # Fill from last segment to chromosome end
        last_end = chr_segs.iloc[-1]['end']
        if last_end < chr_size:
            filled_segments.append({
                'chr': chrom,
                'start': last_end,
                'end': chr_size,
                'cn': chr_segs.iloc[-1]['cn'],
                'filled': True
            })
            total_filled_coverage += chr_size - last_end
    
    filled_df = pd.DataFrame(filled_segments)
    
    # Print statistics
    total_coverage = total_original_coverage + total_filled_coverage
    print(f"Original CN coverage: {total_original_coverage:,} bp ({100*total_original_coverage/total_coverage:.1f}%)")
    print(f"Filled coverage: {total_filled_coverage:,} bp ({100*total_filled_coverage/total_coverage:.1f}%)")
    print(f"Total segments: {len(cn_segments)} original + {len(filled_df) - len(cn_segments)} filled = {len(filled_df)}")
    
    return filled_df[['chr', 'start', 'end', 'cn']].sort_values(['chr', 'start'])

def correct_pat_file_probabilistic(
    pat_file: str, 
    hatchet_file: str, 
    cpg_bed_path: str, 
    output_file: str,
    seed: int = 42,
    chunk_size: int = 1_000_000,
    fill_gaps: bool = True
):
    """
    Create CNA-corrected PAT file using probabilistic rounding to preserve pattern diversity.
    
    Args:
        pat_file: Input PAT file path
        hatchet_file: HATCHET best.bbc.ucn file path
        cpg_bed_path: CpG reference BED file path (gzipped)
        output_file: Output CNA-corrected PAT file path
        seed: Random seed for reproducibility
        chunk_size: Number of lines to process at once
        fill_gaps: Whether to fill gaps between CN segments
    """
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Load reference data
    cpg_mapping = load_cpg_positions(cpg_bed_path)
    print(f"Loaded {len(cpg_mapping)} CpG positions")
    print(f"CpG mapping chromosomes: {cpg_mapping.index.get_level_values('chr').unique()[:5].tolist()}")
    
    cn_segments = parse_hatchet_cn(hatchet_file)
    print(f"CN segments chromosomes: {cn_segments['chr'].unique()[:5].tolist()}")
    print(f"CN range: {cn_segments['cn'].min():.2f} - {cn_segments['cn'].max():.2f}")
    
    # Fill gaps in CN segments
    cn_segments = fill_cn_gaps(cn_segments, fill_gaps=fill_gaps)
    
    # Statistics tracking
    stats = {
        'total_patterns': 0,
        'patterns_removed': 0,
        'patterns_unchanged': 0,
        'patterns_adjusted': 0,
        'total_reads_before': 0,
        'total_reads_after': 0,
        'cn_distribution': {}
    }
    
    # Count total lines for progress bar
    print("Counting patterns...")
    if pat_file.endswith('.gz'):
        with gzip.open(pat_file, 'rt') as f:
            total_lines = sum(1 for _ in f)
    else:
        with open(pat_file, 'r') as f:
            total_lines = sum(1 for _ in f)
    
    print(f"Processing {total_lines:,} patterns...")
    
    # Process in chunks
    with tqdm(total=total_lines, desc="Processing patterns") as pbar:
        # Open output file
        with open(output_file, 'w') as outfile:
            # Read PAT file in chunks
            for chunk in pd.read_csv(pat_file, sep='\t', header=None, 
                                    names=['chr', 'cpg_idx', 'pattern', 'count'],
                                    chunksize=chunk_size):
                
                # Update statistics
                stats['total_patterns'] += len(chunk)
                stats['total_reads_before'] += chunk['count'].sum()
                
                # Debug: Check first few entries
                if stats['total_patterns'] <= 1000000:
                    print(f"\nFirst few PAT entries:")
                    print(chunk.head())
                    print(f"\nUnique chromosomes in chunk: {chunk['chr'].unique()[:5]}")
                    print(f"CpG index range: {chunk['cpg_idx'].min()} - {chunk['cpg_idx'].max()}")
                
                # Merge with CpG positions to get genomic coordinates
                chunk_with_pos = chunk.merge(
                    cpg_mapping, 
                    left_on=['chr', 'cpg_idx'], 
                    right_index=True, 
                    how='left'
                )
                
                # Filter out unmapped positions
                valid_mask = ~chunk_with_pos['position'].isna()
                if not valid_mask.all():
                    unmapped_count = (~valid_mask).sum()
                    if unmapped_count > 100000:  # Only show warning for large numbers
                        print(f"Warning: {unmapped_count} positions couldn't be mapped")
                
                chunk_with_pos = chunk_with_pos[valid_mask].copy()
                
                if len(chunk_with_pos) == 0:
                    pbar.update(len(chunk))
                    continue
                
                # Assign copy numbers
                chunk_with_pos['cn'] = assign_cn_to_positions_vectorized(chunk_with_pos, cn_segments)
                
                # Track CN distribution
                cn_counts = chunk_with_pos['cn'].value_counts()
                for cn, count in cn_counts.items():
                    cn_key = f"{cn:.1f}"
                    stats['cn_distribution'][cn_key] = stats['cn_distribution'].get(cn_key, 0) + count
                
                # Calculate adjustment factor
                chunk_with_pos['adjustment_factor'] = 2.0 / chunk_with_pos['cn']
                
                # Apply probabilistic correction
                adjusted_values = chunk_with_pos['count'] * chunk_with_pos['adjustment_factor']
                chunk_with_pos['adjusted_count'] = probabilistic_round_vectorized(adjusted_values, rng)
                
                # Update statistics
                unchanged_mask = chunk_with_pos['cn'] == 2.0
                stats['patterns_unchanged'] += unchanged_mask.sum()
                
                adjusted_mask = (chunk_with_pos['cn'] != 2.0) & (chunk_with_pos['adjusted_count'] > 0)
                stats['patterns_adjusted'] += adjusted_mask.sum()
                
                removed_mask = (chunk_with_pos['cn'] != 2.0) & (chunk_with_pos['adjusted_count'] == 0)
                stats['patterns_removed'] += removed_mask.sum()
                
                # Filter out patterns with 0 count
                output_chunk = chunk_with_pos[chunk_with_pos['adjusted_count'] > 0].copy()
                
                stats['total_reads_after'] += output_chunk['adjusted_count'].sum()
                
                # Write output
                output_chunk[['chr', 'cpg_idx', 'pattern', 'adjusted_count']].to_csv(
                    outfile, 
                    sep='\t', 
                    header=False, 
                    index=False,
                    mode='a'
                )
                
                pbar.update(len(chunk))
    
    # Print statistics
    print("\n=== CNA Correction Statistics ===")
    print(f"Total patterns processed: {stats['total_patterns']:,}")
    print(f"Patterns removed (adjusted to 0): {stats['patterns_removed']:,} ({100*stats['patterns_removed']/stats['total_patterns']:.2f}%)")
    print(f"Patterns unchanged (CN=2): {stats['patterns_unchanged']:,} ({100*stats['patterns_unchanged']/stats['total_patterns']:.2f}%)")
    print(f"Patterns adjusted: {stats['patterns_adjusted']:,} ({100*stats['patterns_adjusted']/stats['total_patterns']:.2f}%)")
    print(f"\nTotal reads before: {stats['total_reads_before']:,}")
    print(f"Total reads after: {stats['total_reads_after']:,}")
    print(f"Read preservation: {100*stats['total_reads_after']/stats['total_reads_before']:.2f}%")
    
    print("\nCopy number distribution:")
    for cn, count in sorted(stats['cn_distribution'].items()):
        print(f"  CN={cn}: {count:,} patterns ({100*count/stats['total_patterns']:.2f}%)")
    
    return stats

def process_multiple_samples(
    sample_pairs: list,  # List of (pat_file, hatchet_file) tuples
    cpg_bed_path: str,
    output_dir: str,
    seed: int = 42,
    fill_gaps: bool = True
):
    """
    Process multiple tumor samples to create CNA-corrected PAT files.
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    
    for pat_file, hatchet_file in sample_pairs:
        sample_name = Path(pat_file).stem.replace('.pat', '')
        output_file = Path(output_dir) / f"{sample_name}_cna_corrected.pat"
        
        print(f"\n{'='*60}")
        print(f"Processing sample: {sample_name}")
        print(f"{'='*60}")
        
        stats = correct_pat_file_probabilistic(
            pat_file=pat_file,
            hatchet_file=hatchet_file,
            cpg_bed_path=cpg_bed_path,
            output_file=str(output_file),
            seed=seed,
            fill_gaps=fill_gaps
        )
        
        all_stats[sample_name] = stats
    
    return all_stats

if __name__ == "__main__":
    sample_pairs = [
        ("/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Results/1.6/pat/129-001_ScrBsl_tumour.pat.gz", "/mnt/lustre/users/bschuster/OAC_Trial_WGS_Tissue_CNA-Hatchet/Results/129-001:ScrBsl:duodenum-ScrBsl/best.bbc.ucn"),
        ("/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Results/1.6/pat/071-021_ScrBsl_tumour.pat.gz", "/mnt/lustre/users/bschuster/OAC_Trial_WGS_Tissue_CNA-Hatchet/Results/071-021:ScrBsl:duodenum-ScrBsl/best.bbc.ucn"),
        ("/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Results/1.6/pat/071-011_ScrBsl_tumour.pat.gz", "/mnt/lustre/users/bschuster/OAC_Trial_WGS_Tissue_CNA-Hatchet/Results/071-011:ScrBsl:duodenum-ScrBsl/best.bbc.ucn"),
        ("/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Results/1.6/pat/069-009_ScrBsl_tumour.pat.gz", "/mnt/lustre/users/bschuster/OAC_Trial_WGS_Tissue_CNA-Hatchet/Results/069-009:ScrBsl:duodenum-ScrBsl/best.bbc.ucn"),
        ("/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Results/1.6/pat/071-043_ScrBsl_tumour.pat.gz", "/mnt/lustre/users/bschuster/OAC_Trial_WGS_Tissue_CNA-Hatchet/Results/071-043:ScrBsl:duodenum-ScrBsl/best.bbc.ucn"),
        ("/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Results/1.6/pat/071-022_ScrBsl_tumour.pat.gz", "/mnt/lustre/users/bschuster/OAC_Trial_WGS_Tissue_CNA-Hatchet/Results/071-022:ScrBsl:duodenum-ScrBsl/best.bbc.ucn"),
    ]
    
    process_multiple_samples(
        sample_pairs=sample_pairs,
        cpg_bed_path="/users/zetzioni/sharedscratch/wgbs_tools/references/hg38/CpG.bed.gz",
        output_dir="/users/zetzioni/sharedscratch/loyfer_atlas/cna_corrected_pats"
    )