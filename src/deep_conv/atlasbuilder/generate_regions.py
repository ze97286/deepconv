import argparse 
import pandas as pd
from pathlib import Path


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
        j = i + min_cpgs
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


if __name__ == "__main__":
    """
      The script calculates all of the valid regions with minimum of consecutive <min_cpgs> and a maximum length of <max_distance> bases. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpg_file', type=str, required=True)
    parser.add_argument('--chr', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--min_cpgs', type=int, default=4)
    parser.add_argument('--max_distance', type=int, default=1000)
    args = parser.parse_args()
    generate_regions(
        cpg_file=args.cpg_file, 
        chromosome=args.chr, 
        output_dir=args.output_dir, 
        min_cpgs=args.min_cpgs, 
        max_distance=args.max_distance
    )
    