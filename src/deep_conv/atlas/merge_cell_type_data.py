import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from functools import partial


# python -m deep_conv.atlas.merge_cell_type_data \
#   --input_dir /users/zetzioni/sharedscratch/atlas/pats/tmp \
#   --output_dir /users/zetzioni/sharedscratch/atlas/pats \
#   --class_file /users/zetzioni/sharedscratch/atlas/taps_atlas_class.csv

def merge_chromosome(cell_type, chr_num, input_dir, output_dir):
    """
    Merge entries for a specific chromosome for a given cell type using pandas.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / cell_type
    output_path.mkdir(parents=True, exist_ok=True)
    # Collect all files for this cell type and chromosome
    files = list(input_path.glob(f"{cell_type}_chr{chr_num}_index.txt"))
    if not files:
        print(f"No files found for {cell_type}, chr{chr_num}")
        return
    # Load and concatenate all data into a single DataFrame
    df_list = []
    for file in files:
        df = pd.read_csv(file, sep="\t", header=None, names=["chr", "pos", "group", "count"])
        df_list.append(df)
    combined_df = pd.concat(df_list)
    # Group by key columns and sum the counts
    merged_df = combined_df.groupby(["chr", "pos", "group"], as_index=False).sum()
    # Save merged results to a gzipped file
    output_file = output_path / f"coverage_index_chr{chr_num}.txt.gz"
    merged_df.to_csv(output_file, sep="\t", index=False, header=False, compression="gzip")


def merge_cell_type(cell_type, chromosomes, input_dir, output_dir):
    """
    Merge all chromosomes for a given cell type sequentially and combine into a single file.
    """
    output_path = Path(output_dir) / cell_type
    output_path.mkdir(parents=True, exist_ok=True)
    # Process each chromosome sequentially
    for chr_num in chromosomes:
        merge_chromosome(cell_type, chr_num, input_dir, output_dir)
    # Combine all chromosome files into a single file
    combined_file = Path(output_dir) / f"{cell_type}_merged_coverage_index.txt.gz"
    # Concatenate all chromosome files into one
    df_list = []
    for chr_num in chromosomes:
        chr_file = output_path / f"coverage_index_chr{chr_num}.txt.gz"
        if chr_file.exists():
            df = pd.read_csv(chr_file, sep="\t", header=None, names=["chr", "pos", "group", "count"])
            df_list.append(df)
    if df_list:
        merged_df = pd.concat(df_list)
        merged_df.to_csv(combined_file, sep="\t", index=False, header=False, compression="gzip")


def parallel_merge(cell_types, chromosomes, input_dir, output_dir):
    """
    Parallelize the merging process by cell type (chromosome merging is sequential).
    """
    with Pool() as pool:
        pool.map(
            partial(merge_cell_type, chromosomes=chromosomes, input_dir=input_dir, output_dir=output_dir),
            cell_types
        )
    merge_all_cell_types(cell_types, output_dir)


def merge_all_cell_types(cell_types, output_dir):
    """
    Merge all cell type files into a single zipped CSV.
    """
    merged_file = Path(output_dir) / "cell_type_pat_index_l4.csv.gz"
    # Concatenate all cell type files into one
    df_list = []
    for cell_type in cell_types:
        cell_type_file = Path(output_dir) / f"{cell_type}_merged_coverage_index.txt.gz"
        if cell_type_file.exists():
            df = pd.read_csv(cell_type_file, sep="\t", header=None, names=["chr", "pos", "group", "count"])
            df["cell_type"] = cell_type  # Add cell_type column
            df_list.append(df)
    if df_list:
        merged_df = pd.concat(df_list)
        merged_df.to_csv(merged_file, sep="\t", index=False, compression="gzip")
        print(f"Merged all cell types into {merged_file}")
    else:
        print("No cell type files found to merge.")


if __name__ == "__main__":
    import argparse

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Parallel merge by cell type and chromosome.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing intermediate files.")
    parser.add_argument("--class_file", required=True, help="List of cell types.")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged files.")
    
    args = parser.parse_args()
    df = pd.read_csv(args.class_file)
    cell_types = sorted(df.group.unique())

    parallel_merge(cell_types, range(1, 23), args.input_dir, args.output_dir)