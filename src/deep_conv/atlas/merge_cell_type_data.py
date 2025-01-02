import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from functools import partial


# python -m deep_conv.atlas.merge_cell_type_data \
#   --input_dir /users/zetzioni/sharedscratch/atlas/pats/tmp4 \
#   --output_dir /users/zetzioni/sharedscratch/atlas/pat_index \
#   --class_file /users/zetzioni/sharedscratch/atlas/taps_atlas_class.csv
#   --tmp_dir tmp4
#   --min_cpgs 4

def merge_chromosome(cell_type, chr_num, input_dir, tmp_dir, output_dir):
    """
    Merge entries for a specific chromosome for a given cell type using pandas.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / tmp_dir / cell_type
    output_path.mkdir(parents=True, exist_ok=True)
    # Collect all files for this cell type and chromosome
    files = list(input_path.glob(f"{cell_type}_*_chr{chr_num}_index.txt"))
    print("merging cell type",cell_type,"from",files)
    if not files:
        print(f"No files found for {cell_type}, chr{chr_num}")
        return
    # Load and validate the files
    df_list = []
    for file in files:
        df = pd.read_csv(file, sep="\t", header=0, names=["pos", "count"])
        if df.isna().any().any():
            raise ValueError(f"NaN values found in file: {file}")
        if not pd.api.types.is_numeric_dtype(df["count"]):
            raise ValueError(f"Non-numeric counts found in file: {file}")
        df_list.append(df)
    # Combine all DataFrames
    combined_df = pd.concat(df_list)
    merged_df = combined_df.groupby("pos", as_index=False).sum()
    # Add `chr` and `group` columns
    merged_df["chr"] = f"chr{chr_num}"
    merged_df["group"] = cell_type
    # Save merged results to a gzipped file
    output_file = output_path / f"coverage_index_chr{chr_num}.txt.gz"
    merged_df.to_csv(output_file, sep="\t", index=False, compression="gzip")


def merge_cell_type(cell_type, chromosomes, input_dir, tmp_dir, output_dir):
    """
    Merge all chromosomes for a given cell type sequentially and combine into a single file.
    """
    output_path = Path(output_dir) / tmp_dir / cell_type
    output_path.mkdir(parents=True, exist_ok=True)
    # Process each chromosome sequentially
    for chr_num in chromosomes:
        merge_chromosome(cell_type, chr_num, input_dir, tmp_dir, output_dir)
    # Combine all chromosome files into a single file
    df_list = [
        pd.read_csv(output_path / f"coverage_index_chr{chr_num}.txt.gz", sep="\t", header=0)
        for chr_num in chromosomes
        if (output_path / f"coverage_index_chr{chr_num}.txt.gz").exists()
    ]
    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        combined_file = Path(output_dir) / f"{tmp_dir}/{cell_type}_merged_coverage_index.txt.gz"
        merged_df.to_csv(combined_file, sep="\t", index=False, compression="gzip")


def parallel_merge(cell_types, chromosomes, min_cpgs, input_dir, tmp_dir, output_dir, output_prefix):
    """
    Parallelize the merging process by cell type (chromosome merging is sequential).
    """
    with Pool() as pool:
        pool.map(
            partial(merge_cell_type, chromosomes=chromosomes, input_dir=input_dir, tmp_dir=tmp_dir, output_dir=output_dir),
            cell_types
        )
    merge_all_cell_types(cell_types, min_cpgs, tmp_dir, output_dir, output_prefix)


def merge_all_cell_types(cell_types, min_cpgs, tmp_dir, output_dir,output_prefix):
    """
    Merge all cell type files into a single zipped CSV.
    """
    merged_file = Path(output_dir) / f"{output_prefix}cell_type_pat_index_l{min_cpgs}.csv.gz"
    df_list = [
        pd.read_csv(Path(output_dir) / f"{tmp_dir}/{cell_type}_merged_coverage_index.txt.gz", sep="\t", header=0)
        for cell_type in cell_types
        if (Path(output_dir) / f"{tmp_dir}/{cell_type}_merged_coverage_index.txt.gz").exists()
    ]
    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df = merged_df.sort_values(["chr", "pos", "group"])
        merged_df[["chr", "pos", "group","count"]].to_csv(merged_file, sep="\t", index=False, compression="gzip")
        print(f"Merged all cell types into {merged_file}")
    else:
        print("No cell type files found to merge.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parallel merge by cell type and chromosome.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing intermediate files.")
    parser.add_argument("--class_file", required=True, help="List of cell types.")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged files.")
    parser.add_argument("--tmp_dir", required=True, help="Temporary directory for intermediate merged files.")
    parser.add_argument("--output_prefix", required=False,default=None, help="Prefix for the output file.")
    parser.add_argument("--min_cpgs", required=True, help="the minimum number of cpgs per read.")

    args = parser.parse_args()
    df = pd.read_csv(args.class_file)
    cell_types = sorted(df.group.unique())
    output_prefix = ""
    if args.output_prefix is not None:
        output_prefix = args.output_prefix+"_"
    parallel_merge(cell_types, range(1, 23), args.min_cpgs, args.input_dir, args.tmp_dir, args.output_dir, output_prefix)    
    final_df = pd.read_csv(args.output_dir+f"/{output_prefix}cell_type_pat_index_l{args.min_cpgs}.csv.gz", sep="\t")
    print(final_df.head())
    print(final_df.info())