import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from deep_conv.atlasbuilder.find_marker_candidates import create_marker_matrices, get_ground_truth
import plotly.graph_objects as go
import plotly.subplots as sp
import math


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


def merge_all():
    merge("/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/", 5, "eval", "high")
    merge("/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/", 5, "eval", "med")
    merge("/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/", 5, "eval", "low")


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