import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from deep_conv.atlasbuilder.find_marker_candidates import create_marker_matrices, get_ground_truth

def prepare(atlas_path, pat_dir, cell_types, min_cpgs=4, threads=32):
    if os.path.exists(Path(pat_dir)/"marker_values.parquet"):
        X = pd.read_parquet(Path(pat_dir)/"marker_values.parquet")
    else:
        X, coverage = create_marker_matrices(atlas_path, pat_dir, min_cpgs, threads)
        X.to_parquet(pat_dir/"marker_values.parquet", index=False)
        coverage.to_parquet(pat_dir/"coverage.parquet", index=False)
    y = get_ground_truth(pat_dir,X.columns[2:], cell_types)
    y.to_parquet(pat_dir/"ground_truth_y.parquet", index=False)


def merge(base_dir, prefix):
	markers = []
	coverage = []
	y = []
	suffixes = [f"_batch{i}" for i in range(1,11)]
	for i in range(1,11):		
		markers.append(pd.read_parquet(base_dir+str(i)+"/"+prefix+"/marker_values.parquet"))
		coverage.append(pd.read_parquet(base_dir+str(i)+"/"+prefix+"/coverage.parquet"))
		y.append(pd.read_parquet(base_dir+str(i)+"/"+prefix+"/ground_truth_y.parquet"))		
	merged_markers = markers[0]
	for i, m in enumerate(markers[1:]):
		merged_markers = merged_markers.merge(m, on=['name', 'direction'], how='outer',suffixes=('', suffixes[i]))
	merged_coverage = coverage[0]
	for i, c in enumerate(coverage[1:]):
		merged_coverage = merged_coverage.merge(c, on=['name', 'direction'], how='outer',suffixes=('', suffixes[i]))
	y = pd.concat(y, ignore_index=True)
	merged_markers.to_parquet(f"{base_dir}/{prefix}/marker_values.parquet", index=False)
	merged_coverage.to_parquet(f"{base_dir}/{prefix}/coverage.parquet", index=False)
	y.to_parquet(f"{base_dir}/{prefix}/ground_truth_y.parquet", index=False)
	print(f"saved data to {base_dir}/{prefix}/")


def merge_all():
    base_dir = "/users/zetzioni/sharedscratch/atlas/training/general"
    merge(base_dir, "train")
    merge(base_dir, "eval")


# python -m deep_conv.atlasbuilder.collect_markers_for_training \
# --atlas_path /users/zetzioni/sharedscratch/atlas/atlas/atlas_zohar.blood+gi+tum.l4.bed \
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
    CELL_TYPES = [
        'B-cells', 
        'CD34-erythroblasts', 
        'CD34-megakaryocytes', 
        'CD4-T-cells', 
        'CD8-T-cells', 
        'Colon', 
        'Duodenum', 
        'Eosinophils', 
        'Esophagus', 
        'Monocytes', 
        'Neutrophils', 
        'NK-cells', 
        'OAC', 
        'Pancreas', 
        'Stomach'
    ]
    prepare(args.atlas_path, train_dir, CELL_TYPES, args.min_cpgs, args.threads)
    prepare(args.atlas_path, eval_dir, CELL_TYPES, args.min_cpgs, args.threads)

if __name__ == "__main__":    
    main()