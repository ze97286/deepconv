import argparse
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple 

from deep_conv.benchmark.benchmark_utils import *
from deep_conv.presence.model import *
from deep_conv.presence.train import train_presence_model
from deep_conv.presence.evaluate import evaluate_presence_model


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_validation_set(eval_pat_dir: str, atlas: pd.DataFrame, names: set) -> Tuple[DataLoader, torch.Tensor]:
    """
    Reads marker coverage, methylation, and ground-truth label files from a validation set directory,
    filters them down to the set of markers in 'names', and returns a DataLoader plus normalized labels.

    Args:
        eval_pat_dir (str):
            Directory containing "marker_values.parquet", "coverage.parquet", and "ground_truth_y.parquet" 
            for the validation data.
        atlas (pd.DataFrame):
            A DataFrame that includes marker metadata and cell-type columns. 
            We use atlas.columns[8:] as the cell-type columns for the dataset constructor.
        names (set):
            The set of marker names (strings) to include, ensuring we only keep markers that appear
            in both 'atlas' and the parquet files.

    Returns:
        val_loader (DataLoader):
            A DataLoader wrapping the TissueDeconvolutionDataset for the validation set, 
            with batch_size=512.
        y_val (torch.Tensor):
            A [N, C] Tensor of ground-truth cell-type proportions, normalized so each row sums to 1.
    """
    # Load marker values, coverage, and labels from parquet
    X_val = pd.read_parquet(Path(eval_pat_dir) / "marker_values.parquet")
    coverage_val = pd.read_parquet(Path(eval_pat_dir) / "coverage.parquet")
    y_val = pd.read_parquet(Path(eval_pat_dir) / "ground_truth_y.parquet")

    # Filter to only include markers in 'names'
    X_val = X_val[X_val.name.isin(names)]
    coverage_val = coverage_val[coverage_val.name.isin(names)]
    
    # Drop name/direction columns and transpose => shape [samples, markers]
    X_val = X_val.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_val = coverage_val.drop(columns=["name", "direction"]).T.to_numpy()

    # Print some coverage stats for debug/monitoring
    print("median coverage", 
          np.median(coverage_val, axis=1), 
          "median of medians", 
          np.median(np.median(coverage_val, axis=1)), 
          "mean median", 
          np.median(coverage_val, axis=1).mean())
    
    y_val = y_val.to_numpy()
    
    val_dataset = TissueDeconvolutionDataset(
        X_val,
        coverage_val,
        atlas[atlas.columns[8:]].T.to_numpy(),
        y_val
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=4,
        persistent_workers=True,
        shuffle=False
    )
    
    # Convert y_val to a PyTorch tensor and normalize each row
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_val = y_val / y_val.sum(dim=1, keepdim=True)
    
    return val_loader, y_val


def load_training(base_dir: str, atlas: pd.DataFrame, names: set, num_files: int = 4) -> DataLoader:
    """
    Loads and merges multiple parquet files containing training data (marker_values, coverage, ground_truth_y),
    filters them to only include the markers in 'names', and returns a DataLoader for training.

    Args:
        base_dir (str):
            Path prefix for the training parquet files. We expect files named like:
                base_dir + "1_marker_values.parquet",
                base_dir + "1_coverage.parquet",
                base_dir + "1_ground_truth_y.parquet",
                and so on up to num_files.
        atlas (pd.DataFrame):
            DataFrame with marker metadata plus columns for each cell type (atlas.columns[8:] are cell types).
        names (set):
            The set of marker names to keep (usually matches the set in the atlas).
        num_files (int):
            Number of parquet file batches to merge. Defaults to 4.

    Returns:
        DataLoader:
            A DataLoader over the merged training dataset with batch_size=256, shuffle=True, etc.
    """
    markers = []
    coverage = []
    y = []
    
    print("loading training from", base_dir)
    suffixes = [f"_batch{i}" for i in range(1, num_files + 1)]
    
    # Read multiple parquet files and accumulate marker values, coverage, and ground-truth
    for i in range(1, num_files + 1):
        markers.append(pd.read_parquet(base_dir + str(i) + "_marker_values.parquet"))
        coverage.append(pd.read_parquet(base_dir + str(i) + "_coverage.parquet"))
        y.append(pd.read_parquet(base_dir + str(i) + "_ground_truth_y.parquet"))
    
    # Merge all marker tables on ['name','direction']
    merged_markers = markers[0]
    for i, m in enumerate(markers[1:]):
        merged_markers = merged_markers.merge(
            m,
            on=['name', 'direction'],
            how='outer',
            suffixes=('', suffixes[i])
        )
    
    # Merge all coverage tables on ['name','direction']
    merged_coverage = coverage[0]
    for i, c in enumerate(coverage[1:]):
        merged_coverage = merged_coverage.merge(
            c,
            on=['name', 'direction'],
            how='outer',
            suffixes=('', suffixes[i])
        )
    
    # Concatenate all label DataFrames
    y = pd.concat(y, ignore_index=True).fillna(0)
    
    # Filter out any markers not in 'names'
    X_train = merged_markers[merged_markers.name.isin(names)]
    coverage_train = merged_coverage[merged_coverage.name.isin(names)]
    
    # Drop unnecessary columns and transpose => shape [samples, markers]
    X_train = X_train.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_train = coverage_train.drop(columns=["name", "direction"]).T.to_numpy()
    
    # Print coverage stats for debug
    print("median coverage", 
          np.median(coverage_train, axis=1), 
          "median of medians", 
          np.median(np.median(coverage_train, axis=1)), 
          "mean median", 
          np.median(coverage_train, axis=1).mean())
    
    # Convert the label DataFrame to numpy
    y_train = y.to_numpy()
    
    # Build a TissueDeconvolutionDataset
    train_dataset = TissueDeconvolutionDataset(
        X_train,
        coverage_train,
        atlas[atlas.columns[8:]].T.to_numpy(),
        y_train
    )
    
    # Also create a normalized version of y for potential usage
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_train = y_train / y_train.sum(dim=1, keepdim=True)
    
    # Return a DataLoader for training
    return DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
   

def train_and_eval(
    atlas_path: str,
    train_pat_dir: str,
    eval_pat_dir: str,
    threads: int,
    output_path: str
) -> nn.Module:
    # Fix random seeds and threads for reproducibility
    set_seed()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    # 1) Read the atlas of markers and cell types
    atlas = pd.read_csv(atlas_path, sep="\t")
    
    # The 'names' set ensures we only keep relevant markers
    names = set(atlas.name.unique())

    # 2) Build the training DataLoader from parquet files in train_pat_dir
    train_dl = load_training(train_pat_dir, atlas, names)

    # 3) Build DataLoaders for each validation subset
    tier1_dl, t1_yval = get_validation_set(str(Path(eval_pat_dir) / "tier1"), atlas, names)
    cd4_dl, cd4_yval = get_validation_set(str(Path(eval_pat_dir) / "CD4"), atlas, names)
    cd8_dl, cd8_yval = get_validation_set(str(Path(eval_pat_dir) / "CD8"), atlas, names)
    oac_dl, oac_yval = get_validation_set(str(Path(eval_pat_dir) / "OAC"), atlas, names)

    validation_dls = {
        "tier1": tier1_dl,
        "cd4": cd4_dl,
        "cd8": cd8_dl,
        "oac": oac_dl
    }

    y_vals = {
        "tier1": t1_yval,
        "cd4": cd4_yval,
        "cd8": cd8_yval,
        "oac": oac_yval
    }

    # 4) Identify all cell type columns (atlas.columns[8:])
    cell_types = list(atlas.columns[8:])

    # Build an array mapping each marker to its cell type index
    target_ids = atlas["target"].map(lambda x: cell_types.index(x)).to_numpy()

    # 5) Create the model
    model = CellTypePresenceModel(
        num_markers=len(atlas),
        num_cell_types=len(cell_types),
        target_ids=target_ids,
        feature_dim=32
    )

    # 6) Train the model, saving best checkpoint to `output_path`
    trained_model = train_presence_model(
        model=model,
        train_loader=train_dl,
        val_loaders=validation_dls,
        model_path=output_path,
        num_epochs=1000,
        learning_rate=0.001,
        weight_decay=1e-5,
        presence_threshold=0.0005,  
        patience=10
    )
    
    # 7) Evaluate final model predictions on each validation set
    evaluate_presence_model(
        model=trained_model,
        val_loaders=validation_dls,
        presence_threshold=0.0005,  
        decision_threshold=0.5,
        cell_type_names=cell_types
    )
    return model    


def main():
    parser = argparse.ArgumentParser(description="Deep conv")
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_threads",required=False, type=int, default=32)
    
    args = parser.parse_args()
    
    train_and_eval(args.atlas_path, args.train_path+"/train",args.eval_path+"/eval", args.num_threads, args.output_path)
    

if __name__ == "__main__":    
    main()
