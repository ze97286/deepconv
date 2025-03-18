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
from deep_conv.deconvolution.model import CellTypeDeconvolutionModel, TissueDeconvolutionDataset
from deep_conv.deconvolution.train import train_model
from deep_conv.deconvolution.predict import predict_with_consensus


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


def load_training(base_dir: str, atlas: pd.DataFrame, names: set, num_files: int = 5) -> DataLoader:
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
    for cov in ['high', 'med', 'low']:
        for i in range(1, num_files + 1):
            markers.append(pd.read_parquet(f"{base_dir}_{cov}/{str(i)}_marker_values.parquet"))
            coverage.append(pd.read_parquet(f"{base_dir}_{cov}/{str(i)}_coverage.parquet"))
            y.append(pd.read_parquet(f"{base_dir}_{cov}/{str(i)}_ground_truth_y.parquet"))
    
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
    output_path: str,
    use_loyfer: bool,
    presence_models_dir: str,
) -> nn.Module:
    """
    Loads data, trains a CellTypeDeconvolutionModel, and evaluates it on multiple validation sets.

    Steps:
      1. Set random seeds and threads for reproducibility.
      2. Read the atlas (marker metadata + cell types).
      3. Build a training DataLoader from training parquet files in `train_pat_dir`.
      4. Build validation DataLoaders for multiple subsets (e.g. tier1, CD4, CD8, OAC).
      5. Construct the CellTypeDeconvolutionModel, mapping each marker to a specific cell type.
      6. Train the model (train_model) with early stopping, saving best model checkpoints to `output_path`.
      7. Retrieve the best threshold recommended by the training process.
      8. Evaluate final model predictions on each validation subset and log performance metrics.
      9. Return the trained model (with best checkpoint loaded).

    Args:
        atlas_path (str):
            Path to a TSV (or CSV with sep="\t") containing at least columns 
            ['name', 'target', ... plus cell type columns in columns[8:]].
        train_pat_dir (str):
            Directory containing training parquet files:
               e.g. "1_marker_values.parquet", "1_coverage.parquet", 
                    "1_ground_truth_y.parquet", etc.
        eval_pat_dir (str):
            Directory containing separate validation parquet files 
            (subfolders for "tier1", "CD4", "CD8", "OAC", etc.).
        threads (int):
            Number of CPU threads to use for PyTorch operations and data loading.
        output_path (str):
            Where to save the final trained model checkpoints and any ancillary outputs.

    Returns:
        nn.Module:
            The final trained model with the best checkpoint loaded.
            (Primarily for additional inference in the calling scope.)
    """
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
    
    if use_loyfer:
        validation_dls = {}
        y_vals = {}
        for cov in ['high','med','low']:
            tier1_dl, t1_yval = get_validation_set(str(Path(eval_pat_dir+"_"+cov) / "tier1"), atlas, names)
            tcells_dl, tcells_yval = get_validation_set(str(Path(eval_pat_dir+"_"+cov) / "T-cells"), atlas, names)
            oac_dl, oac_yval = get_validation_set(str(Path(eval_pat_dir+"_"+cov) / "OAC"), atlas, names)

            validation_dls[f"tier1_{cov}"] = tier1_dl
            validation_dls[f"t-cells_{cov}"] = tcells_dl
            validation_dls[f"oac_{cov}"] = oac_dl
            
            y_vals[f"tier1_{cov}"] = t1_yval
            y_vals[f"t-cells_{cov}"] = tcells_yval
            y_vals[f"oac_{cov}"] = oac_yval
    else:
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
    model = CellTypeDeconvolutionModel(
        num_markers=len(atlas),
        num_cell_types=len(cell_types),
        target_ids=target_ids,
        presence_models_dir=presence_models_dir,
    )

    # 6) Train the model, saving best checkpoint to `output_path`
    model, best_threshold = train_model(
        model=model,
        train_loader=train_dl,
        val_loaders=validation_dls,
        model_path=output_path
    )
    
    # Print the threshold the training process found to be best
    print("best_threshold", best_threshold)

    # 7) Evaluate final model predictions on each validation set
    for tier in validation_dls.keys():
        tier_dl = validation_dls[tier]
        y_val = y_vals[tier]

        # Use the simple predict function to get proportions
        deep_conv_estimation = predict_with_consensus(model, tier_dl.dataset.fraction, tier_dl.dataset.coverage)

        # Evaluate with a custom evaluation function
        deep_conv_eval_metrics = evaluate_performance(y_val.detach().numpy(), deep_conv_estimation, cell_types)

        print(f"deepconv validation metrics for tier {tier}")
        log_metrics(deep_conv_eval_metrics)
    
    # Return the trained model for downstream usage
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
