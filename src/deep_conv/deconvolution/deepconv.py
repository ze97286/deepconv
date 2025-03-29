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
from deep_conv.deconvolution.model import CellTypeDeconvolutionModel, TissueDeconvolutionDataset, AugmentedTissueDataset, coverage_matched_augmentation
from deep_conv.deconvolution.train import train_model
from deep_conv.deconvolution.predict import predict_with_post_processing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

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
        num_workers=2,
        persistent_workers=True,
        shuffle=False
    )
    
    # Convert y_val to a PyTorch tensor and normalize each row
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_val = y_val / y_val.sum(dim=1, keepdim=True)
    
    return val_loader, y_val

def get_validation_set_with_augmentation(
    eval_pat_dir: str, 
    atlas: pd.DataFrame, 
    names: set,
    target_dist_params=None,
    enable_augmentation=True
) -> Tuple[DataLoader, torch.Tensor]:
    """
    Enhanced validation set loader with coverage augmentation.
    
    Args:
        eval_pat_dir: Directory with validation data
        atlas: DataFrame with marker metadata
        names: Set of marker names to include
        target_dist_params: Target coverage distribution parameters
        enable_augmentation: Whether to enable augmentation
        
    Returns:
        val_loader: DataLoader with augmentation capability
        y_val: Ground truth labels
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
    print(f"Validation set {Path(eval_pat_dir).name} - Original coverage stats:")
    print("  Mean:", np.mean(coverage_val))
    print("  Median:", np.median(coverage_val))
    
    # Convert label DataFrame to numpy
    y_val_np = y_val.to_numpy()
    
    # Create augmented dataset
    val_dataset = AugmentedTissueDataset(
        X_val,
        coverage_val,
        atlas[atlas.columns[8:]].T.to_numpy(),
        y_val_np,
        target_dist_params=target_dist_params,
        augmentation_probability=1.0 if enable_augmentation else 0.0,
        enable_augmentation=enable_augmentation
    )
    
    # For validation, we create two variants:
    
    # 1. Regular validation dataset (no augmentation)
    val_dataset.set_training(False)  # Disable augmentation for standard evaluation
    
    # 2. Clinical-like validation dataset (with augmentation)
    # Create a separate dataset for clinical-like validation
    clinical_val_dataset = AugmentedTissueDataset(
        X_val,
        coverage_val,
        atlas[atlas.columns[8:]].T.to_numpy(),
        y_val_np,
        target_dist_params=target_dist_params,
        augmentation_probability=1.0,  # Apply to all samples
        enable_augmentation=enable_augmentation
    )
    clinical_val_dataset.set_training(True)  # Enable augmentation
    
    # Create DataLoaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=2,
        persistent_workers=True,
        shuffle=False
    )
    
    clinical_val_loader = DataLoader(
        clinical_val_dataset,
        batch_size=512,
        num_workers=2,
        persistent_workers=True,
        shuffle=False
    )
    
    # Convert y_val to a PyTorch tensor and normalize each row
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)
    y_val_tensor = y_val_tensor / y_val_tensor.sum(dim=1, keepdim=True)
    
    return val_loader, clinical_val_loader, y_val_tensor


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
    suffixes = [f"_batch{i}" for i in range(1, (num_files + 1)*3)]
    
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
   

def load_training_with_augmentation(
    base_dir: str, 
    atlas: pd.DataFrame, 
    names: set, 
    num_files: int = 5,
    enable_augmentation: bool = True,
    target_dist_params: dict = None,
    augmentation_probability: float = 0.5
) -> DataLoader:
    """
    Enhanced training data loader with coverage augmentation.
    
    Args:
        base_dir: Path prefix for training parquet files
        atlas: DataFrame with marker metadata and cell type columns
        names: Set of marker names to keep
        num_files: Number of parquet file batches to merge
        enable_augmentation: Whether to enable coverage augmentation
        target_dist_params: Parameters for target clinical coverage distribution
        augmentation_probability: Probability of applying augmentation
        
    Returns:
        DataLoader: Enhanced DataLoader with augmentation capability
    """
    # Load data using the original method
    markers, coverage, y = [], [], []
    
    print("loading training from", base_dir)
    suffixes = [f"_batch{i}" for i in range(1, (num_files + 1)*3)]
    
    # Read multiple parquet files and accumulate marker values, coverage, and ground-truth
    for cov in ['high', 'med', 'low']:
        for i in range(1, num_files + 1):
            markers.append(pd.read_parquet(f"{base_dir}_{cov}/{str(i)}_marker_values.parquet"))
            coverage.append(pd.read_parquet(f"{base_dir}_{cov}/{str(i)}_coverage.parquet"))
            y.append(pd.read_parquet(f"{base_dir}_{cov}/{str(i)}_ground_truth_y.parquet"))
    
    # Merge and process as in the original function
    merged_markers = markers[0]
    for i, m in enumerate(markers[1:]):
        merged_markers = merged_markers.merge(
            m, on=['name', 'direction'], how='outer', suffixes=('', suffixes[i])
        )
    
    merged_coverage = coverage[0]
    for i, c in enumerate(coverage[1:]):
        merged_coverage = merged_coverage.merge(
            c, on=['name', 'direction'], how='outer', suffixes=('', suffixes[i])
        )
    
    # Concatenate labels
    y = pd.concat(y, ignore_index=True).fillna(0)
    
    # Filter markers
    X_train = merged_markers[merged_markers.name.isin(names)]
    coverage_train = merged_coverage[merged_coverage.name.isin(names)]
    
    # Process data
    X_train = X_train.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_train = coverage_train.drop(columns=["name", "direction"]).T.to_numpy()
    y_train = y.to_numpy()
    
    # Print coverage stats before augmentation
    print("Original coverage statistics:")
    print("  Mean:", np.mean(coverage_train))
    print("  Median:", np.median(coverage_train))
    print("  5th percentile:", np.percentile(coverage_train, 5))
    print("  95th percentile:", np.percentile(coverage_train, 95))
    
    # Create the augmented dataset
    train_dataset = AugmentedTissueDataset(
        X_train,
        coverage_train,
        atlas[atlas.columns[8:]].T.to_numpy(),
        y_train,
        target_dist_params=target_dist_params,
        augmentation_probability=augmentation_probability,
        enable_augmentation=enable_augmentation
    )
    
    # Enable training mode for augmentation
    train_dataset.set_training(True)
    
    print(f"training dataset has {len(y_train)} samples")

    # Return augmented DataLoader
    return DataLoader(
        train_dataset,
        batch_size=64,  
        shuffle=True,
        num_workers=24, 
        pin_memory=True,
        persistent_workers=True
    )


def analyze_coverage_distribution(data_loader):
    """
    Analyze the coverage distribution in a dataset for calibration.
    
    Args:
        data_loader: DataLoader containing coverage information
        
    Returns:
        Distribution parameters for use in augmentation
    """
    coverages = []
    
    # Collect coverage values
    for batch in data_loader:
        coverage = batch['coverage'].numpy()
        coverages.extend(coverage.flatten())
    
    # Convert to array and filter out zeros
    coverages = np.array(coverages)
    non_zero_coverages = coverages[coverages > 0]
    
    # Calculate statistics
    mean = np.mean(non_zero_coverages)
    median = np.median(non_zero_coverages)
    std = np.std(non_zero_coverages)
    
    # Calculate log statistics
    log_coverages = np.log1p(non_zero_coverages)
    log_mean = np.mean(log_coverages)
    log_std = np.std(log_coverages)
    
    # Calculate quantiles
    quantiles = np.percentile(non_zero_coverages, [5, 25, 50, 75, 95])
    
    # Summarize distribution
    dist_params = {
        'mean': mean,
        'median': median,
        'std': std,
        'log_params': {
            'mean': log_mean,
            'std': log_std
        },
        'quantiles': {
            '5%': quantiles[0],
            '25%': quantiles[1],
            '50%': quantiles[2],
            '75%': quantiles[3],
            '95%': quantiles[4]
        },
        'zero_rate': np.mean(coverages == 0)
    }
    
    print(f"Coverage Distribution Analysis:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Median: {median:.2f}")
    print(f"  Std Dev: {std:.2f}")
    print(f"  Zero Rate: {dist_params['zero_rate']*100:.2f}%")
    print(f"  Quantiles:")
    for q, val in dist_params['quantiles'].items():
        print(f"    {q}: {val:.2f}")
    
    return dist_params


def visualize_augmentation_effect(original_coverage, augmented_coverage, title="Coverage Augmentation Effect"):
    """
    Visualize the effect of coverage augmentation.
    
    Args:
        original_coverage: Original coverage values [samples, markers]
        augmented_coverage: Augmented coverage values [samples, markers]
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    # Calculate mean coverage per sample
    orig_mean = np.mean(original_coverage, axis=1)
    aug_mean = np.mean(augmented_coverage, axis=1)
    
    # Create histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot original coverage distribution
    axes[0, 0].hist(orig_mean, bins=50, alpha=0.7)
    axes[0, 0].set_title("Original Mean Coverage Distribution")
    axes[0, 0].set_xlabel("Mean Coverage")
    axes[0, 0].set_ylabel("Count")
    
    # Plot augmented coverage distribution
    axes[0, 1].hist(aug_mean, bins=50, alpha=0.7)
    axes[0, 1].set_title("Augmented Mean Coverage Distribution")
    axes[0, 1].set_xlabel("Mean Coverage")
    axes[0, 1].set_ylabel("Count")
    
    # Plot original vs augmented as scatter
    axes[1, 0].scatter(orig_mean, aug_mean, alpha=0.3)
    axes[1, 0].set_title("Original vs Augmented Coverage")
    axes[1, 0].set_xlabel("Original Mean Coverage")
    axes[1, 0].set_ylabel("Augmented Mean Coverage")
    axes[1, 0].plot([0, max(orig_mean)], [0, max(orig_mean)], 'r--')  # Diagonal line
    
    # Plot original and augmented log histograms
    axes[1, 1].hist(np.log1p(orig_mean), bins=50, alpha=0.5, label="Original")
    axes[1, 1].hist(np.log1p(aug_mean), bins=50, alpha=0.5, label="Augmented")
    axes[1, 1].set_title("Log Coverage Comparison")
    axes[1, 1].set_xlabel("Log(Mean Coverage + 1)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()
    
    # Add title and adjust layout
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def test_augmentation(train_pat_dir, atlas, names, target_dist_params=None):
    """
    Generate test augmentations and visualize their effect.
    
    Args:
        train_pat_dir: Directory containing training files
        atlas: DataFrame with marker metadata
        names: Set of marker names to include
        target_dist_params: Target coverage distribution parameters
    
    Returns:
        Matplotlib figure showing the augmentation effect
    """
    # Load a small subset of training data
    markers = []
    coverage = []
    y = []
    
    # Just load one batch for testing
    cov = 'high'  # Start with high coverage to see the effect clearly
    i = 1
    markers.append(pd.read_parquet(f"{train_pat_dir}_{cov}/{str(i)}_marker_values.parquet"))
    coverage.append(pd.read_parquet(f"{train_pat_dir}_{cov}/{str(i)}_coverage.parquet"))
    y.append(pd.read_parquet(f"{train_pat_dir}_{cov}/{str(i)}_ground_truth_y.parquet"))
    
    # Process data
    X_train = markers[0][markers[0].name.isin(names)]
    coverage_train = coverage[0][coverage[0].name.isin(names)]
    
    X_train = X_train.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_train = coverage_train.drop(columns=["name", "direction"]).T.to_numpy()
    y_train = y[0].to_numpy()
    
    # Apply augmentation to all samples (100% probability)
    aug_X_train, aug_coverage_train = coverage_matched_augmentation(
        X_train, 
        coverage_train, 
        target_dist_params=target_dist_params,
        augmentation_probability=1.0  # Apply to all samples for testing
    )
    
    # Create the visualization
    fig = visualize_augmentation_effect(
        coverage_train, 
        aug_coverage_train,
        title="Coverage Augmentation Effect on Training Data"
    )
    
    # Print some statistics
    print(f"Original coverage - Mean: {np.mean(coverage_train):.2f}, Median: {np.median(coverage_train):.2f}")
    print(f"Augmented coverage - Mean: {np.mean(aug_coverage_train):.2f}, Median: {np.median(aug_coverage_train):.2f}")
    
    return fig


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

    # 2) Build the training DataLoader from parquet files in train_pat_dir with coverage augmentation
    clinical_dist_params = {
        'mean': 5.0,
        'std': 4.0,
        'log_params': {
            'mean': 1.2,
            'std': 0.8
        },
        'quantiles': {
            '5%': 0.5,
            '25%': 2.0,
            '50%': 4.0,
            '75%': 7.0,
            '95%': 12.0
        },
        'zero_rate': 0.1
    }

    train_dl = load_training_with_augmentation(
        train_pat_dir, 
        atlas, 
        names,
        enable_augmentation=True,
        target_dist_params=clinical_dist_params,
        augmentation_probability=0.5  # Adjust this as needed
    )
    analyze_coverage_distribution(train_dl)
    import matplotlib.pyplot as plt
    # Call the test function
    fig = test_augmentation(
        train_pat_dir, 
        atlas, 
        names,
        target_dist_params=clinical_dist_params
    )

    # save the figure
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_path/"augmentation_effect.png", dpi=300)
        
    # train_dl = load_training(train_pat_dir, atlas, names)

    # 3) Build DataLoaders for each validation subset
    
    if use_loyfer:
        validation_dls = {}
        clinical_validation_dls = {}  # New clinical-like validation set
        y_vals = {}
        
        for cov in ['high','med','low']:
            # For each validation set, get both standard and clinical variants
            tier1_dl, tier1_clinical_dl, t1_yval = get_validation_set_with_augmentation(
                str(Path(eval_pat_dir+"_"+cov) / "tier1"), 
                atlas, names, 
                target_dist_params=clinical_dist_params
            )

            print(f"validation set for {cov} tier1 length={len(t1_yval)}")
            
            tcells_dl, tcells_clinical_dl, tcells_yval = get_validation_set_with_augmentation(
                str(Path(eval_pat_dir+"_"+cov) / "T-cells"), 
                atlas, names,
                target_dist_params=clinical_dist_params
            )

            print(f"validation set for {cov} tcells length={len(tcells_yval)}")
            
            oac_dl, oac_clinical_dl, oac_yval = get_validation_set_with_augmentation(
                str(Path(eval_pat_dir+"_"+cov) / "OAC"), 
                atlas, names,
                target_dist_params=clinical_dist_params
            )

            print(f"validation set for {cov} oac length={len(oac_yval)}")

            # Store both standard and clinical variants
            validation_dls[f"tier1_{cov}"] = tier1_dl
            validation_dls[f"t-cells_{cov}"] = tcells_dl
            validation_dls[f"oac_{cov}"] = oac_dl
            
            clinical_validation_dls[f"tier1_{cov}_clinical"] = tier1_clinical_dl
            clinical_validation_dls[f"t-cells_{cov}_clinical"] = tcells_clinical_dl
            clinical_validation_dls[f"oac_{cov}_clinical"] = oac_clinical_dl

            y_vals[f"tier1_{cov}"] = t1_yval
            y_vals[f"t-cells_{cov}"] = tcells_yval
            y_vals[f"oac_{cov}"] = oac_yval
            
            # Use the same ground truth for clinical variants
            y_vals[f"tier1_{cov}_clinical"] = t1_yval
            y_vals[f"t-cells_{cov}_clinical"] = tcells_yval
            y_vals[f"oac_{cov}_clinical"] = oac_yval
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
    combined_val_loaders = {**validation_dls, **clinical_validation_dls}
    model, best_threshold = train_model(
        model=model,
        train_loader=train_dl,
        val_loaders=combined_val_loaders,
        model_path=output_path
    )
    
    # Print the threshold the training process found to be best
    print("best_threshold", best_threshold)

    # Evaluate on both standard and clinical validation sets
    print("\nStandard Validation Sets:")
    for tier in validation_dls.keys():
        tier_dl = validation_dls[tier]
        y_val = y_vals[tier]
        
        # Get marker to cell type mapping
        marker_to_cell_mapping = model.target_ids.cpu().numpy()
        
        # Use predict_with_post_processing instead of predict_with_consensus
        deep_conv_estimation = predict_with_post_processing(
            model, 
            tier_dl.dataset.fraction,
            tier_dl.dataset.coverage,
            marker_to_cell_mapping,
            min_coverage_threshold=5.0,
            min_signal_threshold=0.01
        )
        
        deep_conv_eval_metrics = evaluate_performance(
            y_val.detach().numpy(), 
            deep_conv_estimation, 
            cell_types
        )
        
        print(f"Standard validation metrics for tier {tier}")
        log_metrics(deep_conv_eval_metrics)

    print("\nClinical-Like Validation Sets:")
    for tier in clinical_validation_dls.keys():
        tier_dl = clinical_validation_dls[tier]
        y_val = y_vals[tier]
        
        # Get marker to cell type mapping
        marker_to_cell_mapping = model.target_ids.cpu().numpy()
        
        # Use predict_with_post_processing with stricter thresholds for clinical data
        deep_conv_estimation = predict_with_post_processing(
            model, 
            tier_dl.dataset.fraction,
            tier_dl.dataset.coverage,
            marker_to_cell_mapping,
            min_coverage_threshold=3.0,  # Possibly lower threshold for clinical data
            min_signal_threshold=0.02    # Possibly higher threshold to be more conservative
        )
        
        deep_conv_eval_metrics = evaluate_performance(
            y_val.detach().numpy(), 
            deep_conv_estimation, 
            cell_types
        )
        
        print(f"Clinical-like validation metrics for tier {tier}")
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
