import argparse
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Tuple 

from deep_conv.benchmark.benchmark_utils import *
from deep_conv.deconvolution.model import *
from deep_conv.deconvolution.train import train_model
from deep_conv.deconvolution.predict import *
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Sampler

class RandomSubsetSampler(Sampler[int]):
    """ 
    Randomly sample a fixed subset_size of indices each epoch 
    from a dataset of total length N.
    """
    def __init__(self, data_source, subset_size: int):
        self.data_source = data_source
        self.subset_size = min(subset_size, len(data_source))

    def __len__(self):
        # By definition, we'll yield subset_size each epoch
        return self.subset_size

    def __iter__(self):
        # Generate a random permutation of all indices
        all_indices = np.arange(len(self.data_source))
        np.random.shuffle(all_indices)
        # Take the first subset_size
        chosen = all_indices[:self.subset_size]
        return iter(chosen.tolist())
    

class FixedBlockSampler(Sampler[int]):
    """
    Samples a fixed number of items from each contiguous block in a dataset
    without requiring an explicit block_to_indices dict.

    Assumes dataset is physically laid out in contiguous blocks:
      - Block 0: indices [0 .. block_size-1]
      - Block 1: indices [block_size .. 2*block_size-1]
      - ...
    up to the final block which may have fewer than 'block_size' items if
    dataset_size is not a multiple of block_size.
    """

    def __init__(self, dataset_size: int, block_size: int,
                 samples_per_block: int, shuffle_within_block: bool = False):
        """
        Args:
            dataset_size (int): Total number of samples in the dataset.
            block_size (int): The size of each contiguous block.
            samples_per_block (int): How many samples to take from each block.
            shuffle_within_block (bool): If True, randomly shuffle indices
                                         within each block before taking samples_per_block.
        """
        self.dataset_size = dataset_size
        self.block_size = block_size
        self.samples_per_block = samples_per_block
        self.shuffle_within_block = shuffle_within_block

        self.final_indices = []
        start_idx = 0

        # Partition the dataset into blocks of size 'block_size'
        while start_idx < dataset_size:
            end_idx = min(start_idx + block_size, dataset_size)
            block_indices = np.arange(start_idx, end_idx)

            # Optionally shuffle within the block
            if self.shuffle_within_block:
                np.random.shuffle(block_indices)

            # Take the first 'samples_per_block' from this block
            # (or all if the block size < samples_per_block)
            chosen = block_indices[:samples_per_block]
            self.final_indices.extend(chosen)

            start_idx += block_size

        # Sort final indices so iteration order is stable
        self.final_indices.sort()

    def __iter__(self):
        return iter(self.final_indices)

    def __len__(self):
        return len(self.final_indices)
    

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_validation_set_with_augmentation(
    eval_pat_dir: str, 
    atlas: pd.DataFrame, 
    names: set,
    block_size: int,
    target_dist_params=None,
    enable_augmentation=True,
    target_size: int = None
) -> tuple[DataLoader, torch.Tensor]:
    """
    Validation set loader with light coverage augmentation and block-based subsampling.
    
    Args:
        eval_pat_dir: Directory with validation data
        atlas: DataFrame with marker metadata
        names: Set of marker names to include
        block_size: Size of each block (e.g., 10,000 for T-cells, 1,000 for OAC)
        target_dist_params: Target coverage distribution parameters
        enable_augmentation: Whether to enable augmentation
        target_size: Target number of samples to subsample (if None, use full dataset)
        
    Returns:
        val_loader: DataLoader with light augmentation
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
    
    # Create dataset with light augmentation
    val_dataset = AugmentedTissueDataset(
        X_val,
        coverage_val,
        atlas[atlas.columns[8:]].T.to_numpy(),
        y_val_np,
        target_dist_params=target_dist_params,
        augmentation_probability=0.3 if enable_augmentation else 0.0,
        enable_augmentation=enable_augmentation
    )
    
    # Enable augmentation before subsampling
    val_dataset.set_training(True)  # Moved before subsampling
    
    # Block-based subsampling
    if target_size is not None and target_size < len(val_dataset):
        # Calculate the number of blocks
        dataset_size = len(val_dataset)
        num_blocks = (dataset_size + block_size - 1) // block_size  # Ceiling division
        if num_blocks == 0:
            num_blocks = 1
        
        # Calculate samples to take from each block
        samples_per_block = target_size // num_blocks
        if samples_per_block == 0:
            samples_per_block = 1  # Ensure at least 1 sample per block
        
        indices = []
        for block_idx in range(num_blocks):
            start_idx = block_idx * block_size
            end_idx = min((block_idx + 1) * block_size, dataset_size)
            block_indices = np.arange(start_idx, end_idx)
            np.random.shuffle(block_indices)
            # Take up to samples_per_block, or fewer if the block is smaller
            selected_indices = block_indices[:min(samples_per_block, len(block_indices))]
            indices.extend(selected_indices)
        
        # Adjust for any rounding errors (if we have fewer samples than target_size)
        if len(indices) < target_size:
            remaining = target_size - len(indices)
            all_indices = np.arange(dataset_size)
            remaining_indices = np.setdiff1d(all_indices, indices)
            np.random.shuffle(remaining_indices)
            indices.extend(remaining_indices[:remaining])
        
        # Sort indices to maintain order
        indices = np.sort(indices)
        val_dataset = Subset(val_dataset, indices)
        y_val_np = y_val_np[indices]
    
    # Create DataLoader with shuffling
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=2,
        persistent_workers=True
    )
    
    # Convert y_val to a PyTorch tensor and normalize each row
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)
    y_val_tensor = y_val_tensor / y_val_tensor.sum(dim=1, keepdim=True)
    
    return val_loader, y_val_tensor


def load_training_with_augmentation(
    base_dir: str, 
    atlas: pd.DataFrame, 
    names: set, 
    num_files: int = 5,
    enable_augmentation: bool = True,
    target_dist_params: dict = None,
    augmentation_probability: float = 0.8
) -> DataLoader:
    """
    Enhanced training data loader with coverage augmentation.
    
    Args:
        base_dir: Path prefix for training parquet files
        atlas: DataFrame with marker metadata
        names: Set of marker names to keep
        num_files: Number of parquet file batches to merge
        enable_augmentation: Whether to enable coverage augmentation
        target_dist_params: Parameters for target clinical coverage distribution
        augmentation_probability: Probability of applying augmentation
        
    Returns:
        DataLoader: DataLoader over the full training data with shuffling.
    """
    # Load data
    markers, coverage, y = [], [], []
    print("loading training from", base_dir)
    for i in range(1, num_files + 1):
        markers.append(pd.read_parquet(f"{base_dir}/{str(i)}_marker_values.parquet"))
        coverage.append(pd.read_parquet(f"{base_dir}/{str(i)}_coverage.parquet"))
        y.append(pd.read_parquet(f"{base_dir}/{str(i)}_ground_truth_y.parquet"))
    
    merged_markers = markers[0]
    suffixes = [f"_batch{i}" for i in range(1, len(markers))]
    for i, m in enumerate(markers[1:]):
        merged_markers = merged_markers.merge(
            m, on=['name', 'direction'], how='outer', suffixes=('', suffixes[i])
        )
    
    merged_coverage = coverage[0]
    for i, c in enumerate(coverage[1:]):
        merged_coverage = merged_coverage.merge(
            c, on=['name', 'direction'], how='outer', suffixes=('', suffixes[i])
        )
    
    y = pd.concat(y, ignore_index=True).fillna(0)
    
    X_train = merged_markers[merged_markers.name.isin(names)]
    coverage_train = merged_coverage[merged_coverage.name.isin(names)]
    
    X_train = X_train.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_train = coverage_train.drop(columns=["name", "direction"]).T.to_numpy()
    y_train = y.to_numpy()
    
    print("Original coverage statistics:")
    print("  Mean:", np.mean(coverage_train))
    print("  Median:", np.median(coverage_train))
    print("  5th percentile:", np.percentile(coverage_train, 5))
    print("  95th percentile:", np.percentile(coverage_train, 95))
    
    train_dataset = AugmentedTissueDataset(
        X_train,
        coverage_train,
        atlas[atlas.columns[8:]].T.to_numpy(),
        y_train,
        target_dist_params=target_dist_params,
        augmentation_probability=augmentation_probability,
        enable_augmentation=enable_augmentation
    )
    
    train_dataset.set_training(True)
    
    dataset_size = len(train_dataset)
    print(f"Training dataset has {dataset_size} samples.")
    
    return DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        persistent_workers=True
    )


def enhanced_negative_examples(train_dl, cell_types, atlas, sample_fraction=0.01):
    """
    Create additional negative examples by applying coverage reduction and augmentation.
    
    Args:
        train_dl: Training DataLoader yielding dictionaries with keys 'X', 'coverage', and 'y'
        cell_types: List of cell type names (order must match columns in y)
        sample_fraction: Fraction of eligible negative examples to select for each cell type.
        
    Returns:
        Enhanced DataLoader over a new dataset with original and negative examples.
    """
    # Collect original data from the DataLoader
    all_X, all_coverage, all_y = [], [], []
    print("Collecting original data...")
    for batch in tqdm.tqdm(train_dl):
        all_X.append(batch['X'].numpy())
        all_coverage.append(batch['coverage'].numpy())
        all_y.append(batch['y'].numpy())
    
    X = np.vstack(all_X)          # shape [N, M]
    coverage = np.vstack(all_coverage)  # shape [N, M]
    y = np.vstack(all_y)          # shape [N, C]
    original_count = len(X)
    print(f"Original dataset size: {original_count} samples")
    
    # Lists to store negative variants
    neg_X_list = []
    neg_coverage_list = []
    neg_y_list = []
    
    total_added = 0
    
    print("Creating negative examples for each cell type...")
    for cell_idx, cell_type in enumerate(cell_types):
        print(f"Processing {cell_type} (index {cell_idx})...")
        # Find indices where cell type is absent
        neg_mask = y[:, cell_idx] < 0.001
        negative_indices = np.where(neg_mask)[0]
        samples_count = len(negative_indices)
        print(f"  Found {samples_count} samples with {cell_type} absent")
        if samples_count == 0:
            continue
        
        num_to_select = max(1, int(samples_count * sample_fraction))
        selected_indices = np.random.choice(negative_indices, num_to_select, replace=False)
        print(f"  Selected {num_to_select} samples to use as negative examples")
        
        # Extract selected samples
        sel_X = X[selected_indices]             # [N_sel, M]
        sel_cov = coverage[selected_indices]      # [N_sel, M]
        sel_y = y[selected_indices].copy()        # [N_sel, C]
        sel_y[:, cell_idx] = 0.0
        
        N_sel, M = sel_X.shape
        
        # Low Coverage Variant (50% reduction)
        low_cov = sel_cov * 0.5
        orig_reads = np.maximum(1, np.rint(sel_cov)).astype(np.int32)
        target_reads_low = np.maximum(1, np.rint(low_cov)).astype(np.int32)
        orig_methylated = np.rint(sel_X * orig_reads).astype(np.int32)
        mask_low = (sel_cov > 0) & (target_reads_low < orig_reads)
        low_X = sel_X.copy()
        if np.any(mask_low):
            sampled_low = np.random.hypergeometric(
                orig_methylated[mask_low],
                (orig_reads - orig_methylated)[mask_low],
                target_reads_low[mask_low]
            )
            low_X[mask_low] = sampled_low / target_reads_low[mask_low].astype(np.float32)
        
        # Append only the low coverage variant
        neg_X_list.append(low_X)
        neg_coverage_list.append(low_cov)
        neg_y_list.append(sel_y)
        
        total_added += N_sel
    
    if total_added > 0:
        print(f"Adding {total_added} negative examples to the dataset")
        combined_X = np.vstack([X] + neg_X_list)
        combined_coverage = np.vstack([coverage] + neg_coverage_list)
        combined_y = np.vstack([y] + neg_y_list)
    else:
        print("No negative examples were added")
        combined_X = X
        combined_coverage = coverage
        combined_y = y

    print(f"Enhanced dataset created: {len(X)} → {len(combined_X)} samples")
    print(f"Added {len(combined_X) - len(X)} explicit negative examples")
    
    # Create new dataset and dataloader
    enhanced_dataset = AugmentedTissueDataset(
        combined_X,
        combined_coverage,
        atlas[atlas.columns[8:]].T.to_numpy(),
        combined_y,
        target_dist_params=train_dl.dataset.target_dist_params,
        augmentation_probability=train_dl.dataset.augmentation_probability,
        enable_augmentation=train_dl.dataset.enable_augmentation
    )
    
    enhanced_dataset.set_training(True)
    
    enhanced_loader = DataLoader(
        enhanced_dataset,
        batch_size=train_dl.batch_size,
        shuffle=True,
        num_workers=getattr(train_dl, 'num_workers', 4),
        persistent_workers=getattr(train_dl, 'persistent_workers', False)
    )
    
    return enhanced_loader


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


def test_augmentation(base_dir, atlas, names, target_dist_params):
    # Load a small subset of data
    dataset = load_training_with_augmentation(
        base_dir, atlas, names, num_files=1, enable_augmentation=False, subset_size=1000
    )
    fraction = dataset.dataset.fraction
    coverage = dataset.dataset.coverage
    
    # Ensure coverage and fraction are 2D (num_samples, num_markers)
    if fraction.dim() > 2:
        fraction = fraction.reshape(fraction.shape[0], -1)
    if coverage.dim() > 2:
        coverage = coverage.reshape(coverage.shape[0], -1)
    
    # Apply augmentation
    aug_fraction, aug_coverage = coverage_matched_augmentation(
        fraction, coverage, target_dist_params, augmentation_prob=1.0
    )
    
    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Original mean coverage
    orig_means = coverage.mean(dim=1)  # Shape: (num_samples,)
    axes[0, 0].hist(orig_means.cpu().numpy() if isinstance(orig_means, torch.Tensor) else orig_means, 
                    bins=50, color='blue', alpha=0.7)
    axes[0, 0].set_title("Original Mean Coverage Distribution")
    axes[0, 0].set_xlabel("Mean Coverage")
    
    # Augmented mean coverage
    aug_means = aug_coverage.mean(dim=1)  # Shape: (num_samples,)
    axes[0, 1].hist(aug_means.cpu().numpy() if isinstance(aug_means, torch.Tensor) else aug_means, 
                    bins=50, color='blue', alpha=0.7)
    axes[0, 1].set_title("Augmented Mean Coverage Distribution")
    axes[0, 1].set_xlabel("Mean Coverage")
    
    # Scatter: Original vs. Augmented
    axes[1, 0].scatter(orig_means.cpu().numpy() if isinstance(orig_means, torch.Tensor) else orig_means, 
                       aug_means.cpu().numpy() if isinstance(aug_means, torch.Tensor) else aug_means, 
                       color='blue', alpha=0.5)
    axes[1, 0].plot([0, 14], [0, 14], 'r--')
    axes[1, 0].set_xlabel("Original Mean Coverage")
    axes[1, 0].set_ylabel("Augmented Mean Coverage")
    axes[1, 0].set_title("Original vs Augmented Coverage")
    
    # Per-marker coverage comparison (not per-sample mean)
    coverage_flat = coverage.flatten()
    aug_coverage_flat = aug_coverage.flatten()
    axes[1, 1].hist(coverage_flat.cpu().numpy() if isinstance(coverage_flat, torch.Tensor) else coverage_flat, 
                    bins=50, color='blue', alpha=0.5, label='Original', range=(0, 20))
    axes[1, 1].hist(aug_coverage_flat.cpu().numpy() if isinstance(aug_coverage_flat, torch.Tensor) else aug_coverage_flat, 
                    bins=50, color='orange', alpha=0.5, label='Augmented', range=(0, 20))
    axes[1, 1].set_xlabel("Coverage (Per Marker)")
    axes[1, 1].set_title("Per-Marker Coverage Comparison")
    axes[1, 1].legend()
    
    plt.tight_layout()
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
    set_seed()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    atlas = pd.read_csv(atlas_path, sep="\t")
    names = set(atlas.name.unique())

    clinical_dist_params = {
        'low': {
            'mean': 10.0, 'std': 6.0, 'log_params': {'mean': 2.08, 'std': 0.8},
            'quantiles': {'5%': 1.0, '25%': 4.0, '50%': 8.0, '75%': 12.0, '95%': 20.0},
            'zero_rate': 0.03
        },
        'med': {
            'mean': 25.0, 'std': 10.0, 'log_params': {'mean': 3.0, 'std': 0.7},
            'quantiles': {'5%': 5.0, '25%': 12.0, '50%': 20.0, '75%': 30.0, '95%': 50.0},
            'zero_rate': 0.004
        },
        'high': {
            'mean': 70.0, 'std': 20.0, 'log_params': {'mean': 4.2, 'std': 0.6},
            'quantiles': {'5%': 30.0, '25%': 50.0, '50%': 67.0, '75%': 85.0, '95%': 120.0},
            'zero_rate': 0.004
        },
        'clinical': {
            'mean': 5.0, 'std': 4.0, 'log_params': {'mean': 1.4, 'std': 0.9},
            'quantiles': {'5%': 0.5, '25%': 2.0, '50%': 4.0, '75%': 7.0, '95%': 12.0},
            'zero_rate': 0.2
        }
    }

    train_dl_low = load_training_with_augmentation(
        f"{train_pat_dir}_low", atlas, names, num_files=5,
        enable_augmentation=True, target_dist_params=clinical_dist_params['low'],
        augmentation_probability=0.8
    )
    train_dl_med = load_training_with_augmentation(
        f"{train_pat_dir}_med", atlas, names, num_files=4,
        enable_augmentation=True, target_dist_params=clinical_dist_params['med'],
        augmentation_probability=0.8
    )
    train_dl_high = load_training_with_augmentation(
        f"{train_pat_dir}_high", atlas, names, num_files=4,
        enable_augmentation=True, target_dist_params=clinical_dist_params['high'],
        augmentation_probability=0.8
    )

    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([train_dl_low.dataset, train_dl_med.dataset, train_dl_high.dataset])
    train_dl = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        persistent_workers=True
    )

    dist_params = analyze_coverage_distribution(train_dl)
    print("Training coverage distribution:", dist_params)

    import matplotlib.pyplot as plt
    fig = test_augmentation(train_pat_dir + "_low", atlas, names, clinical_dist_params['low'])
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "augmentation_effect.png"), dpi=300)

    validation_dls = {}
    y_vals = {}
    for cov in ['high', 'med', 'low', 'clinical']:
        tier1_dl, t1_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "tier1"), atlas, names,
            block_size=50_000,  # Block size for Tier1
            target_dist_params=clinical_dist_params[cov],
            enable_augmentation=True,
            target_size=20_000
        )
        print(f"Validation set {cov} tier1 length={len(t1_yval)}")

        tcells_dl, tcells_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "T-cells"), atlas, names,
            block_size=10_000,  # Block size matches the 7 types (10,000 each)
            target_dist_params=clinical_dist_params[cov],
            enable_augmentation=True,
            target_size=15_000
        )
        print(f"Validation set {cov} tcells length={len(tcells_yval)}")

        oac_dl, oac_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "OAC"), atlas, names,
            block_size=1_000,  # Block size matches the 12 sets (1,000 each)
            target_dist_params=clinical_dist_params[cov],
            enable_augmentation=True,
            target_size=2_000
        )
        print(f"Validation set {cov} oac length={len(oac_yval)}")

        validation_dls[f"tier1_{cov}"] = tier1_dl
        validation_dls[f"t-cells_{cov}"] = tcells_dl
        validation_dls[f"oac_{cov}"] = oac_dl
        y_vals[f"tier1_{cov}"] = t1_yval
        y_vals[f"t-cells_{cov}"] = tcells_yval
        y_vals[f"oac_{cov}"] = oac_yval

    # 5) Enhance negative examples
    cell_types = list(atlas.columns[8:])
    enhanced_train_dl = enhanced_negative_examples(train_dl, cell_types, atlas, sample_fraction=0.01)

    # 6) Create the model
    target_ids = atlas["target"].map(lambda x: cell_types.index(x)).to_numpy()
    model = CellTypeDeconvolutionModel(
        num_markers=len(atlas), num_cell_types=len(cell_types),
        target_ids=target_ids, presence_models_dir=presence_models_dir, feature_dim=64
    )

    # 7) Train the model
    model, _ = train_model(
        model=model, train_loader=enhanced_train_dl, val_loaders=validation_dls,
        model_path=output_path
    )

    # 8) Evaluate
    print("\nStandard Validation Sets:")
    for tier in validation_dls.keys():
        tier_dl = validation_dls[tier]
        y_val = y_vals[tier]
        deep_conv_estimation = predict_with_consensus(model, tier_dl.dataset.fraction, tier_dl.dataset.coverage)
        deep_conv_eval_metrics = evaluate_performance(y_val.detach().numpy(), deep_conv_estimation, cell_types)
        print(f"Standard validation metrics for tier {tier}")
        log_metrics(deep_conv_eval_metrics)

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
