import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from deep_conv.benchmark.benchmark_utils import *
from deep_conv.deconvolution.model import *
from deep_conv.deconvolution.train import train_model
from deep_conv.deconvolution.predict import *

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_validation_set_with_augmentation(
    eval_pat_dir: str, 
    atlas: pd.DataFrame, 
    names: set,
    block_size: int,
    target_dist_params=None,
    enable_augmentation=True,
    target_size: int = None,
    cell_types=None,
    filter_tcells_below: float = 0.0,
    filter_oac_below: float = 0.0
) -> tuple[DataLoader, torch.Tensor]:
    """
    Validation set loader with pre-augmented data, block-based subsampling,
    and optional filtering of T-cells and OAC samples below concentration thresholds.
    
    Args:
        eval_pat_dir: Directory with validation data
        atlas: DataFrame with marker metadata
        names: Set of marker names to include
        block_size: Size of each block (e.g., 10,000 for T-cells, 1,000 for OAC)
        target_dist_params: Target coverage distribution parameters
        enable_augmentation: Whether to enable augmentation (pre-augmentation will be used)
        target_size: Target number of samples to subsample (if None, use full dataset)
        cell_types: List of cell type names (to identify T-cells and OAC columns)
        filter_tcells_below: Concentration threshold below which to filter T-cell samples
        filter_oac_below: Concentration threshold below which to filter OAC samples
        
    Returns:
        val_loader: DataLoader with pre-augmented data
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
    
    # Filter T-cells or OAC samples below their respective thresholds (if applicable)
    if cell_types is not None:
        # Filter T-cells
        if "T-cells" in Path(eval_pat_dir).name and filter_tcells_below > 0:
            tcells_idx = cell_types.index("T-cells") if "T-cells" in cell_types else -1
            if tcells_idx >= 0:
                print(f"Filtering T-cell samples with concentration below {filter_tcells_below}...")
                tcells_concentration = y_val_np[:, tcells_idx]
                keep_indices = np.where(tcells_concentration >= filter_tcells_below)[0]
                print(f"Original number of samples: {len(y_val_np)}")
                print(f"Number of samples after T-cells filtering: {len(keep_indices)}")
                
                # Apply filtering
                X_val = X_val[keep_indices]
                coverage_val = coverage_val[keep_indices]
                y_val_np = y_val_np[keep_indices]
        
        # Filter OAC
        if "OAC" in Path(eval_pat_dir).name and filter_oac_below > 0:
            oac_idx = cell_types.index("OAC") if "OAC" in cell_types else -1
            if oac_idx >= 0:
                print(f"Filtering OAC samples with concentration below {filter_oac_below}...")
                oac_concentration = y_val_np[:, oac_idx]
                keep_indices = np.where(oac_concentration >= filter_oac_below)[0]
                print(f"Original number of samples: {len(y_val_np)}")
                print(f"Number of samples after OAC filtering: {len(keep_indices)}")
                
                # Apply filtering
                X_val = X_val[keep_indices]
                coverage_val = coverage_val[keep_indices]
                y_val_np = y_val_np[keep_indices]
    
    # Block-based subsampling (if target_size is specified)
    if target_size is not None and target_size < len(y_val_np):
        # Calculate the number of blocks
        dataset_size = len(y_val_np)
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
        
        # Subsample the data directly
        X_val = X_val[indices]
        coverage_val = coverage_val[indices]
        y_val_np = y_val_np[indices]
    
    # Pre-augment 50% of the dataset if augmentation is enabled
    if enable_augmentation:
        num_samples = len(y_val_np)
        num_to_augment = num_samples // 2
        indices_to_augment = np.random.choice(num_samples, num_to_augment, replace=False)

        # Temporary dataset for augmentation
        temp_dataset = TissueDeconvolutionDataset(X_val, coverage_val, atlas[atlas.columns[8:]].T.to_numpy(), y_val_np)

        augmented_fraction = []
        augmented_coverage = []
        for idx in indices_to_augment:
            item = temp_dataset[idx]
            fraction_np = item['X'].numpy().reshape(1, -1)
            coverage_np = item['coverage'].numpy().reshape(1, -1)
            aug_fraction, aug_coverage = coverage_matched_augmentation(
                fraction_np,
                coverage_np,
                target_dist_params,
                augmentation_prob=1.0
            )
            augmented_fraction.append(aug_fraction[0])
            augmented_coverage.append(aug_coverage[0])

        # Combine original and augmented data
        augmented_fraction = np.stack(augmented_fraction)
        augmented_coverage = np.stack(augmented_coverage)
        combined_fraction = np.concatenate([X_val[:num_to_augment], augmented_fraction])
        combined_coverage = np.concatenate([coverage_val[:num_to_augment], augmented_coverage])
        combined_y = np.concatenate([y_val_np[:num_to_augment], y_val_np[indices_to_augment]])

        # Save to disk with a unique name based on eval_pat_dir
        dataset_name = Path(eval_pat_dir).name
        np.save(f"pre_augmented_fraction_{dataset_name}.npy", combined_fraction)
        np.save(f"pre_augmented_coverage_{dataset_name}.npy", combined_coverage)
        np.save(f"pre_augmented_y_{dataset_name}.npy", combined_y)

        # Create pre-augmented dataset
        val_dataset = PreAugmentedTissueDataset(
            combined_fraction,
            combined_coverage,
            atlas[atlas.columns[8:]].T.to_numpy(),
            combined_y
        )
    else:
        # No augmentation, use original data
        val_dataset = PreAugmentedTissueDataset(
            X_val,
            coverage_val,
            atlas[atlas.columns[8:]].T.to_numpy(),
            y_val_np
        )
    
    # Create DataLoader with shuffling
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        persistent_workers=False
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
    target_dist_params: dict = None,
) -> DataLoader:
    """
    Enhanced training data loader with pre-augmented data.
    
    Args:
        base_dir: Path prefix for training parquet files
        atlas: DataFrame with marker metadata
        names: Set of marker names to keep
        num_files: Number of parquet file batches to merge
        target_dist_params: Parameters for target clinical coverage distribution
        
    Returns:
        DataLoader: DataLoader over the pre-augmented training data with shuffling.
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
    
    # Augment 50% of the dataset
    num_samples = len(X_train)
    num_to_augment = num_samples // 2
    indices_to_augment = np.random.choice(num_samples, num_to_augment, replace=False)

    # Temporary dataset for augmentation
    temp_dataset = TissueDeconvolutionDataset(X_train, coverage_train, atlas[atlas.columns[8:]].T.to_numpy(), y_train)

    augmented_fraction = []
    augmented_coverage = []
    for idx in indices_to_augment:
        item = temp_dataset[idx]
        fraction_np = item['X'].numpy().reshape(1, -1)
        coverage_np = item['coverage'].numpy().reshape(1, -1)
        aug_fraction, aug_coverage = coverage_matched_augmentation(
            fraction_np,
            coverage_np,
            target_dist_params,
            augmentation_prob=1.0
        )
        augmented_fraction.append(aug_fraction[0])
        augmented_coverage.append(aug_coverage[0])

    # Combine original and augmented data
    augmented_fraction = np.stack(augmented_fraction)
    augmented_coverage = np.stack(augmented_coverage)
    combined_fraction = np.concatenate([X_train[:num_to_augment], augmented_fraction])
    combined_coverage = np.concatenate([coverage_train[:num_to_augment], augmented_coverage])
    combined_y = np.concatenate([y_train[:num_to_augment], y_train[indices_to_augment]])

    # Save to disk with a unique name based on base_dir
    dataset_name = Path(base_dir).name
    np.save(f"pre_augmented_fraction_{dataset_name}.npy", combined_fraction)
    np.save(f"pre_augmented_coverage_{dataset_name}.npy", combined_coverage)
    np.save(f"pre_augmented_y_{dataset_name}.npy", combined_y)

    # Load pre-augmented dataset
    pre_augmented_dataset = PreAugmentedTissueDataset(
        combined_fraction,
        combined_coverage,
        atlas[atlas.columns[8:]].T.to_numpy(),
        combined_y
    )
    print(f"Training dataset has {len(pre_augmented_dataset)} samples.")

    # Create DataLoader with shuffling
    train_dl = DataLoader(
        pre_augmented_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=28,
        pin_memory=False,
        persistent_workers=True
    )
    return train_dl

def enhanced_negative_examples(
    train_dl: DataLoader,
    cell_types: list,
    atlas: pd.DataFrame,
    sample_fraction: float = 0.01,
    target_dist_params: dict = None,
) -> DataLoader:
    """
    Enhance the training dataset by adding negative examples for each cell type.
    
    Args:
        train_dl: Original training DataLoader
        cell_types: List of cell type names
        atlas: DataFrame with marker metadata
        sample_fraction: Fraction of samples to use as negative examples per cell type
        target_dist_params: Target coverage distribution parameters
        
    Returns:
        DataLoader: Enhanced DataLoader with negative examples
    """
    # Extract the original dataset
    dataset = train_dl.dataset
    fraction = dataset.datasets[0].fraction.numpy()  # Assuming ConcatDataset
    coverage = dataset.datasets[0].coverage.numpy()
    y = dataset.datasets[0].y.numpy()
    for d in dataset.datasets[1:]:
        fraction = np.concatenate([fraction, d.fraction.numpy()])
        coverage = np.concatenate([coverage, d.coverage.numpy()])
        y = np.concatenate([y, d.y.numpy()])

    print("Original dataset size:", len(y), "samples")

    # Create negative examples for each cell type
    negative_fractions = []
    negative_coverages = []
    negative_ys = []
    print("Creating negative examples for each cell type...")
    for cell_idx, cell_type in enumerate(cell_types):
        print(f"Processing {cell_type} (index {cell_idx})...")
        # Find samples where this cell type is absent (proportion = 0)
        absent_mask = y[:, cell_idx] == 0
        absent_indices = np.where(absent_mask)[0]
        print(f"  Found {len(absent_indices)} samples with {cell_type} absent")

        # Select a fraction of these samples
        num_samples = int(len(absent_indices) * sample_fraction)
        selected_indices = np.random.choice(absent_indices, num_samples, replace=False)
        print(f"  Selected {num_samples} samples to use as negative examples")

        # Extract the selected samples
        selected_fraction = fraction[selected_indices]
        selected_coverage = coverage[selected_indices]
        selected_y = y[selected_indices]

        negative_fractions.append(selected_fraction)
        negative_coverages.append(selected_coverage)
        negative_ys.append(selected_y)

    # Combine all negative examples
    negative_fraction = np.concatenate(negative_fractions)
    negative_coverage = np.concatenate(negative_coverages)
    negative_y = np.concatenate(negative_ys)

    # Pre-augment 50% of the negative examples
    num_negative_samples = len(negative_y)
    num_to_augment = num_negative_samples // 2
    indices_to_augment = np.random.choice(num_negative_samples, num_to_augment, replace=False)

    # Temporary dataset for augmentation
    temp_dataset = TissueDeconvolutionDataset(
        negative_fraction,
        negative_coverage,
        atlas[atlas.columns[8:]].T.to_numpy(),
        negative_y
    )

    augmented_fraction = []
    augmented_coverage = []
    for idx in indices_to_augment:
        item = temp_dataset[idx]
        fraction_np = item['X'].numpy().reshape(1, -1)
        coverage_np = item['coverage'].numpy().reshape(1, -1)
        aug_fraction, aug_coverage = coverage_matched_augmentation(
            fraction_np,
            coverage_np,
            target_dist_params,
            augmentation_prob=1.0
        )
        augmented_fraction.append(aug_fraction[0])
        augmented_coverage.append(aug_coverage[0])

    # Combine original and augmented negative examples
    augmented_fraction = np.stack(augmented_fraction)
    augmented_coverage = np.stack(augmented_coverage)
    combined_negative_fraction = np.concatenate([negative_fraction[:num_to_augment], augmented_fraction])
    combined_negative_coverage = np.concatenate([negative_coverage[:num_to_augment], augmented_coverage])
    combined_negative_y = np.concatenate([negative_y[:num_to_augment], negative_y[indices_to_augment]])

    # Save to disk
    np.save("pre_augmented_negative_fraction.npy", combined_negative_fraction)
    np.save("pre_augmented_negative_coverage.npy", combined_negative_coverage)
    np.save("pre_augmented_negative_y.npy", combined_negative_y)

    # Create pre-augmented dataset for negative examples
    negative_dataset = PreAugmentedTissueDataset(
        combined_negative_fraction,
        combined_negative_coverage,
        atlas[atlas.columns[8:]].T.to_numpy(),
        combined_negative_y
    )

    # Combine original dataset with negative examples
    enhanced_dataset = ConcatDataset([train_dl.dataset, negative_dataset])
    print(f"Adding {len(negative_dataset)} negative examples to the dataset")
    print(f"Enhanced dataset created: {len(train_dl.dataset)} → {len(enhanced_dataset)} samples")

    # Create new DataLoader
    enhanced_train_dl = DataLoader(
        enhanced_dataset,
        batch_size=train_dl.batch_size,
        shuffle=True,
        num_workers=train_dl.num_workers,
        pin_memory=train_dl.pin_memory,
        persistent_workers=train_dl.persistent_workers
    )

    return enhanced_train_dl

def train_and_eval(
    atlas_path: str,
    train_pat_dir: str,
    eval_pat_dir: str,
    threads: int,
    output_path: str,
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

    cell_types = list(atlas.columns[8:]) 

    train_dl_low = load_training_with_augmentation(
        f"{train_pat_dir}_low", atlas, names, num_files=3,
        target_dist_params=clinical_dist_params['low'],
    )
    train_dl_med = load_training_with_augmentation(
        f"{train_pat_dir}_med", atlas, names, num_files=1,
        target_dist_params=clinical_dist_params['med'],
    )
    train_dl_high = load_training_with_augmentation(
        f"{train_pat_dir}_high", atlas, names, num_files=1,
        target_dist_params=clinical_dist_params['high'],
    )

    train_dataset = ConcatDataset([train_dl_low.dataset, train_dl_med.dataset, train_dl_high.dataset])
    train_dl = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=28,
        pin_memory=False,
        persistent_workers=True
    )

    validation_dls = {}
    y_vals = {}
    for cov in ['high', 'med', 'low', 'clinical']:
        tier1_dl, t1_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "tier1"), atlas, names,
            block_size=50_000,
            target_dist_params=clinical_dist_params[cov],
            enable_augmentation=True,
            target_size=20_000,
            cell_types=cell_types,
            filter_tcells_below=0.0,
            filter_oac_below=0.0
        )
        print(f"Validation set {cov} tier1 length={len(t1_yval)}")

        tcells_dl, tcells_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "T-cells"), atlas, names,
            block_size=10_000,
            target_dist_params=clinical_dist_params[cov],
            enable_augmentation=True,
            target_size=15_000,
            cell_types=cell_types,
            filter_tcells_below=0.006,
            filter_oac_below=0.0
        )
        print(f"Validation set {cov} tcells length={len(tcells_yval)}")

        oac_dl, oac_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "OAC"), atlas, names,
            block_size=1_000,
            target_dist_params=clinical_dist_params[cov],
            enable_augmentation=True,
            target_size=2_000,
            cell_types=cell_types,
            filter_tcells_below=0.0,
            filter_oac_below=0.001
        )
        print(f"Validation set {cov} oac length={len(oac_yval)}")

        validation_dls[f"tier1_{cov}"] = tier1_dl
        validation_dls[f"t-cells_{cov}"] = tcells_dl
        validation_dls[f"oac_{cov}"] = oac_dl
        y_vals[f"tier1_{cov}"] = t1_yval
        y_vals[f"t-cells_{cov}"] = tcells_yval
        y_vals[f"oac_{cov}"] = oac_yval

    # Enhance negative examples
    cell_types = list(atlas.columns[8:])
    enhanced_train_dl = enhanced_negative_examples(
        train_dl,
        cell_types,
        atlas,
        sample_fraction=0.01,
        target_dist_params=clinical_dist_params['clinical'],
    )

    # Create the model
    target_ids = atlas["target"].map(lambda x: cell_types.index(x)).to_numpy()
    model = CellTypeDeconvolutionModel(
        num_markers=len(atlas), num_cell_types=len(cell_types),
        target_ids=target_ids, presence_models_dir=presence_models_dir
    )

    # Train the model
    model, _ = train_model(
        model=model,
        train_loader=enhanced_train_dl,
        val_loaders=validation_dls,
        model_path=output_path,
        num_epochs=1000,
        patience=20, 
        lr=5e-4, 
        weight_decay=1e-4 
    )

    # Evaluate
    print("\nStandard Validation Sets:")
    for tier in validation_dls.keys():
        tier_dl = validation_dls[tier]
        y_val = y_vals[tier]
        # Use the DataLoader for prediction
        deep_conv_estimations = []
        for batch in tier_dl:
            fraction = batch['X']
            coverage = batch['coverage']
            batch_estimation = predict_with_consensus(model, fraction, coverage)
            deep_conv_estimations.append(batch_estimation)
        deep_conv_estimation = np.concatenate(deep_conv_estimations, axis=0)
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
    parser.add_argument("--num_threads", required=False, type=int, default=32)
    parser.add_argument("--presence_models_dir", type=str, required=True)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    
    args = parser.parse_args()
    
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    
    train_and_eval(
        args.atlas_path,
        args.train_path + "/train",
        args.eval_path + "/eval",
        args.num_threads,
        args.output_path,
        args.presence_models_dir
    )

if __name__ == "__main__":    
    main()