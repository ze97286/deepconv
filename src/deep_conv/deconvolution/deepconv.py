import random
import pandas as pd
import numpy as np
import torch
import time
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
    enable_augmentation=True,
    clinical_coverage_params=None,
    model=None,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Validation set loader with pre-augmented data, creating both regular and clinical-like sets.
    Uses the full dataset with no subsampling and no shuffling.
    
    Args:
        eval_pat_dir: Directory with validation data
        atlas: DataFrame with marker metadata
        names: Set of marker names to include
        enable_augmentation: Whether to enable augmentation for clinical-like set
        clinical_coverage_params: Parameters for target clinical coverage distribution
        model: Model instance for presence computation
        
    Returns:
        val_loader: DataLoader for regular (unaugmented) validation set
        clinical_val_loader: DataLoader for clinical-like (augmented) validation set
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

    # Print coverage stats for debug/monitoring
    print(f"Validation set {Path(eval_pat_dir).name} - Original coverage stats:")
    print("  Mean:", np.mean(coverage_val))
    print("  Median:", np.median(coverage_val))

    # Convert label DataFrame to numpy
    y_val_np = y_val.to_numpy()

    # Compute x_nnls for original data
    atlas_np = atlas[atlas.columns[8:]].T.to_numpy()
    x_nnls_original = run_weighted_nnls(X_val, coverage_val, atlas_np)
    print("shape of x_nnls_original:", x_nnls_original.shape)

    # Create regular (unaugmented) validation set
    regular_dataset = PreAugmentedTissueDataset(
        X_val,
        coverage_val,
        y_val_np,
        x_nnls=x_nnls_original,
        model=model,
        is_clinical_like=False,
        is_augmented=np.zeros(len(X_val), dtype=bool)
    )

    # Create clinical-like (augmented) validation set
    if enable_augmentation:
        num_samples = len(y_val_np)
        indices_to_augment = np.random.choice(num_samples, num_samples, replace=False)

        temp_dataset = TissueDeconvolutionDataset(X_val, coverage_val, y_val_np)

        augmented_fraction = []
        augmented_coverage = []
        for idx in indices_to_augment:
            item = temp_dataset[idx]
            fraction_np = item['X'].numpy().reshape(1, -1)
            coverage_np = item['coverage'].numpy().reshape(1, -1)
            aug_fraction, aug_coverage = coverage_matched_augmentation(
                fraction_np,
                coverage_np,
                clinical_coverage_params=clinical_coverage_params,
                augmentation_prob=1.0
            )
            augmented_fraction.append(aug_fraction[0])
            augmented_coverage.append(aug_coverage[0])

        augmented_fraction = np.stack(augmented_fraction)
        augmented_coverage = np.stack(augmented_coverage)
        # Use only augmented samples for clinical-like set
        combined_fraction = augmented_fraction
        combined_coverage = augmented_coverage
        combined_y = y_val_np[indices_to_augment]
        combined_x_nnls = x_nnls_original[indices_to_augment]

        dataset_name = Path(eval_pat_dir).name
        np.save(f"pre_augmented_clinical_fraction_{dataset_name}.npy", combined_fraction)
        np.save(f"pre_augmented_clinical_coverage_{dataset_name}.npy", combined_coverage)
        np.save(f"pre_augmented_clinical_y_{dataset_name}.npy", combined_y)
        np.save(f"pre_augmented_clinical_x_nnls_{dataset_name}.npy", combined_x_nnls)

        clinical_dataset = PreAugmentedTissueDataset(
            combined_fraction,
            combined_coverage,
            combined_y,
            x_nnls=combined_x_nnls,
            model=model,
            is_clinical_like=True,
            is_augmented=np.ones(len(combined_y), dtype=bool)
        )
    else:
        clinical_dataset = regular_dataset

    # Create DataLoaders without shuffling
    val_loader = DataLoader(
        regular_dataset,
        batch_size=512,
        shuffle=False, 
        num_workers=16,
        pin_memory=False,
        persistent_workers=False
    )

    clinical_val_loader = DataLoader(
        clinical_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=16,
        pin_memory=False,
        persistent_workers=False
    )

    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)
    y_val_tensor = y_val_tensor / y_val_tensor.sum(dim=1, keepdim=True)

    return val_loader, clinical_val_loader, y_val_tensor

def load_training_with_augmentation(
    base_dir: str, 
    atlas: pd.DataFrame, 
    names: set, 
    num_files: int = 5,
    model=None,
    clinical_coverage_params=None,
) -> DataLoader:
    """
    Enhanced training data loader with pre-augmented data, using the full dataset with shuffling.
    
    Args:
        base_dir: Path prefix for training parquet files
        atlas: DataFrame with marker metadata
        names: Set of marker names to keep
        num_files: Number of parquet file batches to merge
        model: Model instance for presence computation
        clinical_coverage_params: Parameters for target clinical coverage distribution
        
    Returns:
        DataLoader: DataLoader over the pre-augmented training data.
    """
    # Load data
    markers, coverage, y = [], [], []
    print("loading training from", base_dir)

    for i in range(1, num_files + 1):
        try:
            markers.append(pd.read_parquet(f"{base_dir}/{str(i)}_marker_values.parquet"))
            coverage.append(pd.read_parquet(f"{base_dir}/{str(i)}_coverage.parquet"))
            y.append(pd.read_parquet(f"{base_dir}/{str(i)}_ground_truth_y.parquet"))
        except FileNotFoundError:
            continue
    
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
    
    # Compute x_nnls for original data
    atlas_np = atlas[atlas.columns[8:]].T.to_numpy()
    x_nnls_original = run_weighted_nnls(X_train, coverage_train, atlas_np)
    print("shape of x_nnls_original:", x_nnls_original.shape)
    
    # Augment 50% of the dataset
    num_samples = len(X_train)
    num_to_augment = num_samples // 2
    indices_to_augment = np.random.choice(num_samples, num_to_augment, replace=False)

    # Temporary dataset for augmentation
    temp_dataset = TissueDeconvolutionDataset(X_train, coverage_train, y_train)

    augmented_fraction = []
    augmented_coverage = []
    for idx in indices_to_augment:
        item = temp_dataset[idx]
        fraction_np = item['X'].numpy().reshape(1, -1)
        coverage_np = item['coverage'].numpy().reshape(1, -1)
        aug_fraction, aug_coverage = coverage_matched_augmentation(
            fraction_np,
            coverage_np,
            clinical_coverage_params=clinical_coverage_params,
            augmentation_prob=1.0
        )
        augmented_fraction.append(aug_fraction[0])
        augmented_coverage.append(aug_coverage[0])

    # Combine original and augmented data using consistent indices
    augmented_fraction = np.stack(augmented_fraction)
    augmented_coverage = np.stack(augmented_coverage)
    combined_fraction = np.concatenate([X_train[indices_to_augment], augmented_fraction])
    combined_coverage = np.concatenate([coverage_train[indices_to_augment], augmented_coverage])
    combined_y = np.concatenate([y_train[indices_to_augment], y_train[indices_to_augment]])
    combined_x_nnls = np.concatenate([x_nnls_original[indices_to_augment], x_nnls_original[indices_to_augment]])
    is_augmented = np.zeros(len(combined_y), dtype=bool)
    is_augmented[num_to_augment:] = True

    # Save to disk with a unique name based on base_dir
    dataset_name = Path(base_dir).name
    np.save(f"pre_augmented_fraction_{dataset_name}.npy", combined_fraction)
    np.save(f"pre_augmented_coverage_{dataset_name}.npy", combined_coverage)
    np.save(f"pre_augmented_y_{dataset_name}.npy", combined_y)
    np.save(f"pre_augmented_x_nnls_{dataset_name}.npy", combined_x_nnls)

    # Load pre-augmented dataset with x_nnls
    pre_augmented_dataset = PreAugmentedTissueDataset(
        combined_fraction,
        combined_coverage,
        combined_y,
        x_nnls=combined_x_nnls,
        model=model,
        is_clinical_like=False,
        is_augmented=is_augmented
    )
    print(f"Training dataset has {len(pre_augmented_dataset)} samples.")

    # Create DataLoader with shuffling
    train_dl = DataLoader(
        pre_augmented_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=24,
        pin_memory=False,
        persistent_workers=True
    )
    return train_dl

def enhanced_negative_examples(
    train_dl: DataLoader,
    cell_types: list,
    sample_fraction: float = 0.15,
    model=None,
) -> DataLoader:
    """
    Enhance the training dataset by adding negative examples for each cell type with low/very low coverage variants.
    Uses the full dataset with shuffling.
    
    Args:
        train_dl: Original training DataLoader
        cell_types: List of cell type names
        sample_fraction: Fraction of samples to use as negative examples per cell type
        model: Model instance for presence computation
        
    Returns:
        DataLoader: Enhanced DataLoader with negative examples
    """
    # Extract the original dataset
    dataset = train_dl.dataset

    # Handle ConcatDataset by iterating over underlying datasets
    if isinstance(dataset, ConcatDataset):
        fractions = []
        coverages = []
        ys = []
        x_nnls_list = []
        for sub_dataset in dataset.datasets:
            fractions.append(sub_dataset.fraction)
            coverages.append(sub_dataset.coverage)
            ys.append(sub_dataset.y)
            x_nnls_list.append(sub_dataset.x_nnls)
        fraction = np.concatenate(fractions, axis=0)
        coverage = np.concatenate(coverages, axis=0)
        y = np.concatenate(ys, axis=0)
        x_nnls = np.concatenate(x_nnls_list, axis=0)
    else:
        # Single dataset case (e.g., PreAugmentedTissueDataset)
        fraction = dataset.fraction
        coverage = dataset.coverage
        y = dataset.y
        x_nnls = dataset.x_nnls

    print("Original dataset size:", len(y), "samples")

    # Create negative examples for each cell type
    negative_fractions = []
    negative_coverages = []
    negative_ys = []
    negative_x_nnls = []
    negative_is_augmented = []
    print("Creating negative examples for each cell type...")
    for cell_idx, cell_type in enumerate(cell_types):
        print(f"Processing {cell_type} (index {cell_idx})...")
        # Find samples where this cell type is absent (proportion < 0.001)
        absent_mask = y[:, cell_idx] < 0.001
        absent_indices = np.where(absent_mask)[0]
        print(f"  Found {len(absent_indices)} samples with {cell_type} absent")

        # Select a fraction of these samples
        num_samples = int(len(absent_indices) * sample_fraction)
        selected_indices = np.random.choice(absent_indices, num_samples, replace=False)
        print(f"  Selected {num_samples} samples to use as negative examples")

        # Extract the selected samples
        sel_fraction = fraction[selected_indices]
        sel_coverage = coverage[selected_indices]
        sel_y = y[selected_indices].copy()
        sel_y[:, cell_idx] = 0.0
        sel_x_nnls = x_nnls[selected_indices]

        N_sel, M = sel_fraction.shape

        # Prepare vectorized operations
        orig_reads = np.maximum(1, np.rint(sel_coverage)).astype(np.int32)

        # Low Coverage Variant (50% reduction)
        low_cov = sel_coverage * 0.5
        target_reads_low = np.maximum(1, np.rint(low_cov)).astype(np.int32)
        orig_methylated = np.rint(sel_fraction * orig_reads).astype(np.int32)
        mask_low = (sel_coverage > 0) & (target_reads_low < orig_reads)
        low_fraction = sel_fraction.copy()
        if np.any(mask_low):
            sampled_low = np.random.hypergeometric(
                orig_methylated[mask_low],
                (orig_reads - orig_methylated)[mask_low],
                target_reads_low[mask_low]
            )
            low_fraction[mask_low] = sampled_low / target_reads_low[mask_low].astype(np.float32)

        # Very Low Coverage Variant (80% reduction)
        very_low_cov = sel_coverage * 0.2
        target_reads_very = np.rint(very_low_cov).astype(np.int32)
        very_low_fraction = np.zeros_like(sel_fraction)
        mask_very = (sel_coverage > 0) & (target_reads_very > 0)
        if np.any(mask_very):
            sampled_very = np.random.hypergeometric(
                orig_methylated[mask_very],
                (orig_reads - orig_methylated)[mask_very],
                target_reads_very[mask_very]
            )
            very_low_fraction[mask_very] = sampled_very / target_reads_very[mask_very].astype(np.float32)

        # Append three variants per sample with augmentation flags
        # Original variant (unaugmented)
        negative_fractions.append(sel_fraction)
        negative_coverages.append(sel_coverage)
        negative_ys.append(sel_y)
        negative_x_nnls.append(sel_x_nnls)
        negative_is_augmented.extend([False] * num_samples)

        # Low coverage variant (augmented)
        negative_fractions.append(low_fraction)
        negative_coverages.append(low_cov)
        negative_ys.append(sel_y)
        negative_x_nnls.append(sel_x_nnls)
        negative_is_augmented.extend([True] * num_samples)

        # Very low coverage variant (augmented)
        negative_fractions.append(very_low_fraction)
        negative_coverages.append(very_low_cov)
        negative_ys.append(sel_y)
        negative_x_nnls.append(sel_x_nnls)
        negative_is_augmented.extend([True] * num_samples)

    # Combine all negative examples
    negative_fraction = np.concatenate(negative_fractions)
    negative_coverage = np.concatenate(negative_coverages)
    negative_y = np.concatenate(negative_ys)
    negative_x_nnls = np.concatenate(negative_x_nnls)
    negative_is_augmented = np.array(negative_is_augmented, dtype=bool)

    # Save to disk
    np.save("pre_augmented_negative_fraction.npy", negative_fraction)
    np.save("pre_augmented_negative_coverage.npy", negative_coverage)
    np.save("pre_augmented_negative_y.npy", negative_y)
    np.save("pre_augmented_negative_x_nnls.npy", negative_x_nnls)

    # Create pre-augmented dataset for negative examples
    negative_dataset = PreAugmentedTissueDataset(
        negative_fraction,
        negative_coverage,
        negative_y,
        x_nnls=negative_x_nnls,
        model=model,
        is_clinical_like=False,
        is_augmented=negative_is_augmented
    )

    # Combine original dataset with negative examples
    enhanced_dataset = ConcatDataset([train_dl.dataset, negative_dataset])
    print(f"Adding {len(negative_dataset)} negative examples to the dataset")
    print(f"Enhanced dataset created: {len(train_dl.dataset)} → {len(enhanced_dataset)} samples")

    # Create new DataLoader with shuffling
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
    target_ids = atlas["target"].map(lambda x: cell_types.index(x)).to_numpy()

    model = CellTypeDeconvolutionModel(
        num_markers=len(atlas), 
        num_cell_types=len(cell_types),
        presence_models_dir=presence_models_dir,
        target_ids=target_ids,
        feature_dim=64,
        use_x_nnls=True,  # Enable NNLS priors
        initialise_weights=False
    )

    # Load training data
    train_dl_low = load_training_with_augmentation(
        f"{train_pat_dir}_low", 
        atlas, 
        names, 
        num_files=5,
        model=model,
        clinical_coverage_params=clinical_dist_params['low']
    )
    train_dl_med = load_training_with_augmentation(
        f"{train_pat_dir}_med", 
        atlas, 
        names, 
        num_files=3,
        model=model,
        clinical_coverage_params=clinical_dist_params['med']
    )
    train_dl_high = load_training_with_augmentation(
        f"{train_pat_dir}_high", 
        atlas, 
        names, 
        num_files=2,
        model=model,
        clinical_coverage_params=clinical_dist_params['high']
    )

    train_dataset = ConcatDataset([train_dl_low.dataset, train_dl_med.dataset, train_dl_high.dataset])
    train_dl = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        pin_memory=False,
        persistent_workers=True,
    )

    # Load validation data with dual sets (regular and clinical-like)
    val_loaders = {}
    clinical_val_loaders = {}
    y_vals = {}
    for cov in ['high', 'med', 'low', 'clinical']:
        # Regular (unaugmented) and clinical-like (augmented) validation sets
        tier1_dl, tier1_clinical_dl, t1_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "tier1"), 
            atlas, 
            names,
            enable_augmentation=True,
            clinical_coverage_params=clinical_dist_params[cov],
            model=model,
        )
        print(f"Validation set {cov} tier1 length={len(t1_yval)}")
        
        tcells_dl, tcells_clinical_dl, tcells_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "T-cells"), 
            atlas, 
            names,
            enable_augmentation=True,
            clinical_coverage_params=clinical_dist_params[cov],
            model=model,     
        )
        print(f"Validation set {cov} tcells length={len(tcells_yval)}")
        
        oac_dl, oac_clinical_dl, oac_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "OAC"), 
            atlas, 
            names,
            enable_augmentation=True,
            clinical_coverage_params=clinical_dist_params[cov],
            model=model,
        )
        print(f"Validation set {cov} oac length={len(oac_yval)}")

        # Store in dictionaries
        val_loaders[f"tier1_{cov}"] = tier1_dl
        val_loaders[f"t-cells_{cov}"] = tcells_dl
        val_loaders[f"oac_{cov}"] = oac_dl
        
        clinical_val_loaders[f"tier1_{cov}_clinical"] = tier1_clinical_dl
        clinical_val_loaders[f"t-cells_{cov}_clinical"] = tcells_clinical_dl
        clinical_val_loaders[f"oac_{cov}_clinical"] = oac_clinical_dl

        y_vals[f"tier1_{cov}"] = t1_yval
        y_vals[f"t-cells_{cov}"] = tcells_yval
        y_vals[f"oac_{cov}"] = oac_yval
        y_vals[f"tier1_{cov}_clinical"] = t1_yval
        y_vals[f"t-cells_{cov}_clinical"] = tcells_yval
        y_vals[f"oac_{cov}_clinical"] = oac_yval

    # Enhance training data with negative examples
    enhanced_train_dl = enhanced_negative_examples(
        train_dl,
        cell_types,
        sample_fraction=0.15,
        model=model,
    )

    # Train the model
    model, _ = train_model(
        model=model,
        train_loader=enhanced_train_dl,
        val_loaders=(val_loaders, clinical_val_loaders),
        model_path=output_path,
        cell_types=cell_types,
        num_epochs=1000,
        patience=10, 
        lr=1e-3, 
        weight_decay=1e-5,
    )

    print("\nStandard Validation Sets:")
    for tier in val_loaders.keys():
        tier_dl = val_loaders[tier]
        y_val = y_vals[tier]
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

    print("\nClinical-Like Validation Sets:")
    for tier in clinical_val_loaders.keys():
        tier_dl = clinical_val_loaders[tier]
        y_val = y_vals[tier]
        deep_conv_estimations = []
        for batch in tier_dl:
            fraction = batch['X']
            coverage = batch['coverage']
            batch_estimation = predict_with_consensus(model, fraction, coverage)
            deep_conv_estimations.append(batch_estimation)
        deep_conv_estimation = np.concatenate(deep_conv_estimations, axis=0)
        deep_conv_eval_metrics = evaluate_performance(y_val.detach().numpy(), deep_conv_estimation, cell_types)
        print(f"Clinical-like validation metrics for tier {tier}")
        log_metrics(deep_conv_eval_metrics)

    return model

def main():
    import argparse
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