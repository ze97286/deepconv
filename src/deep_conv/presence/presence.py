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
from deep_conv.presence.train import *
from deep_conv.presence.evaluate import *

from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def analyze_dataset_distribution(dataset_name, dataloader):
    """Analyze and log the distribution of samples in a dataset"""
    all_labels = []
    all_coverages = []
    all_missing_percentages = []
    for batch in dataloader:
        marker_values = batch['X'].numpy()
        coverage = batch['coverage'].numpy()
        if 'label' in batch:
            labels = batch['label'].numpy()
            all_labels.append(labels)
        elif 'concentration' in batch:
            concentrations = batch['concentration'].numpy()
            # Create binary labels based on presence threshold
            presence_threshold = 0.0005
            labels = (concentrations >= presence_threshold).astype(float)
            all_labels.append(labels)
        # Calculate mean coverage for each sample
        mean_coverage = coverage.mean(axis=1)
        all_coverages.append(mean_coverage)
        # Calculate missing marker percentage for each sample
        missing_percentages = (coverage == 0).sum(axis=1) / coverage.shape[1] * 100
        all_missing_percentages.append(missing_percentages)
    # Combine all data
    all_labels = np.concatenate(all_labels).flatten()
    all_coverages = np.concatenate(all_coverages).flatten()
    all_missing_percentages = np.concatenate(all_missing_percentages).flatten()
    # Calculate statistics
    positive_mask = all_labels > 0.5
    negative_mask = ~positive_mask
    print(f"\n=== {dataset_name} Distribution Analysis ===")
    print(f"Total samples: {len(all_labels)}")
    print(f"Positive samples: {np.sum(positive_mask)} ({np.mean(positive_mask)*100:.1f}%)")
    print(f"Negative samples: {np.sum(negative_mask)} ({np.mean(negative_mask)*100:.1f}%)")
    # Coverage distribution
    high_cov = all_coverages >= 30.0
    med_cov = (all_coverages >= 10.0) & (all_coverages < 30.0)
    low_cov = all_coverages < 10.0
    very_low_cov = all_coverages < 5.0
    print("\nCoverage distribution:")
    print(f"High coverage (≥30): {np.sum(high_cov)} ({np.mean(high_cov)*100:.1f}%)")
    print(f"Medium coverage (10-30): {np.sum(med_cov)} ({np.mean(med_cov)*100:.1f}%)")
    print(f"Low coverage (<10): {np.sum(low_cov)} ({np.mean(low_cov)*100:.1f}%)")
    print(f"Very low coverage (<5): {np.sum(very_low_cov)} ({np.mean(very_low_cov)*100:.1f}%)")
    # Class distribution by coverage
    print("\nClass distribution by coverage:")
    print(f"High coverage, positive: {np.sum(high_cov & positive_mask)} ({np.mean(positive_mask[high_cov])*100:.1f}%)")
    print(f"High coverage, negative: {np.sum(high_cov & negative_mask)} ({np.mean(negative_mask[high_cov])*100:.1f}%)")
    print(f"Medium coverage, positive: {np.sum(med_cov & positive_mask)} ({np.mean(positive_mask[med_cov])*100:.1f}%)")
    print(f"Medium coverage, negative: {np.sum(med_cov & negative_mask)} ({np.mean(negative_mask[med_cov])*100:.1f}%)")
    print(f"Low coverage, positive: {np.sum(low_cov & positive_mask)} ({np.mean(positive_mask[low_cov])*100:.1f}%)")
    print(f"Low coverage, negative: {np.sum(low_cov & negative_mask)} ({np.mean(negative_mask[low_cov])*100:.1f}%)")
    # Missing marker statistics
    print("\nMissing marker statistics:")
    print(f"Average % missing markers: {np.mean(all_missing_percentages):.1f}%")
    print(f"Samples with >10% missing: {np.sum(all_missing_percentages > 10)} ({np.mean(all_missing_percentages > 10)*100:.1f}%)")
    print(f"Samples with >30% missing: {np.sum(all_missing_percentages > 30)} ({np.mean(all_missing_percentages > 30)*100:.1f}%)")
    print(f"Samples with >50% missing: {np.sum(all_missing_percentages > 50)} ({np.mean(all_missing_percentages > 50)*100:.1f}%)")
    # Correlation between missing markers and coverage
    corr = np.corrcoef(all_coverages, all_missing_percentages)[0, 1]
    print(f"\nCorrelation between coverage and missing percentage: {corr:.3f}")
    # Missing marker distribution by class
    print("\nMissing marker percentage by class:")
    print(f"Positive samples: {np.mean(all_missing_percentages[positive_mask]):.1f}%")
    print(f"Negative samples: {np.mean(all_missing_percentages[negative_mask]):.1f}%")
    
    return {
        'class_balance': {
            'positive': np.mean(positive_mask),
            'negative': np.mean(negative_mask)
        },
        'coverage': {
            'high': np.mean(high_cov),
            'medium': np.mean(med_cov),
            'low': np.mean(low_cov),
            'very_low': np.mean(very_low_cov)
        },
        'missing_markers': {
            'average': np.mean(all_missing_percentages),
            '>10%': np.mean(all_missing_percentages > 10),
            '>30%': np.mean(all_missing_percentages > 30),
            '>50%': np.mean(all_missing_percentages > 50)
        },
        'missing_by_class': {
            'positive': np.mean(all_missing_percentages[positive_mask]),
            'negative': np.mean(all_missing_percentages[negative_mask])
        },
        'coverage_missing_correlation': corr
    }


def get_validation_set(eval_pat_dir: str, target_cell_type:int, names: set) -> Tuple[DataLoader, torch.Tensor]:
    """
    Reads marker coverage, methylation, and ground-truth label files from a validation set directory,
    filters them down to the set of markers in 'names', and returns a DataLoader plus normalised labels.

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
            A [N, C] Tensor of ground-truth cell-type proportions, normalised so each row sums to 1.
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

    markers, coverage, y_val = augment_presence_model_data(X_val, coverage_val, y_val, target_cell_type)
    
    val_dataset = BinaryCellTypeDataset(
        fraction=markers,
        coverage=coverage,
        y=y_val,
        # target_ids=list(atlas.columns[8:]),
        target_cell_type=target_cell_type,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=2,
        persistent_workers=True,
        shuffle=False
    )
    
    # Convert y_val to a PyTorch tensor and normalise each row
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_val = y_val / y_val.sum(dim=1, keepdim=True)
    
    return val_loader, y_val


def augment_presence_model_data(markers, coverage, y, cell_type_idx, presence_threshold=0.0005):
    """
    Create balanced augmentation across coverage levels and class distributions.
    """
    augmented_markers = []
    augmented_coverage = []
    augmented_y = []
    
    # Keep the original data
    augmented_markers.append(markers.copy())
    augmented_coverage.append(coverage.copy())
    augmented_y.append(y.copy())
    
    # Define coverage levels for stratification
    avg_coverage = coverage.mean(axis=1)
    high_cov_mask = avg_coverage >= 30.0
    med_cov_mask = (avg_coverage >= 10.0) & (avg_coverage < 30.0)
    low_cov_mask = avg_coverage < 10.0
    
    # 1. Augment true negative examples (where target cell type is absent)
    negative_mask = y[:, cell_type_idx] < presence_threshold
    
    # Stratify negative examples by coverage - INCREASE LOW COVERAGE COUNTS
    strata = [
        (high_cov_mask & negative_mask, "high_neg", 6000), 
        (med_cov_mask & negative_mask, "med_neg", 6000),  # Increased from 4000
        (low_cov_mask & negative_mask, "low_neg", 8000)   # Increased from 3000
    ]
    
    for combined_mask, stratum_name, max_samples in strata:
        indices = np.where(combined_mask)[0]
        
        if len(indices) > 0:
            # Sample from this stratum
            sample_size = min(len(indices), max_samples)
            sample_indices = np.random.choice(indices, sample_size, replace=True)  # Allow replacement for small datasets
            
            # Set coverage factors based on stratum - MORE LOW COVERAGE FACTORS
            if "high" in stratum_name:
                cov_factors = [1.0, 0.7, 0.4, 0.2, 0.1]
            elif "med" in stratum_name:
                cov_factors = [1.0, 0.6, 0.3, 0.15, 0.08, 0.05]  # Added 0.05
            else:  # low coverage - MORE ULTRA-LOW FACTORS
                cov_factors = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003]  # Added more factors
            
            for coverage_factor in cov_factors:
                neg_markers = markers[sample_indices].copy()
                neg_coverage = coverage[sample_indices].copy() * coverage_factor
                
                # Choose noise type randomly for more variability
                noise_type = np.random.choice(['gaussian', 'salt_pepper', 'systematic_bias'], 
                                             p=[0.7, 0.15, 0.15])
                
                if noise_type == 'gaussian':
                    # Standard Gaussian noise with coverage-dependent magnitude
                    noise_level = 0.15 / np.sqrt(coverage_factor + 0.1)
                    noise = np.random.normal(0, noise_level, size=neg_markers.shape)
                elif noise_type == 'salt_pepper':
                    # Salt and pepper noise (sparse high/low values)
                    noise = np.zeros_like(neg_markers)
                    salt_prob = 0.02 + 0.03 / (coverage_factor + 0.1)  # More salt at lower coverage
                    pepper_prob = 0.02 + 0.03 / (coverage_factor + 0.1)
                    salt = np.random.random(neg_markers.shape) < salt_prob
                    pepper = np.random.random(neg_markers.shape) < pepper_prob
                    noise[salt] = 0.3  # Salt (high values)
                    noise[pepper] = -0.3  # Pepper (low values)
                elif noise_type == 'systematic_bias':
                    # Systematic bias (shifted mean)
                    bias = 0.05 * np.random.choice([-1, 1])  # Random direction
                    noise_level = 0.08 / np.sqrt(coverage_factor + 0.1)
                    noise = np.random.normal(bias, noise_level, size=neg_markers.shape)
                
                neg_markers = np.clip(neg_markers + noise, 0, 1)
                
                # INCREASE MARKER ZEROING FOR ALL COVERAGE LEVELS
                if coverage_factor < 0.8:  # Lower threshold to apply zeroing
                    # More aggressive zeroing - higher chance at lower coverage
                    zero_prob = min(0.75, 0.3 / (coverage_factor + 0.05))  # More aggressive
                    
                    # Optional: create contiguous dropout regions to simulate technical artifacts
                    # INCREASE FREQUENCY OF DROPOUT REGIONS
                    if np.random.random() < 0.5:  # Increased from 0.3
                        for i in range(len(neg_markers)):
                            # Create 1-4 dropout regions per sample
                            for _ in range(np.random.randint(1, 5)):  # Increased max regions
                                # Dropout region length between 2-12 markers
                                length = np.random.randint(2, min(13, neg_markers.shape[1]//3))  # Increased max length
                                if length > 0 and neg_markers.shape[1] > length:
                                    start = np.random.randint(0, neg_markers.shape[1] - length)
                                    neg_coverage[i, start:start+length] = 0
                    
                    # Apply random individual marker zeroing in addition to contiguous regions
                    zero_mask = np.random.random(neg_markers.shape) < zero_prob
                    neg_coverage[zero_mask] = 0
                
                # Add to augmented datasets with the original labels
                augmented_markers.append(neg_markers)
                augmented_coverage.append(neg_coverage)
                augmented_y.append(y[sample_indices].copy())
    
    # 2. Augment positive examples (where target cell type is present)
    # Stratify positives by concentration and coverage
    # INCREASE VERY LOW CONCENTRATION COUNTS
    ranges = [
        (presence_threshold, 0.005, "very_low", 10000),  # Increased from 8000
        (0.005, 0.01, "low", 8000),                      # Increased from 7000
        (0.01, 0.05, "med_low", 6000),                   # Same
        (0.05, 0.2, "medium", 5000),                     # Same
        (0.2, 0.5, "high", 4000),                        # Same
        (0.5, 1.0, "very_high", 3000)                    # Same
    ]
    
    for min_conc, max_conc, stratum_name, max_samples in ranges:
        # Further stratify by coverage
        for cov_mask, cov_name in [
            (high_cov_mask, "high_"),
            (med_cov_mask, "med_"),
            (low_cov_mask, "low_")
        ]:
            # Create combined mask
            combined_mask = cov_mask & (y[:, cell_type_idx] >= min_conc) & (y[:, cell_type_idx] < max_conc)
            indices = np.where(combined_mask)[0]
            
            if len(indices) > 0:
                # Sample from this stratum (adjust sample size based on coverage)
                # INCREASE LOW COVERAGE SAMPLE SIZES
                if "high_" in cov_name:
                    sample_size = min(len(indices), max_samples // 4)  # Reduced from 1/3
                elif "med_" in cov_name:
                    sample_size = min(len(indices), max_samples // 3)  # Same
                else:  # low coverage - INCREASED PROPORTION
                    sample_size = min(len(indices), max_samples // 2 + max_samples // 4)  # Increased from 1/2
                
                sample_indices = np.random.choice(indices, sample_size, replace=True)  # Allow replacement
                
                # Different coverage ranges based on concentration and coverage stratum
                # ADD MORE ULTRA-LOW COVERAGE FACTORS FOR ALL STRATA
                if stratum_name in ["very_low", "low"]:
                    if "high_" in cov_name:
                        cov_factors = [1.0, 0.6, 0.3, 0.15, 0.08, 0.04]  # Added lower factors
                    elif "med_" in cov_name:
                        cov_factors = [1.0, 0.6, 0.3, 0.15, 0.08, 0.04, 0.02]  # Added lower factors
                    else:  # low coverage
                        cov_factors = [1.0, 0.6, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]  # Added lower factors
                elif stratum_name in ["med_low", "medium"]:
                    if "high_" in cov_name:
                        cov_factors = [1.0, 0.6, 0.3, 0.15, 0.08]  # Added lower factors
                    elif "med_" in cov_name:
                        cov_factors = [1.0, 0.6, 0.3, 0.15, 0.08, 0.04]  # Added lower factors
                    else:  # low coverage
                        cov_factors = [1.0, 0.6, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01]  # Added lower factors
                else:  # high and very_high
                    if "high_" in cov_name:
                        cov_factors = [1.0, 0.7, 0.4, 0.2, 0.1]  # Added lower factors
                    elif "med_" in cov_name:
                        cov_factors = [1.0, 0.7, 0.4, 0.2, 0.1, 0.05]  # Added lower factors
                    else:  # low coverage
                        cov_factors = [1.0, 0.7, 0.4, 0.2, 0.1, 0.05, 0.03, 0.01]  # Added lower factors
                
                for coverage_factor in cov_factors:
                    pos_markers = markers[sample_indices].copy()
                    pos_coverage = coverage[sample_indices].copy() * coverage_factor
                    
                    # Choose noise type randomly
                    noise_type = np.random.choice(['gaussian', 'salt_pepper', 'systematic_bias'], 
                                                 p=[0.7, 0.15, 0.15])
                    
                    if noise_type == 'gaussian':
                        # Scale noise level inversely with concentration
                        # Less noise for high concentrations where signal should be clearer
                        if stratum_name in ["high", "very_high"]:
                            base_noise = 0.12
                        else:
                            base_noise = 0.15
                            
                        noise_level = base_noise / np.sqrt(coverage_factor + 0.1)
                        noise = np.random.normal(0, noise_level, size=pos_markers.shape)
                    elif noise_type == 'salt_pepper':
                        noise = np.zeros_like(pos_markers)
                        # Scale salt/pepper probability by concentration
                        base_prob = 0.02 if stratum_name in ["high", "very_high"] else 0.03
                        sp_prob = base_prob + base_prob / (coverage_factor + 0.1)
                        
                        salt = np.random.random(pos_markers.shape) < sp_prob
                        pepper = np.random.random(pos_markers.shape) < sp_prob
                        noise[salt] = 0.3
                        noise[pepper] = -0.3
                    elif noise_type == 'systematic_bias':
                        bias = 0.05 * np.random.choice([-1, 1])
                        noise_level = 0.08 / np.sqrt(coverage_factor + 0.1)
                        noise = np.random.normal(bias, noise_level, size=pos_markers.shape)
                        
                    pos_markers = np.clip(pos_markers + noise, 0, 1)
                    
                    # INCREASED ZEROING FOR POSITIVE SAMPLES TO MATCH NEGATIVES BETTER
                    if coverage_factor < 0.4:  # Lowered threshold (was 0.3)
                        # Scale zeroing probability by concentration stratum
                        if stratum_name in ["very_low", "low"]:
                            base_zero_prob = 0.1  # Increased
                        elif stratum_name in ["med_low", "medium"]:
                            base_zero_prob = 0.08  # Increased
                        else:
                            base_zero_prob = 0.06  # Increased
                            
                        zero_prob = min(0.7, base_zero_prob / coverage_factor)  # More aggressive
                        
                        # INCREASE FREQUENCY OF CONTIGUOUS DROPOUT FOR POSITIVES
                        if np.random.random() < 0.4:  # Increased from 0.3
                            for i in range(len(pos_markers)):
                                # Create 1-3 dropout regions per sample
                                for _ in range(np.random.randint(1, 4)):  # Increased max regions
                                    length = np.random.randint(2, min(10, pos_markers.shape[1]//4))  # Increased max length
                                    if length > 0 and pos_markers.shape[1] > length:
                                        start = np.random.randint(0, pos_markers.shape[1] - length)
                                        pos_coverage[i, start:start+length] = 0
                        
                        # Apply random individual marker zeroing
                        zero_mask = np.random.random(pos_markers.shape) < zero_prob
                        pos_coverage[zero_mask] = 0
                    
                    # Add to augmented datasets with the original labels
                    augmented_markers.append(pos_markers)
                    augmented_coverage.append(pos_coverage)
                    augmented_y.append(y[sample_indices].copy())
    
    # 3. Create synthetic high-concentration examples if needed
    # Check if we have too few high-concentration examples
    very_high_pos_mask = y[:, cell_type_idx] >= 0.5
    if np.sum(very_high_pos_mask) < 1000:
        # Look for medium-high concentration examples to enrich
        med_high_pos_mask = (y[:, cell_type_idx] >= 0.2) & (y[:, cell_type_idx] < 0.5)
        med_high_indices = np.where(med_high_pos_mask)[0]
        
        if len(med_high_indices) > 0:
            sample_size = min(len(med_high_indices), 3000)
            sample_indices = np.random.choice(med_high_indices, sample_size, replace=False)
            
            # Get the original concentrations
            orig_conc = y[sample_indices, cell_type_idx].copy()
            
            # Create enriched copies (scale up the concentration)
            for enrichment_factor in [1.5, 2.0, 2.5]:
                # Copy the samples
                enriched_markers = markers[sample_indices].copy()
                enriched_coverage = coverage[sample_indices].copy()
                enriched_y = y[sample_indices].copy()
                
                # Scale up the target cell type concentration
                enriched_y[:, cell_type_idx] = np.clip(orig_conc * enrichment_factor, 0, 0.9)
                
                # Rescale other concentrations to maintain sum <= 1
                row_sums = enriched_y.sum(axis=1)
                for i in range(len(enriched_y)):
                    if row_sums[i] > 1.0:
                        # Rescale all non-target concentrations
                        scale_factor = (1.0 - enriched_y[i, cell_type_idx]) / (row_sums[i] - enriched_y[i, cell_type_idx])
                        for j in range(enriched_y.shape[1]):
                            if j != cell_type_idx:
                                enriched_y[i, j] *= scale_factor
                
                # Add to augmented datasets
                augmented_markers.append(enriched_markers)
                augmented_coverage.append(enriched_coverage)
                augmented_y.append(enriched_y)
    
    # 4. Add extremely challenging cases for robustness
    # Focus on medium and high concentrations for robustness training
    focus_pos_mask = (y[:, cell_type_idx] >= 0.05)
    focus_pos_indices = np.where(focus_pos_mask)[0]
    
    if len(focus_pos_indices) > 0:
        # Sample a larger subset for extreme augmentation - INCREASED COUNT
        sample_size = min(len(focus_pos_indices), 4000)  # Increased from 2000
        sample_indices = np.random.choice(focus_pos_indices, sample_size, replace=True)
        
        # Create extreme variants - ADDED MORE EXTREME FACTORS
        for coverage_factor in [0.05, 0.03, 0.015, 0.008, 0.004, 0.002]:  # Added 0.05, 0.004, 0.002
            extreme_markers = markers[sample_indices].copy()
            extreme_coverage = coverage[sample_indices].copy() * coverage_factor
            
            # Aggressive noise
            noise_level = 0.2
            noise = np.random.normal(0, noise_level, size=extreme_markers.shape)
            extreme_markers = np.clip(extreme_markers + noise, 0, 1)
            
            # Very high marker dropout (60-90%) - INCREASED UPPER BOUND
            zero_prob = np.random.uniform(0.6, 0.9)  # Increased upper bound from 0.8
            zero_mask = np.random.random(extreme_markers.shape) < zero_prob
            extreme_coverage[zero_mask] = 0
            
            # Add these challenging examples
            augmented_markers.append(extreme_markers)
            augmented_coverage.append(extreme_coverage)
            augmented_y.append(y[sample_indices].copy())
    
    # 5. ADD SPECIFIC MISSING MARKER PERCENTAGE EXAMPLES
    # Create examples with specific percentages of missing markers
    for missing_percentage in [30, 50, 70]:  # Added 70% missing
        # For both positive and negative examples
        for is_positive, base_mask in [(True, y[:, cell_type_idx] >= presence_threshold), 
                                       (False, y[:, cell_type_idx] < presence_threshold)]:
            # Sample from different coverage levels
            for cov_mask, cov_name in [
                (high_cov_mask, "high_"),
                (med_cov_mask, "med_"),
                (low_cov_mask, "low_")
            ]:
                combined_mask = cov_mask & base_mask
                indices = np.where(combined_mask)[0]
                
                if len(indices) > 0:
                    sample_size = min(len(indices), 2000)  # Create 2000 samples per category
                    sample_indices = np.random.choice(indices, sample_size, replace=True)
                    
                    missing_markers = markers[sample_indices].copy()
                    missing_coverage = coverage[sample_indices].copy()
                    
                    # Create missing pattern
                    for i in range(len(missing_markers)):
                        # Determine how many markers to zero out
                        zero_count = int(missing_markers.shape[1] * missing_percentage / 100)
                        
                        # Decide between random zeroing and contiguous zeroing
                        if np.random.random() < 0.5:
                            # Random zeroing
                            zero_indices = np.random.choice(missing_markers.shape[1], zero_count, replace=False)
                            missing_coverage[i, zero_indices] = 0
                        else:
                            # Contiguous zeroing (in chunks)
                            remaining = zero_count
                            while remaining > 0:
                                # Choose chunk size (between 1 and remaining, up to 1/4 of markers)
                                chunk_size = min(remaining, np.random.randint(1, max(2, min(remaining, missing_markers.shape[1]//4))))
                                
                                # Choose start position
                                if missing_markers.shape[1] > chunk_size:
                                    start = np.random.randint(0, missing_markers.shape[1] - chunk_size)
                                    missing_coverage[i, start:start+chunk_size] = 0
                                    remaining -= chunk_size
                                else:
                                    break
                    
                    # Add to augmented datasets
                    augmented_markers.append(missing_markers)
                    augmented_coverage.append(missing_coverage)
                    augmented_y.append(y[sample_indices].copy())
    
    # 6. ADD MORE REALISTIC LOW COVERAGE SIMULATION
    # For some samples, use binomial sampling to more realistically model low coverage
    for cov_factor in [0.2, 0.1, 0.05, 0.02]:
        # Choose samples to apply this to
        sample_count = 5000  # Number of samples to process with this method
        
        # Select samples from both positive and negative classes
        pos_indices = np.where(y[:, cell_type_idx] >= presence_threshold)[0]
        neg_indices = np.where(y[:, cell_type_idx] < presence_threshold)[0]
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            # Balance between positive and negative
            pos_sample_size = min(len(pos_indices), sample_count // 2)
            neg_sample_size = min(len(neg_indices), sample_count // 2)
            
            pos_sample_indices = np.random.choice(pos_indices, pos_sample_size, replace=True)
            neg_sample_indices = np.random.choice(neg_indices, neg_sample_size, replace=True)
            
            for sample_indices in [pos_sample_indices, neg_sample_indices]:
                bin_markers = markers[sample_indices].copy()
                bin_coverage = coverage[sample_indices].copy()
                bin_y = y[sample_indices].copy()
                
                # Apply binomial sampling to each value
                for i in range(len(bin_markers)):
                    for j in range(bin_markers.shape[1]):
                        if bin_coverage[i, j] > 0:
                            # Calculate target coverage (minimum 1)
                            target_cov = max(1, int(bin_coverage[i, j] * cov_factor))
                            
                            # Generate a new marker value by binomial sampling
                            if bin_markers[i, j] > 0 and target_cov > 0:
                                # Original count of positive reads
                                orig_count = int(bin_markers[i, j] * bin_coverage[i, j])
                                
                                # Only do binomial sampling if we have at least 1 positive read
                                if orig_count > 0:
                                    # Probability of each read being positive
                                    prob = orig_count / bin_coverage[i, j]
                                    
                                    # Resample with new (lower) coverage
                                    new_pos_count = np.random.binomial(target_cov, prob)
                                    
                                    # Update marker value
                                    bin_markers[i, j] = new_pos_count / target_cov
                            
                            # Update coverage
                            bin_coverage[i, j] = target_cov
                
                # Add to augmented datasets
                augmented_markers.append(bin_markers)
                augmented_coverage.append(bin_coverage)
                augmented_y.append(bin_y)
    
    # 7. BALANCE CLASSES AT LOW COVERAGE
    # Analyze the class balance we have so far
    final_markers = np.vstack(augmented_markers)
    final_coverage = np.vstack(augmented_coverage)
    final_y = np.vstack(augmented_y)
    
    # Calculate mean coverage
    mean_covs = final_coverage.mean(axis=1)
    
    # Define coverage levels
    very_low_cov = mean_covs < 5.0
    low_cov = (mean_covs >= 5.0) & (mean_covs < 10.0)
    med_cov = (mean_covs >= 10.0) & (mean_covs < 30.0)
    
    # Define presence/absence
    presence = final_y[:, cell_type_idx] >= presence_threshold
    
    # Check class balance at each coverage level
    very_low_pos_count = np.sum(very_low_cov & presence)
    very_low_neg_count = np.sum(very_low_cov & ~presence)
    low_pos_count = np.sum(low_cov & presence)
    low_neg_count = np.sum(low_cov & ~presence)
    med_pos_count = np.sum(med_cov & presence)
    med_neg_count = np.sum(med_cov & ~presence)
    
    # If we have severe imbalance in low coverage, add more of the minority class
    if very_low_pos_count > 2 * very_low_neg_count:  # If positive class is over-represented
        # Add more negative samples at very low coverage
        existing_neg = np.where(~presence)[0]
        if len(existing_neg) > 0:
            # Sample from existing negatives
            sample_count = min(very_low_pos_count - very_low_neg_count, len(existing_neg))
            sample_indices = np.random.choice(existing_neg, sample_count, replace=True)
            
            # Create very low coverage versions
            for _ in range(2):  # Add multiple copies to balance
                new_markers = final_markers[sample_indices].copy()
                new_coverage = final_coverage[sample_indices].copy() * 0.05  # Very low coverage factor
                new_y = final_y[sample_indices].copy()
                
                # Apply heavy zeroing
                zero_prob = 0.7  # Zero out 70% of markers
                for i in range(len(new_markers)):
                    # Random zeroing
                    zero_mask = np.random.random(new_markers.shape[1]) < zero_prob
                    new_coverage[i, zero_mask] = 0
                
                augmented_markers.append(new_markers)
                augmented_coverage.append(new_coverage)
                augmented_y.append(new_y)
    
    elif very_low_neg_count > 2 * very_low_pos_count:  # If negative class is over-represented
        # Add more positive samples at very low coverage
        existing_pos = np.where(presence)[0]
        if len(existing_pos) > 0:
            # Sample from existing positives
            sample_count = min(very_low_neg_count - very_low_pos_count, len(existing_pos))
            sample_indices = np.random.choice(existing_pos, sample_count, replace=True)
            
            # Create very low coverage versions
            for _ in range(2):  # Add multiple copies to balance
                new_markers = final_markers[sample_indices].copy()
                new_coverage = final_coverage[sample_indices].copy() * 0.05  # Very low coverage factor
                new_y = final_y[sample_indices].copy()
                
                # Apply heavy zeroing
                zero_prob = 0.7  # Zero out 70% of markers
                for i in range(len(new_markers)):
                    # Random zeroing
                    zero_mask = np.random.random(new_markers.shape[1]) < zero_prob
                    new_coverage[i, zero_mask] = 0
                
                augmented_markers.append(new_markers)
                augmented_coverage.append(new_coverage)
                augmented_y.append(new_y)
    
    # Similar balancing for low coverage (not very low)
    if low_pos_count > 2 * low_neg_count:  # If positive class is over-represented
        # Add more negative samples at low coverage
        existing_neg = np.where(~presence)[0]
        if len(existing_neg) > 0:
            # Sample from existing negatives
            sample_count = min(low_pos_count - low_neg_count, len(existing_neg))
            sample_indices = np.random.choice(existing_neg, sample_count, replace=True)
            
            # Create low coverage versions
            new_markers = final_markers[sample_indices].copy()
            new_coverage = final_coverage[sample_indices].copy() * 0.1  # Low coverage factor
            new_y = final_y[sample_indices].copy()
            
            # Apply moderate zeroing
            zero_prob = 0.5  # Zero out 50% of markers
            for i in range(len(new_markers)):
                # Random zeroing
                zero_mask = np.random.random(new_markers.shape[1]) < zero_prob
                new_coverage[i, zero_mask] = 0
            
            augmented_markers.append(new_markers)
            augmented_coverage.append(new_coverage)
            augmented_y.append(new_y)
    
    elif low_neg_count > 2 * low_pos_count:  # If negative class is over-represented
        # Add more positive samples at low coverage
        existing_pos = np.where(presence)[0]
        if len(existing_pos) > 0:
            # Sample from existing positives
            sample_count = min(low_neg_count - low_pos_count, len(existing_pos))
            sample_indices = np.random.choice(existing_pos, sample_count, replace=True)
            
            # Create low coverage versions
            new_markers = final_markers[sample_indices].copy()
            new_coverage = final_coverage[sample_indices].copy() * 0.1  # Low coverage factor
            new_y = final_y[sample_indices].copy()
            
            # Apply moderate zeroing
            zero_prob = 0.5  # Zero out 50% of markers
            for i in range(len(new_markers)):
                # Random zeroing
                zero_mask = np.random.random(new_markers.shape[1]) < zero_prob
                new_coverage[i, zero_mask] = 0
            
            augmented_markers.append(new_markers)
            augmented_coverage.append(new_coverage)
            augmented_y.append(new_y)
    
    # Combine all augmented data
    final_markers = np.vstack(augmented_markers)
    final_coverage = np.vstack(augmented_coverage)
    final_y = np.vstack(augmented_y)
    
    return final_markers, final_coverage, final_y


def load_training(base_dir: str, names: set, target_cell_type: int, num_files: int = 5) -> DataLoader:
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
    
    # augment with positive and negative examples
    markers, coverage, y_train = augment_presence_model_data(X_train, coverage_train, y_train, target_cell_type)

    # Build a BinaryCellTypeDataset
    train_dataset = BinaryCellTypeDataset(
        fraction=markers,
        coverage=coverage,
        y=y_train,
        target_cell_type=target_cell_type,
    )
    
    # Also create a normalised version of y for potential usage
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


def check_prediction_distributions(model, dataloader, device=None):
    """Analyse the raw prediction probabilities for positive and negative samples."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    pos_probs = []
    neg_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, _ = model(marker_values, coverage)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # Store probabilities by true label
            for i, (prob, label) in enumerate(zip(probs, labels)):
                if label > 0.5:  # True positive
                    pos_probs.append(prob)
                else:  # True negative
                    neg_probs.append(prob)
        
    # Print statistics
    print(f"Positive samples: {len(pos_probs)}")
    print(f"  - Mean probability: {np.mean(pos_probs):.4f}")
    print(f"  - Median probability: {np.median(pos_probs):.4f}")
    print(f"  - Min: {np.min(pos_probs):.4f}, Max: {np.max(pos_probs):.4f}")
    
    print(f"Negative samples: {len(neg_probs)}")
    print(f"  - Mean probability: {np.mean(neg_probs):.4f}")
    print(f"  - Median probability: {np.median(neg_probs):.4f}")
    print(f"  - Min: {np.min(neg_probs):.4f}, Max: {np.max(neg_probs):.4f}")
    
    # Check threshold effect
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        tp = sum(p >= threshold for p in pos_probs)
        fn = sum(p < threshold for p in pos_probs)
        fp = sum(p >= threshold for p in neg_probs)
        tn = sum(p < threshold for p in neg_probs)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Threshold {threshold:.1f}: Recall={recall:.4f}, Precision={precision:.4f}, "
              f"Specificity={specificity:.4f}, F1={f1:.4f}")
    
    return pos_probs, neg_probs


def train_and_eval(
    atlas_path: str,
    train_pat_dir: str,
    eval_pat_dir: str,
    threads: int,
    output_path: str,  
    target_cell_type_name: str,
    use_loyfer: bool, 
) -> nn.Module:
    # Fix random seeds and threads for reproducibility
    set_seed()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    # 1) Read the atlas of markers and cell types
    atlas = pd.read_csv(atlas_path, sep="\t")
    cell_types = list(atlas.columns[8:])
    print("training presence model for",target_cell_type_name)
    # The 'names' set ensures we only keep relevant markers
    names = set(atlas[atlas.target==target_cell_type_name].name.unique())
    print("using", len(names), "markers for detection of presence of",target_cell_type_name)
    target_cell_type=cell_types.index(target_cell_type_name)
    # 2) Build the training DataLoader from parquet files in train_pat_dir
    train_dl = load_training(train_pat_dir, names, target_cell_type=target_cell_type)
    # 3) Build DataLoaders for each validation subset
    analyze_dataset_distribution("training", train_dl)

    if use_loyfer:
        validation_dls = {}
        y_vals = {}
        for cov in ['high','med','low']:
            tier1_dl, t1_yval = get_validation_set(str(Path(eval_pat_dir+"_"+cov) / "tier1"), target_cell_type, names)
            analyze_dataset_distribution(cov+"_validation", tier1_dl)
            # tcells_dl, tcells_yval = get_validation_set(str(Path(eval_pat_dir+"_"+cov) / "T-cells"), target_cell_type, names)
            # oac_dl, oac_yval = get_validation_set(str(Path(eval_pat_dir+"_"+cov) / "OAC"), target_cell_type, names)
            validation_dls[f"tier1_{cov}"] = tier1_dl
            # validation_dls[f"t-cells_{cov}"] = tcells_dl
            # validation_dls[f"oac_{cov}"] = oac_dl

            y_vals[f"tier1_{cov}"] = t1_yval
            # y_vals[f"t-cells_{cov}"] = tcells_yval
            # y_vals[f"oac_{cov}"] = oac_yval        
    else:
        tier1_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "tier1"), target_cell_type, names)
        tier2_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "OAC"), target_cell_type, names)
        tier3_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "CD4"), target_cell_type, names)
        tier4_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "CD8"), target_cell_type, names)
        validation_dls = {
            "tier1": tier1_dl,        
            "tier2": tier2_dl,        
            "tier3": tier3_dl,        
            "tier4": tier4_dl,        
        }

    single_model = SingleCellTypePresenceModel()

    # Train it
    trained_model = train_binary_classifier(
        model=single_model,
        dataloaders={"train":train_dl, "val": validation_dls},
        model_path=output_path,
        num_epochs=100,
        learning_rate=1e-3,
        target_cell_type_index=target_cell_type,
    )

    val_dl = validation_dls[f"tier1_low"]
    
    results_df = analyse_detection_by_concentration(trained_model, val_dl, output_path, target_cell_type_name)
    find_minimum_detection_concentration_continuous(results_df, output_path, target_cell_type_name)


def analyse_detection_by_concentration(model, dataloader, 
                                       output_path,
                                       target_cell_type,
                                      concentration_groups=None,
                                      threshold=0.5,
                                      device=None):
    """
    Analyse the model's detection performance across different concentration levels using Plotly visualisations.
    
    Args:
        model: Binary classifier model
        dataloader: DataLoader containing samples with concentration information
        concentration_groups: Dictionary mapping group names to concentration ranges
        threshold: Decision threshold for binary classification
        device: Device to run on
        
    Returns:
        results_df: DataFrame with detection results by sample
        group_stats: DataFrame with detection statistics by concentration group
        threshold_mapping: Dictionary mapping concentration ranges to optimal thresholds
    """
    if device is None:
        device = next(model.parameters()).device
    
    if concentration_groups is None:
        # Default concentration groups if not provided
        concentration_groups = {
            'super_high': (0.25, 1.0),   # 24% to 100%
            'very_high': (0.10, 0.25),   # 10% to 25%
            'high': (0.05, 0.10),        # 5% to 20%
            'medium': (0.01, 0.05),      # 1% to 5%
            'low': (0.001, 0.01),        # 0.1% to 1%
            'very_low': (0.0005, 0.001), # 0.05% to 0.1%
            'ultra_low': (0.0001, 0.0005)# 0.001% to 0.05%
        }
    
    model.eval()
    
    # Store results for each sample
    results = []
    
    # Evaluate
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing concentration detection"):
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            labels = batch['label'].to(device).view(-1, 1)
            
            # Get concentrations
            if 'concentration' in batch:
                concentrations = batch['concentration'].cpu().numpy()
            else:
                # If no concentration provided, use label as binary indicator
                concentrations = labels.cpu().numpy()
            
            # Forward pass
            logits, attention_weights = model(marker_values, coverage)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions = (probabilities >= threshold).astype(int)
            
            # Store results for each sample
            for i in range(len(concentrations)):
                results.append({
                    'concentration': concentrations[i],
                    'probability': probabilities[i][0],
                    'prediction': predictions[i][0],
                    'ground_truth': labels[i].item()
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Add concentration group column
    def get_concentration_group(conc):
        for group, (min_conc, max_conc) in concentration_groups.items():
            if min_conc <= conc < max_conc:
                return group
        return 'other'
    
    results_df['concentration_group'] = results_df['concentration'].apply(get_concentration_group)
    for group in concentration_groups.keys():
        group_data = results_df[results_df['concentration_group'] == group]
        n_samples = len(group_data)
        n_unique_gt = len(group_data['ground_truth'].unique())
        n_unique_pred = len(group_data['probability'].unique())
        
        print(f"Group {group}: {n_samples} samples, {n_unique_gt} unique ground truth values, {n_unique_pred} unique predictions")
        if len(group_data) > 10 and len(group_data['ground_truth'].unique()) > 1:
            try:
                fpr, tpr, _ = roc_curve(group_data['ground_truth'], group_data['probability'])
                print(f"Group {group}: ROC curve calculated with {len(fpr)} points")
                print(f"First few points: {list(zip(fpr[:5], tpr[:5]))}")
            except Exception as e:
                print(f"Error calculating ROC for {group}: {e}")
                
        

    # Calculate detection statistics by concentration group
    group_stats = results_df.groupby('concentration_group').agg({
        'prediction': 'mean',  # Detection rate
        'probability': ['mean', 'std', 'count'],
        'concentration': ['mean', 'min', 'max'],
        'ground_truth': 'mean'  # Actual rate of positives
    }).reset_index()
    
    # Flatten column names
    group_stats.columns = ['_'.join(col).strip('_') for col in group_stats.columns.values]
    
    # Rename columns
    group_stats = group_stats.rename(columns={
        'prediction_mean': 'detection_rate',
        'probability_mean': 'mean_probability',
        'probability_std': 'std_probability',
        'probability_count': 'sample_count',
        'concentration_mean': 'mean_concentration',
        'concentration_min': 'min_concentration',
        'concentration_max': 'max_concentration',
        'ground_truth_mean': 'true_positive_rate'
    })
    
    # Sort by mean concentration (descending)
    group_stats = group_stats.sort_values('mean_concentration', ascending=False)
    
    print("Detection rates by concentration group:")
    print(group_stats[['concentration_group', 'detection_rate', 'true_positive_rate', 
                       'mean_probability', 'sample_count', 'mean_concentration']])
    
    # Create Plotly subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Detection Rate by Concentration Group', 
            'Probability Distribution by Concentration Group',
            'Predicted Probability vs Concentration (Log Scale)', 
            'ROC Curves by Concentration Group'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Detection rate by concentration group
    fig.add_trace(
        go.Bar(
            x=group_stats['concentration_group'],
            y=group_stats['detection_rate'],
            name='Detection Rate',
            marker_color='skyblue'
        ),
        row=1, col=1
    )
    
    # 2. Distribution of probabilities by concentration group
    for group in group_stats['concentration_group']:
        group_data = results_df[results_df['concentration_group'] == group]
        
        fig.add_trace(
            go.Box(
                x=group_data['concentration_group'],
                y=group_data['probability'],
                name=group
            ),
            row=1, col=2
        )
    
    # Add threshold line to box plot
    fig.add_trace(
        go.Scatter(
            x=group_stats['concentration_group'],
            y=[threshold] * len(group_stats),
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Threshold ({threshold:.2f})'
        ),
        row=1, col=2
    )
    
    # 3. Scatter plot of probabilities vs concentration (log scale)
    fig.add_trace(
        go.Scatter(
            x=results_df['concentration'],
            y=results_df['probability'],
            mode='markers',
            marker=dict(
                color='blue',
                opacity=0.5,
                size=8
            ),
            name='Predictions'
        ),
        row=2, col=1
    )
    
    # Add threshold line to scatter plot
    fig.add_trace(
        go.Scatter(
            x=[results_df['concentration'].min(), results_df['concentration'].max()],
            y=[threshold, threshold],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Threshold ({threshold:.2f})'
        ),
        row=2, col=1
    )
    
    # 4. ROC curve for different concentration groups
    for group in concentration_groups.keys():
        group_data = results_df[results_df['concentration_group'] == group]
        
        # Only calculate ROC if there are enough samples with both classes
        if len(group_data) > 10 and len(group_data['ground_truth'].unique()) > 1:
            fpr, tpr, _ = roc_curve(group_data['ground_truth'], group_data['probability'])
            
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=group
                ),
                row=2, col=2
            )
    
    # Add diagonal line to ROC plot
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Random'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text='Cell Type Detection Performance by Concentration',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update x-axis for log scale on scatter plot
    fig.update_xaxes(type="log", row=2, col=1, title_text='Concentration (log scale)')
    fig.update_xaxes(title_text='Concentration Group', row=1, col=1)
    fig.update_xaxes(title_text='Concentration Group', row=1, col=2)
    fig.update_xaxes(title_text='False Positive Rate', row=2, col=2)
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Detection Rate', row=1, col=1)
    fig.update_yaxes(title_text='Predicted Probability', row=1, col=2)
    fig.update_yaxes(title_text='Predicted Probability', row=2, col=1)
    fig.update_yaxes(title_text='True Positive Rate', row=2, col=2)
    
    # Show the plot
    fig.write_html(output_path/f"{target_cell_type}_model_analysis.html")
    
    # Determine optimal thresholds for each concentration group
    opt_thresholds = {}
    
    for group in concentration_groups.keys():
        group_data = results_df[results_df['concentration_group'] == group]
        
        # Only calculate if there are enough samples with both classes
        if len(group_data) > 10 and len(group_data['ground_truth'].unique()) > 1:
            precision, recall, thresholds = precision_recall_curve(
                group_data['ground_truth'], group_data['probability'])
            
            # Calculate F1 score for each threshold
            f1_scores = []
            for p, r in zip(precision[:-1], recall[:-1]):  # precision_recall_curve returns one more precision than threshold
                if p + r > 0:
                    f1 = 2 * p * r / (p + r)
                else:
                    f1 = 0
                f1_scores.append(f1)
            
            # Find threshold with best F1 score
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            opt_thresholds[group] = {
                'threshold': best_threshold,
                'f1_score': best_f1,
                'precision': precision[best_idx],
                'recall': recall[best_idx]
            }
    
    # Print optimal thresholds
    print("\nOptimal thresholds by concentration group:")
    for group, stats in opt_thresholds.items():
        print(f"{group}: threshold={stats['threshold']:.4f}, F1={stats['f1_score']:.4f}, "
              f"precision={stats['precision']:.4f}, recall={stats['recall']:.4f}")
    
    # Create a mapping of concentration ranges to optimal thresholds
    threshold_mapping = {}
    for group, (min_conc, max_conc) in concentration_groups.items():
        if group in opt_thresholds:
            threshold_mapping[(min_conc, max_conc)] = opt_thresholds[group]['threshold']
    
    group_stats.to_csv(output_path/f"{target_cell_type}_group_stats.csv")
    print("threshold_mapping:",threshold_mapping)
    return results_df


def find_minimum_detection_concentration_continuous(
    results_df, output_path, target_cell_type, 
    detection_rate_threshold=0.95,
    confidence_level=0.95,  # Typically 0.95 for 95% confidence
    min_sample_size=10      # Minimum samples required for reliable estimation
):
    """
    Find minimum detection concentration treating concentration as a continuous variable.
    
    This approach directly calculates detection rates at each unique concentration
    value without binning, then finds the exact threshold where detection rate crosses
    the specified threshold.
    
    Args:
        results_df: DataFrame with columns for concentration, ground_truth, and prediction
        output_path: Path to save visualizations
        target_cell_type: Name of the cell type being analyzed
        detection_rate_threshold: Minimum acceptable detection rate (e.g., 0.95 for 95%)
        confidence_level: Level for confidence interval calculations (e.g., 0.95 for 95% CI)
        min_sample_size: Minimum number of samples required for reliable estimation
        
    Returns:
        min_reliable_conc: Minimum concentration with detection rate above threshold
        results_table: DataFrame with detection rates and confidence intervals by concentration
    """
    from scipy import stats
    
    # Identify column names
    concentration_col = 'concentration' if 'concentration' in results_df.columns else 'true_concentration'
    ground_truth_col = 'ground_truth' if 'ground_truth' in results_df.columns else 'label'
    prediction_col = 'prediction' if 'prediction' in results_df.columns else 'predicted'
    
    print(f"Using columns: concentration={concentration_col}, ground_truth={ground_truth_col}, prediction={prediction_col}")
    
    # Filter to positive samples only (since we care about detection rate of true positives)
    positive_samples = results_df[results_df[ground_truth_col] == 1].copy()
    print(f"Analyzing {len(positive_samples)} positive samples")
    
    # Sort by concentration for cumulative analysis
    positive_samples = positive_samples.sort_values(by=concentration_col)
    
    # Initialize arrays for tracking
    concentrations = []
    detection_rates = []
    sample_counts = []
    
    # This approach calculates the detection rate for all samples at or above each concentration point
    prev_conc = None
    for conc in sorted(positive_samples[concentration_col].unique()):
        # Skip duplicate concentrations
        if conc == prev_conc:
            continue
        prev_conc = conc
        
        # Get all samples at or above this concentration
        samples_at_or_above = positive_samples[positive_samples[concentration_col] >= conc]
        
        # Calculate detection rate (true positive rate)
        if len(samples_at_or_above) > 0:
            true_positives = samples_at_or_above[samples_at_or_above[prediction_col] == 1].shape[0]
            detection_rate = true_positives / len(samples_at_or_above)
            
            # Store results
            concentrations.append(conc)
            detection_rates.append(detection_rate)
            sample_counts.append(len(samples_at_or_above))
    
    # Create a DataFrame with results
    results_table = pd.DataFrame({
        'concentration': concentrations,
        'detection_rate': detection_rates,
        'sample_count': sample_counts
    })
    
    # Add columns for confidence intervals using Wilson score interval
    results_table['reliable_estimate'] = False
    results_table['lower_ci'] = None
    results_table['upper_ci'] = None
    
    for idx, row in results_table.iterrows():
        n = row['sample_count']
        p = row['detection_rate']
        
        if n < min_sample_size:
            # Mark as unreliable if sample size is too small
            continue
            
        # Calculate Wilson score interval
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
        
        results_table.at[idx, 'reliable_estimate'] = True
        results_table.at[idx, 'lower_ci'] = max(0, center - half_width)
        results_table.at[idx, 'upper_ci'] = min(1, center + half_width)
    
    # Find minimum concentration with detection rate at or above threshold
    reliable_results = results_table[results_table['reliable_estimate']]
    
    # First check with both detection rate and lower CI above threshold (conservative)
    conservative_above_threshold = reliable_results[
        (reliable_results['detection_rate'] >= detection_rate_threshold) & 
        (reliable_results['lower_ci'] >= detection_rate_threshold)
    ]
    
    if len(conservative_above_threshold) > 0:
        min_reliable_conc = conservative_above_threshold['concentration'].min()
        detection_at_min = conservative_above_threshold.loc[
            conservative_above_threshold['concentration'].idxmin(), 'detection_rate'
        ]
        lower_ci_at_min = conservative_above_threshold.loc[
            conservative_above_threshold['concentration'].idxmin(), 'lower_ci'
        ]
        print(f"Minimum concentration with {detection_rate_threshold*100:.1f}% detection rate (with 95% confidence): "
              f"{min_reliable_conc:.8f} (rate: {detection_at_min:.4f}, lower CI: {lower_ci_at_min:.4f})")
        is_conservative = True
    else:
        # Fall back to just detection rate above threshold if no points satisfy the conservative criteria
        above_threshold = results_table[results_table['detection_rate'] >= detection_rate_threshold]
        if len(above_threshold) > 0:
            min_reliable_conc = above_threshold['concentration'].min()
            detection_at_min = above_threshold.loc[above_threshold['concentration'].idxmin(), 'detection_rate']
            print(f"Minimum concentration with {detection_rate_threshold*100:.1f}% detection rate (lower confidence bound below threshold): "
                  f"{min_reliable_conc:.8f} (actual rate: {detection_at_min:.4f})")
            is_conservative = False
        else:
            min_reliable_conc = None
            print(f"No concentration achieved {detection_rate_threshold*100:.1f}% detection rate")
            is_conservative = False
    
    # Create a plot showing the continuous nature of the data
    fig = go.Figure()
    
    # Add detection rate curve
    fig.add_trace(
        go.Scatter(
            x=results_table['concentration'],
            y=results_table['detection_rate'],
            mode='lines',
            line=dict(color='blue', width=3),
            name='Detection Rate',
            hovertemplate='Concentration: %{x:.8f}<br>Detection Rate: %{y:.4f}<br>Samples: %{text}',
            text=results_table['sample_count']
        )
    )
    
    # Add confidence interval curves for reliable estimates
    reliable_for_plot = reliable_results.sort_values('concentration')
    
    if len(reliable_for_plot) > 0:
        fig.add_trace(
            go.Scatter(
                x=reliable_for_plot['concentration'],
                y=reliable_for_plot['lower_ci'],
                mode='lines',
                line=dict(color='blue', width=1, dash='dot'),
                name=f'{int(confidence_level*100)}% CI Lower',
                showlegend=True
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=reliable_for_plot['concentration'],
                y=reliable_for_plot['upper_ci'],
                mode='lines',
                line=dict(color='blue', width=1, dash='dot'),
                name=f'{int(confidence_level*100)}% CI Upper',
                showlegend=True,
                fill='tonexty',  # Fill area between this trace and the previous one
                fillcolor='rgba(0, 0, 255, 0.1)'  # Light blue fill
            )
        )
    
    # Add sample points for reference
    fig.add_trace(
        go.Scatter(
            x=results_table['concentration'],
            y=results_table['detection_rate'],
            mode='markers',
            marker=dict(
                color=results_table['reliable_estimate'].map({True: 'blue', False: 'gray'}),
                size=8,
                opacity=0.5,
                symbol=results_table['reliable_estimate'].map({True: 'circle', False: 'x'})
            ),
            name='Data Points',
            text=results_table['sample_count'].apply(lambda x: f"n={x}"),
            hovertemplate='Concentration: %{x:.8f}<br>Detection Rate: %{y:.4f}<br>%{text}',
            showlegend=False
        )
    )
    
    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[min(results_table['concentration']), max(results_table['concentration'])],
            y=[detection_rate_threshold, detection_rate_threshold],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Target Rate ({detection_rate_threshold:.2f})'
        )
    )
    
    # Add marker for minimum reliable concentration if found
    if min_reliable_conc is not None:
        marker_color = 'green' if is_conservative else 'orange'
        marker_name = 'Min Reliable Conc (with CI)' if is_conservative else 'Min Reliable Conc'
        
        fig.add_trace(
            go.Scatter(
                x=[min_reliable_conc],
                y=[detection_rate_threshold],
                mode='markers',
                marker=dict(color=marker_color, size=15, symbol='star'),
                name=f'{marker_name}: {min_reliable_conc:.8f}'
            )
        )
        
        # Add annotation for the minimum concentration
        fig.add_annotation(
            x=min_reliable_conc,
            y=detection_rate_threshold + 0.05,
            text=f"Min Reliable Concentration: {min_reliable_conc:.8f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    # Update layout
    fig.update_layout(
        title=f'Detection Rate by Concentration for {target_cell_type}',
        xaxis=dict(
            title='Concentration (log scale)',
            type='log'
        ),
        yaxis=dict(
            title='Detection Rate',
            range=[0, 1.05]
        ),
        height=600,
        width=900,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    # Add sample count as a separate trace with secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=results_table['concentration'],
            y=results_table['sample_count'],
            mode='lines',
            line=dict(color='lightgray', width=2),
            name='Sample Count',
            yaxis='y2'
        )
    )
    
    # Configure secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title='Sample Count',
            titlefont=dict(color='gray'),
            tickfont=dict(color='gray'),
            overlaying='y',
            side='right'
        )
    )
    
    # Save the plot if path provided
    if output_path:
        file_path = os.path.join(output_path, f"{target_cell_type}_detection_rate_continuous.html")
        fig.write_html(file_path)
        print(f"Saved visualization to {file_path}")
    
    # Save the full result table with confidence intervals
    results_table.to_csv(os.path.join(output_path, f"{target_cell_type}_detection_rates.csv"))
    
    # Save the plot
    fig.write_html(f"{str(output_path)}/{target_cell_type}_minimum_detection_concentration.html")
    
    return min_reliable_conc, results_table


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
