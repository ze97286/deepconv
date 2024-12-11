#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
import datetime
import matplotlib.pyplot as plt
from deep_conv.benchmark.benchmark_utils import process_atlas

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProportionRange:
    min_val: float
    max_val: float
    name: str
    target_fraction: float  # Desired fraction of total samples


@dataclass
class TrainingConfig:
    # Proportion ranges with higher emphasis on lower ranges
    proportion_ranges: List[ProportionRange] = None
    # Coverage parameters
    base_coverage: float = 5.0
    coverage_variability: float = 1.0
    coverage_proportion_factor: float = 1.0
    dropout_mean: float = 0.15
    dropout_std: float = 0.05
    min_coverage: int = 0

    def __post_init__(self):
        if self.proportion_ranges is None:
            self.proportion_ranges = [
                ProportionRange(0, 1e-4, "Ultra-low", 0.3),
                ProportionRange(1e-4, 1e-3, "Very-low", 0.25),
                ProportionRange(1e-3, 1e-2, "Low", 0.2),
                ProportionRange(1e-2, 1e-1, "Medium", 0.15),
                ProportionRange(1e-1, 1.0, "High", 0.1)
            ]


def simulate_coverage(
    proportions: np.ndarray,
    reference_profiles: np.ndarray,
    config: TrainingConfig
) -> np.ndarray:
    """
    Simulate realistic coverage patterns.

    Args:
        proportions (np.ndarray): Cell type proportions (n_samples x n_cell_types).
        reference_profiles (np.ndarray): Reference methylation profiles (n_cell_types x n_regions).
        config (TrainingConfig): Configuration object.

    Returns:
        np.ndarray: Simulated coverage values (n_samples x n_regions).
    """
    logger.info("Simulating coverage based on proportions and reference profiles.")
    n_samples, n_cell_types = proportions.shape
    n_regions = reference_profiles.shape[1]

    # Base coverage with variability
    base_coverage = np.random.normal(
        loc=config.base_coverage,
        scale=config.coverage_variability,
        size=(n_samples, n_regions)
    )
    base_coverage = np.maximum(base_coverage, 0)  # Allow coverage to be zero initially

    # Calculate the effect on coverage from proportions
    proportion_effect = np.dot(proportions, reference_profiles)  # Shape: (n_samples x n_regions)
    proportion_effect = 1 + np.log1p(proportion_effect * config.coverage_proportion_factor)

    # Apply the effect
    coverage = base_coverage * proportion_effect

    # Add systematic region biases
    region_bias = np.random.lognormal(mean=0, sigma=0.5, size=n_regions)
    coverage *= region_bias  # Broadcasting over samples

    # Round to integer coverage values
    coverage = np.round(coverage).astype(int)

    # Apply per-sample dropout:
    # For each sample, draw a dropout rate from a normal distribution truncated to [0,1].
    for i in range(n_samples):
        sample_dropout_rate = np.clip(
            np.random.normal(loc=config.dropout_mean, scale=config.dropout_std), 0, 1
        )
        dropout_mask = np.random.random(size=n_regions) < sample_dropout_rate
        coverage[i, dropout_mask] = 0  # Zero out selected regions

        if i % 10000 == 0 and i > 0:
            logger.debug(f"Sample {i}: Dropout rate applied: {sample_dropout_rate:.4f}")

    logger.info("Coverage simulation completed.")
    return coverage


def generate_stratified_proportions(
    n_samples: int,
    n_cell_types: int,
    config: TrainingConfig,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate stratified cell type proportions ensuring representation of all ranges.
    All proportions sum to 1 for each sample.
    """
    if random_state is not None:
        np.random.seed(random_state)

    logger.info(f"Generating stratified proportions for {n_samples} samples and {n_cell_types} cell types.")
    proportions = np.zeros((n_samples, n_cell_types))

    # Calculate samples per range
    samples_per_range = {
        range_.name: int(n_samples * range_.target_fraction)
        for range_ in config.proportion_ranges
    }

    # Adjust for rounding
    total = sum(samples_per_range.values())
    if total < n_samples:
        # Assign remaining samples to the first range
        first_range = config.proportion_ranges[0].name
        samples_per_range[first_range] += n_samples - total
    elif total > n_samples:
        # Reduce samples from the first range if over
        first_range = config.proportion_ranges[0].name
        samples_per_range[first_range] -= total - n_samples

    current_idx = 0
    for range_ in config.proportion_ranges:
        n_range_samples = samples_per_range[range_.name]
        if n_range_samples == 0:
            continue

        if range_.name == "Ultra-low":
            # Special handling for ultra-low: all cell types have low proportions
            # Generate samples where each cell type has a proportion <= range_.max_val
            # Use a Dirichlet distribution with parameters that encourage low proportions
            alpha = np.full(n_cell_types, 0.1)  # Lower alpha for more uniform and lower proportions

            for _ in range(n_range_samples):
                if current_idx >= n_samples:
                    break
                # Draw from Dirichlet
                sample = np.random.dirichlet(alpha)
                # To ensure each cell type proportion <= range_.max_val, clip and renormalize
                sample = np.minimum(sample, range_.max_val)
                sample_sum = sample.sum()
                # Handle case where all proportions are zero after clipping
                if sample_sum == 0:
                    # Assign equal small proportions
                    sample = np.full(n_cell_types, 1.0 / n_cell_types)
                else:
                    # Normalize to sum to 1
                    sample /= sample_sum

                proportions[current_idx] = sample
                current_idx += 1

        else:
            # Original logic for other ranges
            for _ in range(n_range_samples):
                if current_idx >= n_samples:
                    break

                # Randomly select number of cell types to have proportions in this range
                n_types_in_range = np.random.randint(1, max(2, n_cell_types // 2))

                # Select cell types for this range
                selected_types = np.random.choice(n_cell_types, n_types_in_range, replace=False)

                # Generate proportions for selected types within the range
                selected_props = np.random.uniform(
                    low=range_.min_val,
                    high=range_.max_val,
                    size=n_types_in_range
                )

                # Ensure that proportions sum < 1
                remaining_prop = 1.0 - selected_props.sum()
                if remaining_prop <= 0:
                    # Normalize selected_props to sum to 0.9 to leave room
                    selected_props /= selected_props.sum()
                    selected_props *= 0.9
                    remaining_prop = 0.1

                # Distribute remaining proportion among other cell types
                remaining_types = np.setdiff1d(np.arange(n_cell_types), selected_types)
                if len(remaining_types) > 0:
                    remaining_props = np.random.dirichlet(alpha=np.ones(len(remaining_types)))
                    remaining_props *= remaining_prop
                    proportions[current_idx, selected_types] = selected_props
                    proportions[current_idx, remaining_types] = remaining_props
                else:
                    # All cell types selected
                    proportions[current_idx, selected_types] = selected_props
                    # Normalize to sum=1 (for non-ultra-low ranges)
                    proportions[current_idx] /= proportions[current_idx].sum()

                current_idx += 1

    # Handle any remaining samples due to rounding
    if current_idx < n_samples:
        logger.warning(f"Assigning remaining {n_samples - current_idx} samples to random proportions.")
        for i in range(current_idx, n_samples):
            random_props = np.random.dirichlet(alpha=np.ones(n_cell_types))
            proportions[i] = random_props

    # Shuffle the samples
    np.random.shuffle(proportions)

    # Verify that all proportions sum to 1
    if not np.allclose(proportions.sum(axis=1), 1.0, atol=1e-3):
        logger.error("Some proportions do not sum to 1 after generation.")
    else:
        logger.info("All generated proportions sum to 1.")

    logger.info("Stratified proportion generation completed.")
    return proportions


def simulate_methylation_data(
    reference_profiles: np.ndarray,
    proportions: np.ndarray,
    coverage: np.ndarray,
    config: TrainingConfig,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate methylation proportions representing the proportion of reads matching the methylation profile.

    Args:
        reference_profiles (np.ndarray): Reference methylation profiles (n_cell_types x n_regions).
        proportions (np.ndarray): Cell type proportions (n_samples x n_cell_types).
        coverage (np.ndarray): Coverage values (n_samples x n_regions).
        config (TrainingConfig): Configuration object.
        random_state (Optional[int]): Seed for reproducibility.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing methylation proportions for training and validation sets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    logger.info("Simulating methylation proportions based on coverage and proportions.")

    # Calculate expected methylation proportions
    expected_methylation = np.dot(proportions, reference_profiles)  
    expected_methylation = np.clip(expected_methylation, 0, 1)

    # Simulate matching read counts using binomial distribution
    # Number of trials: coverage
    # Probability of success: expected_methylation
    n_matching_reads = np.random.binomial(n=coverage, p=expected_methylation)

    # Calculate matching proportions
    # Handle division by zero where coverage == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        matching_proportion = np.true_divide(n_matching_reads, coverage)
        # Now set places where coverage == 0 to NaN
        matching_proportion[coverage == 0] = np.nan

    logger.info("Methylation proportions simulation completed.")
    return {'X': matching_proportion}


def visualise_data(dataset_dict: Dict[str, np.ndarray], cell_types: List[str], output_dir: str):
    """
    Visualise coverage and methylation distributions for training and validation sets.

    Args:
        dataset_dict (Dict[str, np.ndarray]): Dictionary containing datasets.
        cell_types (List[str]): List of cell type names.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Coverage Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(dataset_dict['coverage_train'].flatten(), bins=100, alpha=0.5, label='Training Coverage')
    plt.hist(dataset_dict['coverage_val'].flatten(), bins=100, alpha=0.5, label='Validation Coverage')
    plt.title('Coverage Distribution')
    plt.xlabel('Coverage')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'coverage_distribution.png'))
    plt.close()

    # Methylation Proportion Distribution
    for dataset_type in ['train', 'val']:
        plt.figure(figsize=(10, 6))
        # Flatten and remove NaNs for plotting
        methyl_flat = dataset_dict[f'X_{dataset_type}'].flatten()
        methyl_flat = methyl_flat[~np.isnan(methyl_flat)]
        plt.hist(methyl_flat, bins=100, alpha=0.5, label=f'{dataset_type.capitalize()} Proportion')
        plt.title(f'Methylation Proportion Distribution - {dataset_type.capitalize()} Set')
        plt.xlabel('Proportion of Matching Reads')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'methylation_distribution_{dataset_type}.png'))
        plt.close()

    # Cell Type Proportion Distribution
    for i, cell_type in enumerate(cell_types):
        plt.figure(figsize=(10, 6))
        plt.hist(dataset_dict['y_train'][:, i], bins=50, alpha=0.5, label='Training')
        plt.hist(dataset_dict['y_val'][:, i], bins=50, alpha=0.5, label='Validation')
        plt.title(f'Proportion Distribution for {cell_type}')
        plt.xlabel('Proportion')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'proportion_distribution_{cell_type}.png'))
        plt.close()

    logger.info(f"Visualisations saved to {output_dir}")


def generate_dataset_with_coverage(
    atlas_path: str,
    n_train: int,
    n_val: int,
    fill_na: str,
    config: Optional[TrainingConfig] = None,
    random_state: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate and save training and validation datasets with coverage and methylation simulation.

    Args:
        atlas_path (str): Path to the atlas file.
        n_train (int): Number of training samples.
        n_val (int): Number of validation samples.
        config (Optional[TrainingConfig]): Configuration object. If None, defaults are used.
        random_state (Optional[int]): Seed for reproducibility.
        save_path (Optional[str]): Path to save the dataset and visualizations. If None, data is not saved.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing training and validation datasets.
    """
    if config is None:
        config = TrainingConfig()

    if random_state is not None:
        np.random.seed(random_state)

    logger.info(f"Processing atlas from {atlas_path} with fill_na='{fill_na}'.")
    reference_profiles, cell_types = process_atlas(atlas_path, fill_na)
    n_cell_types = len(cell_types)

    # Ensure reference_profiles are non-negative
    if np.any(reference_profiles < 0):
        logger.warning("Negative values detected in reference_profiles. Clipping to 0.")
        reference_profiles = np.clip(reference_profiles, 0, None)

    # Optionally, normalize reference_profiles if required
    # For example, ensure that each cell type's reference profile sums to 1
    # reference_profiles = reference_profiles / reference_profiles.sum(axis=1, keepdims=True)

    logger.info(f"Reference profiles shape: {reference_profiles.shape}")
    logger.info(f"Number of cell types: {n_cell_types}")

    logger.info("Generating training data.")
    # Generate training data
    train_proportions = generate_stratified_proportions(n_train, n_cell_types, config, random_state)
    train_coverage = simulate_coverage(train_proportions, reference_profiles, config)
    train_methylation = simulate_methylation_data(
        reference_profiles, train_proportions, train_coverage, config, random_state
    )

    logger.info("Generating validation data.")
    # Generate validation data
    val_proportions = generate_stratified_proportions(n_val, n_cell_types, config, random_state)
    val_coverage = simulate_coverage(val_proportions, reference_profiles, config)
    val_methylation = simulate_methylation_data(
        reference_profiles, val_proportions, val_coverage, config, random_state
    )

    logger.info("Dataset generation completed.")

    # Data integrity checks
    # Check proportions sum to 1
    if not np.allclose(train_proportions.sum(axis=1), 1.0, atol=1e-3):
        # Log detailed information for debugging
        problematic = np.abs(train_proportions.sum(axis=1) - 1.0) > 1e-3
        num_problematic = np.sum(problematic)
        logger.error(f"{num_problematic} training samples do not sum to 1.")
        raise ValueError("Training proportions do not sum to 1.")
    if not np.allclose(val_proportions.sum(axis=1), 1.0, atol=1e-3):
        problematic = np.abs(val_proportions.sum(axis=1) - 1.0) > 1e-3
        num_problematic = np.sum(problematic)
        logger.error(f"{num_problematic} validation samples do not sum to 1.")
        raise ValueError("Validation proportions do not sum to 1.")

    # Check coverage meets minimum threshold
    if not np.all(train_coverage >= config.min_coverage):
        # Log detailed information for debugging
        below_min = train_coverage < config.min_coverage
        num_below = np.sum(below_min)
        logger.error(f"{num_below} training coverage values are below the minimum threshold of {config.min_coverage}.")
        raise ValueError("Training coverage below minimum threshold detected.")
    if not np.all(val_coverage >= config.min_coverage):
        below_min = val_coverage < config.min_coverage
        num_below = np.sum(below_min)
        logger.error(f"{num_below} validation coverage values are below the minimum threshold of {config.min_coverage}.")
        raise ValueError("Validation coverage below minimum threshold detected.")

    dataset_dict = {
        'X_train': train_methylation['X'],
        'coverage_train': train_coverage,
        'y_train': train_proportions,
        'X_val': val_methylation['X'],
        'coverage_val': val_coverage,
        'y_val': val_proportions,
        'cell_types': cell_types,
        'reference_profiles': reference_profiles
    }

    # Verify total coverage counts
    total_train_coverage = dataset_dict['coverage_train'].size
    total_val_coverage = dataset_dict['coverage_val'].size
    logger.info(f"Total training coverage data points: {total_train_coverage}")
    logger.info(f"Total validation coverage data points: {total_val_coverage}")

    # Sum of frequencies
    train_coverage_sum = np.bincount(dataset_dict['coverage_train'].flatten())
    val_coverage_sum = np.bincount(dataset_dict['coverage_val'].flatten())

    logger.info(f"Training Coverage - Unique coverage values: {len(train_coverage_sum)}")
    logger.info(f"Validation Coverage - Unique coverage values: {len(val_coverage_sum)}")

    logger.info(f"Training Coverage - Min: {dataset_dict['coverage_train'].min()}, Max: {dataset_dict['coverage_train'].max()}")
    logger.info(f"Validation Coverage - Min: {dataset_dict['coverage_val'].min()}, Max: {dataset_dict['coverage_val'].max()}")

    if save_path is not None:
        # Create output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Visualise data
        logger.info("Generating visualisations.")
        visualise_data(dataset_dict, cell_types, save_path)

        # Save the dataset with timestamp
        exp_name = f"training_set.npz"
        output_file_path = os.path.join(save_path, exp_name)
        logger.info(f"Saving dataset to {output_file_path}")
        np.savez_compressed(output_file_path, **dataset_dict)
        logger.info("Dataset saved successfully.")

    return dataset_dict


# Example command to run the script:
# python prepare_training.py \
#     --n_train 1000 \
#     --n_val 200 \
#     --atlas_path /path/to/atlas.csv \
#     --output_path /path/to/output/ \
#     --fill_na drop

def main():
    parser = argparse.ArgumentParser(description="Sample Generation for cfDNA Cell Type Deconvolution")
    parser.add_argument("--n_train", type=int, default=100000, help="Number of training samples (default: 100000)")
    parser.add_argument("--n_val", type=int, default=20000, help="Number of validation samples (default: 20000)")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to the atlas file (CSV format)")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the generated datasets and visualisations")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--fill_na", type=str, help="What to do with nans in the atlas (drop, mean_cell_type, mean)")
    args = parser.parse_args()

    generate_dataset_with_coverage(
        atlas_path=args.atlas_path,
        n_train=args.n_train,
        n_val=args.n_val,
        config=None,
        random_state=args.random_state,
        save_path=args.output_path,
        fill_na=args.fill_na,
    )


if __name__ == "__main__":
    main()