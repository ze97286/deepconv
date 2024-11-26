import numpy as np
import pandas as pd
from scipy.stats import nbinom

def process_atlas(atlas_path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(atlas_path, sep='\t')
    cell_type_cols = df.columns[8:]
    methylation_matrix = df[cell_type_cols].replace('NA', np.nan).astype(float)
    methylation_matrix = methylation_matrix.fillna(methylation_matrix.mean())
    reference_profiles = methylation_matrix.to_numpy().T
    return reference_profiles, cell_type_cols.tolist()


def create_synthetic_mixtures(reference_profiles: np.ndarray, 
                            n_samples: int,
                            alpha: float = 0.1,
                            noise_std: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    n_cell_types = reference_profiles.shape[0]
    proportions = np.random.dirichlet(
        alpha=[alpha] * n_cell_types, 
        size=n_samples
    )
    mixtures = np.dot(proportions, reference_profiles)
    noise = np.random.normal(0, noise_std, mixtures.shape)
    mixtures = np.clip(mixtures + noise, 0, 1)
    return mixtures, proportions


def create_synthetic_mixtures_with_coverage(
    reference_profiles: np.ndarray, 
    n_samples: int,
    mean_coverage: int = 30,
    alpha: float = 0.1,
    noise_std: float = 0.02
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic cfDNA methylation samples with coverage information."""
    n_cell_types = reference_profiles.shape[0]
    proportions = np.random.dirichlet(
        alpha=[alpha] * n_cell_types, 
        size=n_samples
    )
    true_mixtures = np.dot(proportions, reference_profiles)
    true_mixtures = np.clip(true_mixtures, 0, 1)
    r = 3  # dispersion parameter
    p = r / (r + mean_coverage)  # probability parameter
    coverage = nbinom.rvs(r, p, size=true_mixtures.shape)
    # Generate methylated read counts
    # Using binomial distribution for each site
    methylated_reads = np.zeros_like(coverage, dtype=float)
    for i in range(coverage.shape[0]):
        for j in range(coverage.shape[1]):
            if coverage[i,j] > 0:
                methylated_reads[i,j] = np.random.binomial(
                    coverage[i,j], 
                    true_mixtures[i,j]
                )
    observed_mixtures = np.zeros_like(true_mixtures)
    mask = coverage > 0
    observed_mixtures[mask] = methylated_reads[mask] / coverage[mask]
    noise = np.random.normal(0, noise_std, observed_mixtures.shape)
    observed_mixtures = np.clip(observed_mixtures + noise, 0, 1)
    return true_mixtures, observed_mixtures, proportions, coverage


def generate_dataset_with_coverage(atlas_path, n_train, n_val):
    """Generate a complete training dataset with coverage information."""
    reference_profiles, cell_types = process_atlas(atlas_path)
    true_train, X_train, y_train, cov_train = create_synthetic_mixtures_with_coverage(
        reference_profiles, 
        n_samples=n_train,
        mean_coverage=30
    )
    true_val, X_val, y_val, cov_val = create_synthetic_mixtures_with_coverage(
        reference_profiles, 
        n_samples=n_val,
        mean_coverage=20,
        noise_std=0.03
    )

    return {
        'X_train': X_train,
        'y_train': y_train,
        'coverage_train': cov_train,
        'true_train': true_train,
        'X_val': X_val,
        'y_val': y_val,
        'coverage_val': cov_val,
        'true_val': true_val,
        'cell_types': cell_types,
        'reference_profiles': reference_profiles,        
    }

