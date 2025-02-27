#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.optimize import nnls
from deep_conv.benchmark.benchmark_utils import *


def run_weighted_nnls(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray
) -> np.ndarray:
    """
    Perform coverage-weighted NNLS optimisation for each sample to estimate cell type proportions.

    Args:
        X (np.ndarray): Methylation proportions matrix (n_samples x n_markers).
        coverage (np.ndarray): Coverage matrix (n_samples x n_markers).
        reference_profiles (np.ndarray): Reference methylation profiles (n_cell_types x n_markers).

    Returns:
        np.ndarray: Estimated cell type proportions matrix (n_samples x n_cell_types).
    """
    n_samples = X.shape[0]
    n_cell_types = reference_profiles.shape[0]
    estimated_proportions = np.zeros((n_samples, n_cell_types))
    A_original = reference_profiles.T
    for i in range(n_samples):
        coverage_i = coverage[i] 
        X_i = X[i]               
        A_weighted = A_original * coverage_i[:, np.newaxis] 
        b_weighted = X_i * coverage_i                     
        zero_coverage_mask = coverage_i == 0
        if np.any(zero_coverage_mask):
            A_weighted[zero_coverage_mask, :] = 0
            b_weighted[zero_coverage_mask] = 0
        estimated, _ = nnls(A_weighted, b_weighted)
        if estimated.sum() > 0:
            estimated /= estimated.sum()
        estimated_proportions[i] = estimated
    return estimated_proportions

def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment: Weighted NNLS optimisation with R² Metric for Cell Type Deconvolution")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the synthetic dataset (.npz file)")
    parser.add_argument("--save_estimates", action='store_true', help="Flag to save the estimated proportions")
    parser.add_argument("--save_path", type=str, default="estimated_proportions", help="Directory to save estimated proportions if --save_estimates is set")
    args = parser.parse_args()
    
    data = load_dataset(args.dataset_path)
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    cell_types = data['cell_types']
    reference_profiles = data['reference_profiles']
    logger.info("Starting Weighted NNLS optimisation for Training Set.")
    estimated_train = run_weighted_nnls(X_train, data['coverage_train'], reference_profiles)
    logger.info("Completed Weighted NNLS optimisation for Training Set.")
    logger.info("Starting Weighted NNLS optimisation for Validation Set.")
    estimated_val = run_weighted_nnls(X_val, data['coverage_val'], reference_profiles)
    logger.info("Completed Weighted NNLS optimisation for Validation Set.")
    logger.info("Evaluating performance on Training Set.")
    metrics_train = evaluate_performance(y_train, estimated_train, cell_types)
    log_metrics(metrics_train)
    logger.info("Evaluating performance on Validation Set.")
    metrics_val = evaluate_performance(y_val, estimated_val, cell_types)
    log_metrics(metrics_val)
    if args.save_estimates:
        save_estimated_proportions(estimated_train, estimated_val, args.save_path)

if __name__ == "__main__":
    main()