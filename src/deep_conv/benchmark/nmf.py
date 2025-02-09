#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from deep_conv.benchmark.benchmark_utils import *

import cvxpy as cp


def apply_coverage_weighting(X: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """
    Apply coverage weighting to the methylation proportions.

    Args:
        X (np.ndarray): Methylation proportions matrix (n_samples x n_markers).
        coverage (np.ndarray): Coverage matrix (n_samples x n_markers).

    Returns:
        np.ndarray: Coverage-weighted methylation proportions matrix (n_samples x n_markers).
    """
    X_weighted = X * coverage
    logger.info("Applied coverage weighting to methylation proportions.")
    return X_weighted


def reference_based_weighted_nmf(
    X_weighted: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    regularisation: str = 'none',  # Options: 'none', 'ridge', 'lasso'
    alpha: float = 0.0             # Regularisation strength
) -> np.ndarray:
    """
    Perform Reference-Based Weighted NMF where H is fixed to reference_profiles and coverage is used as weights.

    Args:
        X_weighted (np.ndarray): Coverage-weighted methylation proportions matrix (n_samples x n_markers).
        coverage (np.ndarray): Coverage matrix (n_samples x n_markers).
        reference_profiles (np.ndarray): Reference methylation profiles (n_cell_types x n_markers).
        regularisation (str): Type of regularisation ('none', 'ridge', 'lasso').
        alpha (float): Regularisation strength.

    Returns:
        np.ndarray: Estimated cell type proportions matrix (n_samples x n_cell_types).
    """
    n_samples, _ = X_weighted.shape
    n_cell_types = reference_profiles.shape[0]
    H = reference_profiles.T 
    W = cp.Variable((n_cell_types, n_samples), nonneg=True)
    W_mat = coverage 
    weighted_residual = cp.multiply(W_mat.T, (H @ W)) - X_weighted.T 
    data_fit = cp.norm(weighted_residual, 'fro')**2
    if regularisation == 'ridge':
        regularisation_term = alpha * cp.sum_squares(W)
    elif regularisation == 'lasso':
        regularisation_term = alpha * cp.norm1(W)
    elif regularisation == 'none':
        regularisation_term = 0
    else:
        raise ValueError(f"Unsupported regularisation type: {regularisation}")

    objective = cp.Minimize(data_fit + regularisation_term)
    constraints = [cp.sum(W, axis=0) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if W.value is not None:
        W_estimated = W.value.T
        W_estimated /= W_estimated.sum(axis=1, keepdims=True)
        logger.info("Reference-Based Weighted NMF optimisation completed successfully.")
        return W_estimated
    else:
        logger.error("Reference-Based Weighted NMF Optimisation failed.")
        raise ValueError("Reference-Based Weighted NMF Optimisation failed.")


def main():
    parser = argparse.ArgumentParser(description="Reference-Based Weighted NMF Cell Type Deconvolution with Coverage Weighting")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the synthetic dataset (.npz file)")
    parser.add_argument("--regularisation", type=str, default='none', choices=['none', 'ridge', 'lasso'], help="Type of regularisation to apply")
    parser.add_argument("--alpha", type=float, default=0.0, help="Regularisation strength")
    parser.add_argument("--save_estimates", action='store_true', help="Flag to save the estimated proportions")
    parser.add_argument("--save_path", type=str, default="estimated_proportions_reference_based_weighted_nmf", help="Directory to save estimated proportions if --save_estimates is set")
    args = parser.parse_args()
    
    data = load_dataset(args.dataset_path)
    X_train = data['X_train']                  
    coverage_train = data['coverage_train']    
    y_train = data['y_train']                  
    X_val = data['X_val']                      
    coverage_val = data['coverage_val']        
    y_val = data['y_val']                      
    cell_types = data['cell_types']            
    reference_profiles = data['reference_profiles'] 
    
    logger.info("Applying coverage weighting to Training and Validation Sets.")
    X_train_weighted = apply_coverage_weighting(X_train, coverage_train)
    X_val_weighted = apply_coverage_weighting(X_val, coverage_val)
    
    logger.info("Starting Reference-Based Weighted NMF optimisation for Training Set.")
    estimated_train = reference_based_weighted_nmf(
        X_weighted=X_train_weighted,
        coverage=coverage_train,
        reference_profiles=reference_profiles,
        regularisation=args.regularisation,
        alpha=args.alpha
    )
    logger.info("Completed Reference-Based Weighted NMF optimisation for Training Set.")
    
    logger.info("Starting Reference-Based Weighted NMF optimisation for Validation Set.")
    estimated_val = reference_based_weighted_nmf(
        X_weighted=X_val_weighted,
        coverage=coverage_val,
        reference_profiles=reference_profiles,
        regularisation=args.regularisation,
        alpha=args.alpha
    )
    logger.info("Completed Reference-Based Weighted NMF optimisation for Validation Set.")
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