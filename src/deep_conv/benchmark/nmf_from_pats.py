#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from deep_conv.benchmark.benchmark_utils import *
from deep_conv.deconvolution.preprocess_pats import pats_to_homog

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
    # Multiply each methylation proportion by its corresponding coverage
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
    H = reference_profiles.T  # Shape: (n_markers x n_cell_types)
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
    parser = argparse.ArgumentParser(description="Baseline Experiment: Weighted NNLS optimisation with R² Metric for Cell Type Deconvolution")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to the altas")
    parser.add_argument("--pats_path", type=str, required=True, help="Path to the pats dir")
    parser.add_argument("--wgbs_tools_exec_path",help="path to wgbs_tools executable",required=True)
    parser.add_argument("--save_estimates", action='store_true', help="Flag to save the estimated proportions")
    parser.add_argument("--save_path", type=str, default="estimated_proportions", help="Directory to save estimated proportions if --save_estimates is set")
    parser.add_argument("--cell_type",help="cell type to analyse",required=True)
    parser.add_argument("--regularisation", type=str, default='ridge', choices=['ridge', 'lasso', 'elasticnet'], help="Type of regularisation to apply")
    parser.add_argument("--alpha", type=float, default=1.0, help="Regularisation strength")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="Ratio between L1 and L2 regularisation (only for elasticnet)")
    args = parser.parse_args()
    
    marker_read_proportions, counts = pats_to_homog(
            atlas_path=args.atlas_path,
            pats_path=args.pats_path,
            wgbs_tools_exec_path=args.wgbs_tools_exec_path,
    )

    marker_read_proportions = marker_read_proportions.drop(columns=['name','direction'])[columns()].T.to_numpy()
    counts = counts.drop(columns=['name','direction'])[columns()].T.to_numpy()
    reference_profiles, cell_types = process_atlas(args.atlas_path)
    cell_type_index = cell_types.index(args.cell_type)

    marker_read_proportions_weighted = apply_coverage_weighting(marker_read_proportions, counts)
    estimated_val = reference_based_weighted_nmf(
        X_weighted=marker_read_proportions_weighted,
        coverage=counts,
        reference_profiles=reference_profiles,
        regularisation=args.regularisation,
        alpha=args.alpha
    )

    logger.info("Completed Weighted NMF optimisation for Validation Set.")
    cell_type_contribution = estimated_val[:,cell_type_index]
    d = {"dilution":dilutions(), "contribution":cell_type_contribution, "sample":columns()}
    results_df = pd.DataFrame.from_dict(d)
    print(results_df.head(50))
    metrics = calculate_dilution_metrics(results_df)
    print(metrics.head(50))
    all_predictions_df = pd.DataFrame(
        estimated_val,
        columns=cell_types,
        index=columns()
    )
    
    if args.save_estimates:
        os.makedirs(args.save_path, exist_ok=True)
        plot_dilution_results(
            results_df, args.cell_type, all_predictions_df, args.save_path
        )
        estimate_out = os.path.join(args.save_path, "nmf_estimations.csv")
        metrics_out = os.path.join(args.save_path, "nmf_metrics.csv")
        results_df.to_csv(estimate_out, index=False)
        metrics.to_csv(metrics_out, index=False)


if __name__ == "__main__":
    main()