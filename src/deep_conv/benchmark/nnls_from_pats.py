#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from deep_conv.benchmark.benchmark_utils import *
from deep_conv.deconvolution.preprocess_pats import pats_to_homog


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
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to the altas")
    parser.add_argument("--pats_path", type=str, required=True, help="Path to the pats dir")
    parser.add_argument("--wgbs_tools_exec_path",help="path to wgbs_tools executable",required=True)
    parser.add_argument("--cell_type",help="cell type to analyse",required=True)
    parser.add_argument("--save_estimates", action='store_true', help="Flag to save the estimated proportions")
    parser.add_argument("--save_path", type=str, default="estimated_proportions", help="Directory to save estimated proportions if --save_estimates is set")
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

    logger.info("Starting Weighted NNLS optimisation for Validation Set.")
    estimated_val = run_weighted_nnls(marker_read_proportions, counts, reference_profiles)
    logger.info("Completed Weighted NNLS optimisation for Validation Set.")
    cell_type_contribution = estimated_val[:,cell_type_index]
    d = {"dilution":dilutions(), "contribution":cell_type_contribution}
    results_df = pd.DataFrame.from_dict(d)
    print(results_df.head(50))
    metrics = calculate_dilution_metrics(results_df)
    all_predictions_df = pd.DataFrame(
        estimated_val,
        columns=cell_types,
        index=columns()
    )
    

    print(metrics.head(50))
    if args.save_estimates:
        os.makedirs(args.save_path, exist_ok=True)
        plot_dilution_results(
            results_df, args.cell_type, all_predictions_df, args.save_path
        )
        estimate_out = os.path.join(args.save_path, "nnls_estimations.csv")
        metrics_out = os.path.join(args.save_path, "nnls_metrics.csv")
        results_df.to_csv(estimate_out, index=False)
        metrics.to_csv(metrics_out, index=False)

    
if __name__ == "__main__":
    main()