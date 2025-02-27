import argparse

import numpy as np
from scipy.optimize import nnls
from scipy.linalg import norm

import itertools
import pandas as pd
from tqdm import tqdm

from deep_conv.benchmark.benchmark_utils import *
from deep_conv.deconvolution.preprocess_pats import pats_to_homog


class IterativeRefinementDeconvolver:
    def __init__(self, reference_profiles, n_refinements=3, base_threshold=1e-4, 
                 selection_ratio=0.4, min_components=3):
        self.H = reference_profiles
        self.n_refinements = n_refinements
        self.base_threshold = base_threshold
        self.selection_ratio = selection_ratio
        self.min_components = min_components

    def _solve_weighted_nnls(self, X, H, coverage, active_set=None):
        """
        Solve weighted NNLS with shape debugging
        """
        sqrt_coverage = np.sqrt(coverage)
        X_weighted = X * sqrt_coverage
        # For NNLS we need: min ||Aw - b||
        # where A should be (n_markers x n_components) and b should be (n_markers,)
        H_weighted = (H * sqrt_coverage[np.newaxis, :]).T  # Transpose to get (n_markers x n_components)
        if active_set is not None:
            H_weighted = H_weighted[:, active_set]
        # Now solve min ||H_weighted w - X_weighted||
        w, residual = nnls(H_weighted, X_weighted)
        if active_set is not None:
            w_full = np.zeros(H.shape[0])
            w_full[active_set] = w
            w = w_full
        
        return w
    

    def _identify_potential_components(self, X, coverage, current_W=None, iteration=0):
        """
        Enhanced component selection strategy
        
        Args:
            X: Observed methylation values
            coverage: Coverage values
            current_W: Current estimate of W (if available)
            iteration: Current refinement iteration
        """
        # Weight residual by coverage
        if current_W is not None:
            residual = X - np.dot(current_W, self.H)
        else:
            residual = X
        
        # Compute correlations with coverage weighting
        weighted_residual = residual * np.sqrt(coverage)
        correlations = np.array([
            np.abs(np.corrcoef(weighted_residual, self.H[i, :])[0, 1])
            for i in range(self.H.shape[0])
        ])
        
        # Handle NaN correlations (can happen with zero variance)
        correlations = np.nan_to_num(correlations)
        
        # Dynamic threshold based on iteration
        # Start permissive, become more selective
        selection_threshold = np.max(correlations) * (0.3 + 0.1 * iteration)
        
        # Get components above threshold
        threshold_indices = np.where(correlations > selection_threshold)[0]
        
        # Always include some minimum number of top components
        if len(threshold_indices) < self.min_components:
            top_indices = np.argsort(correlations)[-self.min_components:]
            threshold_indices = np.union1d(threshold_indices, top_indices)
        
        # Include currently active components if their proportion is significant
        if current_W is not None:
            active_indices = np.where(current_W > self.base_threshold * (1.0 + iteration))[0]
            threshold_indices = np.union1d(threshold_indices, active_indices)
        
        return threshold_indices
    
    def solve_single(self, X, coverage):
        """
        Solve for a single sample with improved refinement
        """
        # Initial solution using all components
        W_current = self._solve_weighted_nnls(X, self.H, coverage)
        best_W = W_current.copy()
        best_error = np.inf
        
        for iteration in range(self.n_refinements):
            # Identify components with iteration-aware selection
            active_set = self._identify_potential_components(X, coverage, W_current, iteration)
            
            # Solve restricted problem
            W_refined = self._solve_weighted_nnls(X, self.H, coverage, active_set)
            
            # Compute weighted error
            error = np.sum(coverage * (X - np.dot(W_refined, self.H))**2)
            
            # Keep best solution
            if error < best_error:
                best_error = error
                best_W = W_refined.copy()
            
            # Update current solution
            W_current = W_refined
            
            # If solution hasn't changed significantly, stop
            if np.allclose(W_current, best_W, rtol=1e-5, atol=1e-8):
                break
        
        # Return best solution found
        best_W = best_W / np.sum(best_W)
        return best_W
    
    def predict(self, X, coverage):
        """
        Predict cell type contributions for multiple samples
        Args:
            X: Matrix of methylation values (n_samples x n_markers)
            coverage: Coverage matrix (n_samples x n_markers)
        """
        results = []
        for i in range(X.shape[0]):
            # Handle NaN values based on coverage
            x = X[i].copy()
            c = coverage[i]
            x[c == 0] = 0
            w = self.solve_single(x, c)
            results.append(w)
        return np.array(results)
    
    def fit(self, X_train=None, W_train=None, coverage_train=None):
        """
        No training needed for this method, but included for API consistency
        """
        pass


def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment: Weighted NNLS optimisation with R² Metric for Cell Type Deconvolution")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to the altas")
    parser.add_argument("--pats_path", type=str, required=True, help="Path to the pats dir")
    parser.add_argument("--wgbs_tools_exec_path",help="path to wgbs_tools executable",required=True)
    parser.add_argument("--cell_type",help="cell type to analyse",required=True)
    parser.add_argument("--save_estimates", action='store_true', help="Flag to save the estimated proportions")
    parser.add_argument("--save_path", type=str, default="estimated_proportions", help="Directory to save estimated proportions if --save_estimates is set")
    args = parser.parse_args()
    
    # Load dataset
    marker_read_proportions, counts = pats_to_homog(
            atlas_path=args.atlas_path,
            pats_path=args.pats_path,
            wgbs_tools_exec_path=args.wgbs_tools_exec_path,
    )

    marker_read_proportions = marker_read_proportions.drop(columns=['name','direction'])[columns()].T.to_numpy()
    counts = counts.drop(columns=['name','direction'])[columns()].T.to_numpy()
    reference_profiles, cell_types = process_atlas(args.atlas_path)
    cell_type_index = cell_types.index(args.cell_type)

    best_params = {
        'n_refinements': 7,
        'base_threshold': 0.005,
        'min_components': 7,
        # 'coverage_power': 1.5
    }

    class ModifiedDeconvolver(IterativeRefinementDeconvolver):
        def _solve_weighted_nnls(self, X, H, coverage, active_set=None):
            # Weight by coverage^power instead of sqrt(coverage)
            coverage_weight = np.power(coverage, 1.5)
            X_weighted = X * coverage_weight
            H_weighted = (H * coverage_weight[np.newaxis, :]).T
            
            if active_set is not None:
                H_weighted = H_weighted[:, active_set]
            
            w, residual = nnls(H_weighted, X_weighted)
            
            if active_set is not None:
                w_full = np.zeros(H.shape[0])
                w_full[active_set] = w
                w = w_full
                
            return w

    deconvolver = ModifiedDeconvolver(
        reference_profiles=reference_profiles, **best_params
    )

    estimated_val = deconvolver.predict(
        X=marker_read_proportions,  
        coverage=counts            
    )

    logger.info("Completed Weighted NNLS optimisation for Validation Set.")
    cell_type_contribution = estimated_val[:,cell_type_index]
    d = {"dilution":dilutions(), "contribution":cell_type_contribution}
    results_df = pd.DataFrame.from_dict(d)
    print(results_df.head(50))
    metrics = calculate_dilution_metrics(results_df)
    print(metrics.head(50))
    if args.save_estimates:
        os.makedirs(args.save_path, exist_ok=True)
        estimate_out = os.path.join(args.save_path, "iterative_estimations.csv")
        metrics_out = os.path.join(args.save_path, "iterative_metrics.csv")
        results_df.to_csv(estimate_out, index=False)
        metrics.to_csv(metrics_out, index=False)

    
if __name__ == "__main__":
    main()