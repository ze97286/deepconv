import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from deep_conv.deconvolution.preprocess_pats import pats_to_homog
from deep_conv.benchmark.benchmark_utils import *

class GroupedDeconvolver:
    def __init__(self, reference_profiles,cell_types, correlation_threshold=0.3):
        """
        Initialize deconvolver with reference profiles and automatic grouping
        Args:
            reference_profiles: Matrix H (n_components × n_markers)
            correlation_threshold: Threshold for grouping cell types
        """
        self.cell_types = cell_types
        self.H = torch.FloatTensor(reference_profiles)
        self.n_components = reference_profiles.shape[0]
        self.n_markers = reference_profiles.shape[1]
        # Create groups based on correlation
        self.groups = self._create_groups(reference_profiles, correlation_threshold)
        self.n_groups = len(self.groups)
        # Create grouping matrix G
        self.G = self._create_grouping_matrix()
    def _create_groups(self, H, threshold=0.3):
        """Create cell type groups based on correlation and known relationships"""
        # Calculate correlation between cell types
        corr_matrix = np.corrcoef(H)
        # First, handle known biological groups
        known_groups = {
            'T_cells': ['CD4-T-cells', 'CD8-T-cells'],
            # Add other known biological groupings here
        }
        # Find indices for known groups
        groups = []
        used_indices = set()
        for group_name, cell_types in known_groups.items():
            group_indices = [i for i, ct in enumerate(self.cell_types) if ct in cell_types]
            if len(group_indices) > 1:  # Only add group if we found multiple cell types
                groups.append(group_indices)
                used_indices.update(group_indices)
        # For remaining cell types, look at correlations
        for i in range(self.n_components):
            if i in used_indices:
                continue
            
            # Find correlated cell types
            correlations = corr_matrix[i]
            corr_indices = [j for j in range(self.n_components) 
                        if j not in used_indices and i != j and abs(correlations[j]) > threshold]
            
            if corr_indices:  # If we found correlations
                group = [i] + corr_indices
                groups.append(group)
                used_indices.update(group)
            else:  # If no correlations, cell type goes in its own group
                groups.append([i])
                used_indices.add(i)
        return groups
    def _create_grouping_matrix(self):
        """Create binary grouping matrix G"""
        G = torch.zeros((self.n_groups, self.n_components))
        for i, group in enumerate(self.groups):
            G[i, group] = 1
        return G
    def _optimize_group_proportions(self, X, coverage=None, max_iter=1000, tol=1e-6):
        """
        Optimize group proportions (A) and within-group proportions (B)
        """
        n_samples = X.shape[0]
        # Initialize A and B
        A = torch.rand(n_samples, self.n_groups)
        A = F.normalize(A, p=1, dim=1)
        B = torch.zeros(self.n_groups, self.n_components)
        for i, group in enumerate(self.groups):
            B[i, group] = 1.0 / len(group)
        # Mask for valid measurements
        if coverage is not None:
            mask = (coverage > 0).float()
        else:
            mask = (~torch.isnan(X)).float()
        X = torch.where(mask > 0, X, torch.zeros_like(X))
        for iteration in range(max_iter):
            # Update A (group proportions)
            BH = torch.mm(B, self.H)  # (n_groups × n_markers)
            WH = torch.mm(A, BH)      # (n_samples × n_markers)
            # Compute numerator and denominator for A update
            A_num = torch.mm(X * mask, BH.t())
            A_denom = torch.mm(WH * mask, BH.t())
            # Update A
            A = A * (A_num / (A_denom + 1e-10))
            A = F.normalize(A, p=1, dim=1)
            # Update B
            AtX = torch.mm(A.t(), X * mask)  # (n_groups × n_markers)
            AtWH = torch.mm(A.t(), WH * mask)  # (n_groups × n_markers)
            for i, group in enumerate(self.groups):
                B_num = torch.mm(AtX[i:i+1, :], self.H[group, :].t())
                B_denom = torch.mm(AtWH[i:i+1, :], self.H[group, :].t())
                B[i, group] *= (B_num / (B_denom + 1e-10)).squeeze()
                B[i, group] = F.normalize(B[i, group], p=1, dim=0)
            # Calculate loss
            X_pred = torch.mm(torch.mm(A, B), self.H)
            loss = torch.sum(mask * (X - X_pred)**2)
            if iteration > 0 and abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        return A, B
    def predict(self, X, coverage=None):
        """
        Predict cell type proportions
        
        Args:
            X: Methylation matrix (n_samples × n_markers)
            coverage: Coverage matrix of same shape as X
        """
        X = torch.FloatTensor(X)
        if coverage is not None:
            coverage = torch.FloatTensor(coverage)
        # Optimize A and B
        A, B = self._optimize_group_proportions(X, coverage)
        # Final proportions
        W = torch.mm(A, B)
        return {
            'proportions': W.numpy(),
            'group_proportions': A.numpy(),
            'within_group_proportions': B.numpy(),
            'groups': self.groups
        }
    def visualize_groups(self, H, cell_types):
        """
        Visualize correlation matrix and resulting groups
        
        Args:
            H: Reference profiles matrix
            cell_types: List of cell type names
        
        Returns:
            fig: Plotly figure with multiple subplots
        """
        # Create correlation matrix
        corr_matrix = np.corrcoef(H)
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Correlation Matrix', 'Grouped Structure'),
            specs=[[{"type": "heatmap"}, {"type": "bar"}]]
        )
        # Add correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=cell_types,
                y=cell_types,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title='Correlation'),
                hoverongaps=False
            ),
            row=1, col=1
        )
        # Create grouped bar plot
        groups_data = []
        colors = px.colors.qualitative.Set3
        for i, group in enumerate(self.groups):
            color = colors[i % len(colors)]
            # Add bars for each cell type in group
            for idx in group:
                groups_data.append({
                    'Cell Type': cell_types[idx],
                    'Group': f'Group {i+1}',
                    'Value': 1,
                    'color': color
                })
        groups_df = pd.DataFrame(groups_data)
        fig.add_trace(
            go.Bar(
                x=groups_df['Cell Type'],
                y=groups_df['Value'],
                marker_color=groups_df['color'],
                name='Groups',
                text=groups_df['Group'],
                textposition='inside'
            ),
            row=1, col=2
        )
        # Update layout
        fig.update_layout(
            height=600,
            width=1200,
            showlegend=False,
            title_text="Cell Type Groups Analysis"
        )
        # Update axes
        fig.update_xaxes(tickangle=45)
        return fig
    def print_group_summary(self, cell_types):
        """
        Print text summary of groups and their correlations
        """
        corr_matrix = np.corrcoef(self.H)
        print("Group Summary:")
        print("-" * 50)
        for i, group in enumerate(self.groups):
            if len(group) > 1:
                print(f"\nGroup {i+1}:")
                print("Cell Types:", ", ".join([cell_types[idx] for idx in group]))
                
                # Print correlation matrix for this group
                print("\nCorrelations within group:")
                group_corr = corr_matrix[np.ix_(group, group)]
                group_types = [cell_types[idx] for idx in group]
                corr_df = pd.DataFrame(
                    group_corr,
                    index=group_types,
                    columns=group_types
                )
                print(corr_df.round(3))
            else:
                print(f"\nGroup {i+1}: {cell_types[group[0]]} (Single cell type)")
        print("\n" + "-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment: Weighted NNLS optimisation with R² Metric for Cell Type Deconvolution")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to the altas")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the save figures")
    parser.add_argument("--pats_path", type=str, required=True, help="Path to the pats dir")
    parser.add_argument("--wgbs_tools_exec_path",help="path to wgbs_tools executable",required=True)
    parser.add_argument("--cell_type",help="cell type to analyse",required=True)
    args = parser.parse_args()
    H, cell_types = process_atlas(args.atlas_path)
    deconvolver = GroupedDeconvolver(H, cell_types)
    fig = deconvolver.visualize_groups(H, cell_types)
    os.makedirs(args.save_path, exist_ok=True)
    fig.write_html(os.path.join(args.save_path, "grouping.html"))

    marker_read_proportions, counts = pats_to_homog(args.atlas_path,args.pats_path,args.wgbs_tools_exec_path) 
    marker_read_proportions = marker_read_proportions.drop(columns=['name','direction'])[columns()].T.to_numpy()
    counts = counts.drop(columns=['name','direction'])[columns()].T.to_numpy()
    results = deconvolver.predict(marker_read_proportions,counts)
    proportions = results['proportions']
    group_props = results['group_proportions']
    within_group_props = results['within_group_proportions']
    estimated_val = proportions
    cell_type_index = cell_types.index(args.cell_type)
    cell_type_contribution = estimated_val[:,cell_type_index]
    d = {"dilution":dilutions(), "contribution":cell_type_contribution}
    results_df = pd.DataFrame.from_dict(d)
    metrics = calculate_dilution_metrics(results_df)
    print(metrics.head())
    all_predictions_df = pd.DataFrame(
        estimated_val,
        columns=cell_types,
        index=columns()
    )
    plot_dilution_results(results_df, args.cell_type, all_predictions_df, args.save_path)
    metrics.to_csv(os.path.join(args.save_path,"metrics_out.csv"), index=False)


if __name__ == "__main__":
    main()