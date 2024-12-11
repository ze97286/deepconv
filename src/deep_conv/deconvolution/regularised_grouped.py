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
from tqdm import tqdm, trange


class RegularisedGroupedDeconvolver(nn.Module):
    def __init__(self, reference_profiles, cell_types, correlation_threshold=0.3, variance_penalty=0.1):
        super().__init__()
        
        self.H = torch.FloatTensor(reference_profiles)
        self.cell_types = cell_types
        self.variance_penalty = variance_penalty
        self.n_components = reference_profiles.shape[0]
        self.n_markers = reference_profiles.shape[1]
        
        # Create groups
        self.groups = self._create_groups(reference_profiles, correlation_threshold)
        self.n_groups = len(self.groups)
        
        # Initialize learnable parameters as nn.Parameter
        self.group_weights = nn.Parameter(torch.ones(self.n_groups))
        self.within_group_weights = nn.Parameter(torch.ones(self.n_components))

    def _create_groups(self, H, threshold):
        """Create cell type groups based on correlation and known relationships"""
        # Calculate correlation between cell types
        corr_matrix = np.corrcoef(H)
        
        # Known biological groups (e.g., T cells)
        known_groups = [
            ['CD4-T-cells', 'CD8-T-cells']
        ]
        
        groups = []
        used_indices = set()
        
        # Add known groups
        for group in known_groups:
            group_indices = []
            for cell_type in group:
                if cell_type in self.cell_types:
                    idx = self.cell_types.index(cell_type)
                    group_indices.append(idx)
                    used_indices.add(idx)
            if len(group_indices) > 1:
                groups.append(group_indices)
        
        # Add correlation-based groups
        for i in range(self.n_components):
            if i in used_indices:
                continue
            
            correlations = corr_matrix[i]
            corr_indices = [j for j in range(self.n_components) 
                          if j not in used_indices and i != j 
                          and abs(correlations[j]) > threshold]
            
            if corr_indices:
                group = [i] + corr_indices
                groups.append(group)
                used_indices.update(group)
            else:
                groups.append([i])
                used_indices.add(i)
        
        return groups
    
    def _compute_training_loss(self, X, W_pred, W_true, coverage):
        """Compute loss during training when we have true W"""
        # Reconstruction loss
        X_pred = torch.mm(W_pred, self.H)
        mask = (coverage > 0).float()
        reconstruction_loss = torch.sum(mask * (X - X_pred)**2) / torch.sum(mask)
        
        # Supervised loss
        supervision_loss = F.mse_loss(W_pred, W_true)
        
        # Variance penalty with learned group weights
        group_variances = []
        for i, group in enumerate(self.groups):
            group_pred = W_pred[:, group]
            weight = F.softplus(self.group_weights[i])
            group_variance = torch.var(group_pred, dim=0).mean()
            group_variances.append(weight * group_variance)
        
        variance_loss = torch.mean(torch.stack(group_variances))
        
        return reconstruction_loss + supervision_loss + self.variance_penalty * variance_loss
    
    def _compute_prediction_loss(self, X, W_pred, coverage):
        """Compute loss during prediction (no W_true available)"""
        # Reconstruction loss
        X_pred = torch.mm(W_pred, self.H)
        mask = (coverage > 0).float()
        reconstruction_loss = torch.sum(mask * (X - X_pred)**2) / torch.sum(mask)
        
        # Apply learned group weights to variance
        group_variances = []
        for i, group in enumerate(self.groups):
            group_pred = W_pred[:, group]
            weight = F.softplus(self.group_weights[i])
            group_variance = torch.var(group_pred, dim=0).mean()
            group_variances.append(weight * group_variance)
        
        variance_loss = torch.mean(torch.stack(group_variances))
        
        return reconstruction_loss + self.variance_penalty * variance_loss
    
    def predict_batch(self, X, coverage):
        """Single batch prediction"""
        # Create the initial tensor as a Parameter
        W_init = nn.Parameter(torch.rand(X.shape[0], self.n_components))
        
        optimizer = torch.optim.Adam([W_init], lr=0.01)
        
        for _ in range(100):  # Max iterations for optimization
            optimizer.zero_grad()
            
            # Apply normalization for loss computation
            W_normalized = F.normalize(F.relu(W_init), p=1, dim=1)
            loss = self._compute_prediction_loss(X, W_normalized, coverage)
            
            loss.backward()
            optimizer.step()
        
        # Final normalization
        with torch.no_grad():
            W_final = F.normalize(F.relu(W_init), p=1, dim=1)
        
        return W_final


    def fit(self, X_train, W_train, coverage_train, 
        X_val, W_val, coverage_val,
        epochs=100, batch_size=32, learning_rate=0.001,
        patience=10):
        """
        Train the model with validation monitoring
        
        Args:
            X_train, W_train, coverage_train: Training data
            X_val, W_val, coverage_val: Validation data
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            patience: Number of epochs to wait for improvement before early stopping
        """
        # Convert all inputs to tensors
        X_train, W_train, coverage_train = map(torch.FloatTensor, [X_train, W_train, coverage_train])
        X_val, W_val, coverage_val = map(torch.FloatTensor, [X_val, W_val, coverage_val])
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"Starting training with {n_samples} training samples and {X_val.shape[0]} validation samples")
        print(f"Batch size: {batch_size}, Number of batches per epoch: {n_batches}")
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in trange(epochs, desc="Training epochs"):
            # Training phase
            self.train()
            train_losses = {'total': 0, 'rec': 0, 'sup': 0, 'var': 0}
            
            batch_iterator = trange(0, n_samples, batch_size, 
                                desc=f"Epoch {epoch}", 
                                leave=False)
            
            for i in batch_iterator:
                batch_indices = torch.randperm(n_samples)[:batch_size]
                batch_X = X_train[batch_indices]
                batch_W = W_train[batch_indices]
                batch_coverage = coverage_train[batch_indices]
                
                optimizer.zero_grad()
                W_pred = self.predict_batch(batch_X, batch_coverage)
                loss, rec_loss, sup_loss, var_loss = self._compute_training_loss(
                    batch_X, W_pred, batch_W, batch_coverage
                )
                
                loss.backward()
                optimizer.step()
                
                # Accumulate losses
                train_losses['total'] += loss.item()
                train_losses['rec'] += rec_loss.item()
                train_losses['sup'] += sup_loss.item()
                train_losses['var'] += var_loss.item()
                
                batch_iterator.set_postfix({
                    'train_loss': f"{loss.item():.6f}",
                    'rec_loss': f"{rec_loss.item():.6f}",
                    'sup_loss': f"{sup_loss.item():.6f}",
                    'var_loss': f"{var_loss.item():.6f}"
                })
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_W_pred = self.predict_batch(X_val, coverage_val)
                val_loss, val_rec_loss, val_sup_loss, val_var_loss = self._compute_training_loss(
                    X_val, val_W_pred, W_val, coverage_val
                )
            
            # Average losses
            train_losses = {k: v/n_batches for k, v in train_losses.items()}
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Training - Total: {train_losses['total']:.6f}, "
                f"Rec: {train_losses['rec']:.6f}, "
                f"Sup: {train_losses['sup']:.6f}, "
                f"Var: {train_losses['var']:.6f}")
            print(f"  Validation - Total: {val_loss:.6f}, "
                f"Rec: {val_rec_loss:.6f}, "
                f"Sup: {val_sup_loss:.6f}, "
                f"Var: {val_var_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    self.load_state_dict(best_model_state)
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        
        return train_losses, {'total': val_loss.item(), 
                            'rec': val_rec_loss.item(), 
                            'sup': val_sup_loss.item(), 
                            'var': val_var_loss.item()}


    def predict(self, X, coverage, batch_size=32):
        """
        Predict cell type proportions for new data
        """
        X = torch.FloatTensor(X)
        coverage = torch.FloatTensor(coverage)
        
        predictions = []
        batch_iterator = tqdm(range(0, X.shape[0], batch_size), 
                            desc="Predicting", 
                            total=(X.shape[0] + batch_size - 1) // batch_size)
        
        for i in batch_iterator:
            batch_X = X[i:i+batch_size]
            batch_coverage = coverage[i:i+batch_size]
            
            with torch.no_grad():
                W_pred = self.predict_batch(batch_X, batch_coverage)
                predictions.append(W_pred)
                
            batch_iterator.set_postfix({'batch_size': len(batch_X)})
        
        return torch.cat(predictions).numpy()

    def _compute_training_loss(self, X, W_pred, W_true, coverage):
        """Compute balanced training losses"""
        # Reconstruction loss - normalize by number of valid observations
        X_pred = torch.mm(W_pred, self.H)
        mask = (coverage > 0).float()
        reconstruction_loss = torch.sum(mask * (X - X_pred)**2) / torch.sum(mask)
        
        # Supervised loss - normalize similarly to reconstruction loss
        supervision_loss = torch.sum((W_pred - W_true)**2) / (W_true.shape[0] * W_true.shape[1])
        
        # Variance penalty
        group_variances = []
        for i, group in enumerate(self.groups):
            group_pred = W_pred[:, group]
            weight = F.softplus(self.group_weights[i])
            # Normalize variance by group size
            group_variance = torch.var(group_pred, dim=0).mean() / len(group)
            group_variances.append(weight * group_variance)
        
        variance_loss = torch.mean(torch.stack(group_variances))
        
        # All losses should now be on similar scales
        total_loss = reconstruction_loss + supervision_loss + self.variance_penalty * variance_loss
        
        return total_loss, reconstruction_loss, supervision_loss, variance_loss


def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment: Weighted NNLS optimisation with R² Metric for Cell Type Deconvolution")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to the altas")
    parser.add_argument("--training_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the save figures")
    parser.add_argument("--pats_path", type=str, required=True, help="Path to the pats dir")
    parser.add_argument("--wgbs_tools_exec_path",help="path to wgbs_tools executable",required=True)
    parser.add_argument("--cell_type",help="cell type to analyse",required=True)
    args = parser.parse_args()
    H, cell_types = process_atlas(args.atlas_path)
    deconvolver = RegularisedGroupedDeconvolver(
        reference_profiles=H,
        cell_types=cell_types,
        correlation_threshold=0.3,
        variance_penalty=0.1
    )
    
    # train
    data = load_dataset(args.training_path)
    deconvolver.fit(
        X_train=data['X_train'],
        W_train=data['y_train'],
        coverage_train=data['coverage_train'],
        X_val=data['X_val'],
        W_val=data['y_val'],
        coverage_val=data['coverage_val'],
        epochs=1000,
        batch_size=32
    )

    marker_read_proportions, counts = pats_to_homog(args.atlas_path,args.pats_path,args.wgbs_tools_exec_path) 
    marker_read_proportions = marker_read_proportions.drop(columns=['name','direction'])[columns()].T.to_numpy()
    counts = counts.drop(columns=['name','direction'])[columns()].T.to_numpy()
    
    results = deconvolver.predict(marker_read_proportions,counts)
    proportions = results['proportions']
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