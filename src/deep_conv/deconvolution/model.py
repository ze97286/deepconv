import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DeepConvDataset(Dataset):
    """Dataset for cell type deconvolution data with improved weighting for low proportions."""
    
    def __init__(self, methylation_values: np.ndarray, coverage: np.ndarray, true_cell_type_proportions: np.ndarray):
        """
        Initializes the dataset with methylation values (proportions of reads matching marker), coverage, and true cell proportions.
        Computes sample weights emphasizing low-proportion samples.
        
        Args:
            methylation_values (np.ndarray): Methylation data (n_samples x n_regions).
            coverage (np.ndarray): Coverage data (n_samples x n_regions).
            true_cell_type_proportions (np.ndarray): Cell type proportions (n_samples x n_cell_types).
        """
        self.methylation = torch.tensor(methylation_values, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.true_cell_type_proportions = torch.tensor(true_cell_type_proportions, dtype=torch.float32)
        
        # Compute weights with stronger emphasis on low proportions
        if isinstance(true_cell_type_proportions, np.ndarray):
            min_props = true_cell_type_proportions.min(axis=1)
        else:
            min_props = torch.min(true_cell_type_proportions, dim=1)[0].cpu().numpy()
        
        # Weighting scheme to give even more importance to very low proportions
        self.weights = 1.0 / (min_props + 1e-6)**0.5  
        
        # Cap the weights to prevent extremely large values
        max_weight = 10.0
        self.weights = np.clip(self.weights, 0, max_weight)
        
        # Normalise weights to have a mean of 1
        self.weights = self.weights / self.weights.mean()
        
        # Convert weights to PyTorch tensor
        self.weights = torch.tensor(self.weights, dtype=torch.float32)
                
    def __len__(self) -> int:
        return len(self.methylation)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'methylation': self.methylation[idx],
            'coverage': self.coverage[idx],
            'proportions': self.true_cell_type_proportions[idx],
            'weight': self.weights[idx]
        }
   
    
class DiffNNLS(nn.Module):
    def __init__(self, n_markers, n_cell_types, max_iter=100):
        super().__init__()
        self.n_markers = n_markers
        self.n_cell_types = n_cell_types
        self.max_iter = max_iter
        
        # NNLS parameters
        self.step_sizes = nn.Parameter(0.1 * torch.ones(max_iter))
        
        # Learn corrections to reference profiles
        self.profile_correction = nn.Parameter(torch.zeros(n_cell_types, n_markers))
        
        # Learn input transformations
        self.input_transform = nn.Sequential(
            nn.Linear(n_markers, n_markers),
            nn.ReLU(),
            nn.Linear(n_markers, n_markers)
        )
        
    def forward(self, marker_props, coverage):
        batch_size = marker_props.shape[0]
        
        # Transform inputs
        transformed_markers = self.input_transform(marker_props)
        weighted_markers = transformed_markers * coverage
        
        # Use corrected reference profiles
        corrected_profiles = self.reference_profiles + F.sigmoid(self.profile_correction)
        
        x = torch.zeros(batch_size, self.n_cell_types, device=marker_props.device)
        
        for i in range(self.max_iter):
            current_recon = torch.matmul(x, corrected_profiles) * coverage
            grad = current_recon - weighted_markers
            grad = torch.matmul(grad, corrected_profiles.t())
            
            x = F.relu(x - self.step_sizes[i] * grad)
            x = x / (x.sum(dim=1, keepdim=True) + 1e-8)
        
        reconstruction = torch.matmul(x, self.reference_profiles) * coverage
        return x, torch.zeros_like(x), reconstruction


class DiffNNLSLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_props, _, true_props, weighted_markers, reconstruction, weight=None):
        # Just minimize reconstruction error like NNLS
        recon_loss = F.mse_loss(reconstruction, weighted_markers)
        
        return recon_loss, {'recon_loss': recon_loss.item()}
    def __init__(self, lambda_sparse=0.01):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        
    def forward(self, pred_props, _, true_props, weighted_markers, reconstruction, weight=None):
        # Main reconstruction loss
        recon_loss = F.mse_loss(reconstruction, weighted_markers)
        
        # Sparsity penalty
        sparsity_loss = self.lambda_sparse * torch.mean(torch.abs(pred_props))
        
        # Combine losses
        total_loss = recon_loss + sparsity_loss
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
        }