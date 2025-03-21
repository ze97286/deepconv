import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os 
import pandas as pd
from pathlib import Path
from deep_conv.presence.model import SingleCellTypePresenceModel


class TissueDeconvolutionDataset(Dataset):
    """
    A PyTorch Dataset for loading cfDNA methylation data and optional labels.
    
    Each sample in this dataset includes:
      - `fraction`: Methylation fractions across markers, in [0..1] (may contain NaNs if coverage=0).
      - `coverage`: Read coverage array of the same shape as `fraction`.
      - `atlas`: (Optional) if using some reference atlas or additional data, 
                 you could store it here. (Currently not directly used in the model.)
      - `y`: Ground-truth cell-type proportions for training/validation, if available.
      
    Args:
        fraction (ndarray or Tensor): Shape [num_samples, num_markers].
            Fractional methylation values. Some entries may be invalid if coverage=0.
        coverage (ndarray or Tensor): Shape [num_samples, num_markers].
            Coverage (read depth) for each sample-marker pair.
        atlas (ndarray or Tensor): Arbitrary shape, often referencing 
            a reference atlas. Not necessarily used in the model code, 
            but stored for convenience.
        y (ndarray or Tensor, optional): Shape [num_samples, num_cell_types].
            Ground-truth proportions for each cell type (if supervised).
            If None, dataset is for inference only.
    """
    def __init__(self, fraction, coverage, atlas, y=None):
        self.fraction = torch.tensor(fraction, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.atlas = torch.tensor(atlas, dtype=torch.float32)        
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return self.fraction.size(0)

    def __getitem__(self, idx):
        """
        Return a dictionary containing:
            'X': The methylation fraction row for this sample
            'coverage': The coverage row for this sample
            'y': The ground-truth proportions, if available
        """
        item = {
            'X': self.fraction[idx],
            'coverage': self.coverage[idx],
        }
        if self.y is not None:
            item['y'] = self.y[idx]
        return item  


class CellTypeDeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types, target_ids, presence_models_dir=None, 
                 feature_dim=32, top_k_fraction=0.5, min_markers_per_celltype=5):
        """
        Cell type deconvolution model with dynamic marker selection based on coverage.
        Key Features of This Approach:

        * Dynamic Marker Selection: For each sample, the model selects the most reliable markers for each cell type based on coverage. This ensures that low-coverage markers don't contaminate the signal.
        * Adaptive Feature Weighting: Features are weighted by their reliability scores, giving more influence to higher-coverage markers.
        * Flexible Selection Criteria: The top_k_fraction parameter controls how many markers to select, and min_markers_per_celltype ensures a minimum number of markers are always used.
        * Log-Space Concentration Error: Optionally uses log-space for concentration errors, which better handles the wide range of concentrations (especially at the low end).
        * Coverage-Stratified Monitoring: Tracks performance separately for different coverage levels to better understand where improvements are happening.

        Args:
            num_markers: Total number of markers in the atlas
            num_cell_types: Number of cell types to predict
            target_ids: Marker to cell type mapping array
            presence_models_dir: Directory containing pre-trained presence models
            feature_dim: Feature dimension for marker encoding
            top_k_fraction: Fraction of available markers to select for each cell type
            min_markers_per_celltype: Minimum markers to use per cell type regardless of coverage
        """
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim
        self.top_k_fraction = top_k_fraction
        self.min_markers_per_celltype = min_markers_per_celltype

        # Store cell-type assignment for each marker
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)
        
        # Calculate marker informativeness (can be learned or initialized heuristically)
        marker_informativeness = torch.ones(num_markers)
        self.register_buffer("marker_informativeness", marker_informativeness)

        # Load presence models if provided
        self.presence_models = self._load_presence_models(presence_models_dir) if presence_models_dir else None

        # Marker feature extractor
        self.marker_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Cell type decoder with presence information
        self.cell_type_decoder = nn.Sequential(
            nn.Linear(feature_dim * num_cell_types + num_cell_types, 128),
            nn.ReLU(),
            nn.Linear(128, num_cell_types)
        )
        
        # Reconstruction decoder
        self.reconstructor = nn.Sequential(
            nn.Linear(num_cell_types, 64),
            nn.ReLU(),
            nn.Linear(64, num_markers)
        )

    def _load_presence_models(self, presence_models_dir):
        """Load pre-trained presence detection models"""
        presence_models = nn.ModuleList()
        
        for cell_type_idx in range(self.num_celltypes):
            model_path = Path(presence_models_dir) / f"presence_model_{cell_type_idx}.pt"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Presence model not found at {model_path}")
            
            # Load the model
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                from deep_conv.presence.model import SingleCellTypePresenceModel
                presence_model = SingleCellTypePresenceModel()
                presence_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                presence_model = checkpoint
            
            presence_model.eval()
            presence_models.append(presence_model)
            
        return presence_models

    def select_reliable_markers(self, coverage, marker_values=None):
        """
        Dynamically select the most reliable markers based on coverage.
        
        Args:
            coverage: [B, M] coverage values for each sample and marker
            marker_values: Optional [B, M] methylation values (can be used for additional selection criteria)
            
        Returns:
            selected_marker_masks: Dictionary mapping cell type index to boolean mask of selected markers
            reliability_scores: [B, M] Reliability score for each marker in each sample
        """
        B, M = coverage.shape
        
        # Base reliability score is simply the coverage
        reliability_scores = coverage.clone()
        
        # Apply marker informativeness as a multiplier
        # This can represent prior knowledge about which markers are most discriminative
        reliability_scores = reliability_scores * self.marker_informativeness.unsqueeze(0)
        
        # Create masks for each cell type's markers
        selected_marker_masks = {}
        
        for cell_type_idx in range(self.num_celltypes):
            # Get mask for this cell type's markers
            cell_type_mask = (self.target_ids == cell_type_idx)
            
            # Skip if no markers for this cell type
            if not cell_type_mask.any():
                selected_marker_masks[cell_type_idx] = torch.zeros(B, M, dtype=torch.bool, device=coverage.device)
                continue
            
            # For each sample, select top-k markers for this cell type
            # We do this sample by sample since coverage varies per sample
            sample_masks = []
            
            for b in range(B):
                # Get reliability scores for this cell type's markers in this sample
                scores = reliability_scores[b, cell_type_mask]
                
                # Calculate how many markers to select
                available_markers = scores.size(0)
                k = max(int(available_markers * self.top_k_fraction), self.min_markers_per_celltype)
                k = min(k, available_markers)  # Can't select more than available
                
                # Select top-k markers
                if k > 0:
                    _, top_indices = torch.topk(scores, k)
                    
                    # Create mask for selected markers
                    marker_indices = torch.arange(M, device=coverage.device)
                    cell_type_indices = marker_indices[cell_type_mask]
                    selected_indices = cell_type_indices[top_indices]
                    
                    sample_mask = torch.zeros(M, dtype=torch.bool, device=coverage.device)
                    sample_mask[selected_indices] = True
                else:
                    sample_mask = torch.zeros(M, dtype=torch.bool, device=coverage.device)
                
                sample_masks.append(sample_mask)
            
            # Combine masks for all samples
            selected_marker_masks[cell_type_idx] = torch.stack(sample_masks)
        
        return selected_marker_masks, reliability_scores

    def get_presence_probs(self, marker_values, coverage):
        """Get cell type presence probabilities using pre-trained models"""
        B = marker_values.shape[0]
        
        # Initialize output tensors
        presence_probs = torch.ones((B, self.num_celltypes), device=marker_values.device)
        presence_logits = torch.zeros((B, self.num_celltypes), device=marker_values.device)
        
        if self.presence_models is None:
            return presence_probs, presence_logits
        
        # For each cell type, use its dedicated presence model
        for cell_type_idx, presence_model in enumerate(self.presence_models):
            with torch.no_grad():
                # Get markers for this cell type
                cell_type_mask = (self.target_ids == cell_type_idx)
                
                # Skip if no markers for this cell type
                if not cell_type_mask.any():
                    continue
                
                # Get marker values and coverage for this cell type
                ct_marker_values = marker_values[:, cell_type_mask]
                ct_coverage = coverage[:, cell_type_mask]
                
                # Skip if no valid markers (all coverage=0)
                if (ct_coverage > 0).sum() == 0:
                    continue
                
                # Get presence probabilities
                logits, _ = presence_model(ct_marker_values, ct_coverage)
                probs = torch.sigmoid(logits)
                
                # Store results
                presence_logits[:, cell_type_idx] = logits.squeeze(-1)
                presence_probs[:, cell_type_idx] = probs.squeeze(-1)
        
        return presence_probs, presence_logits

    def forward(self, marker_values, coverage):
        """
        Forward pass with dynamic marker selection based on coverage.
        """
        B, M = marker_values.shape
        C = self.num_celltypes
        
        # Create valid mask for markers with coverage > 0
        valid_mask = (coverage > 0)
        
        # Handle case with no valid markers
        if not valid_mask.any():
            zeros = torch.zeros((B, C), device=marker_values.device)
            return zeros, zeros, valid_mask, zeros, zeros
        
        # Replace invalid values with zeros
        marker_values_safe = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))
        
        # Step 1: Dynamically select the most reliable markers for each cell type
        selected_markers, reliability_scores = self.select_reliable_markers(coverage, marker_values_safe)
        
        # Step 2: Extract features from all markers
        marker_features = self.marker_encoder(marker_values_safe.unsqueeze(-1))
        
        # Step 3: Aggregate features by cell type using only selected markers
        cell_type_features = []
        
        for cell_type_idx in range(C):
            # Get mask for selected markers for this cell type
            marker_mask = selected_markers[cell_type_idx]
            
            # Skip if no markers selected (shouldn't happen with min_markers_per_celltype)
            if not marker_mask.any():
                cell_type_features.append(torch.zeros(B, self.feature_dim, device=marker_values.device))
                continue
            
            # Get features for selected markers
            batch_features = []
            
            for b in range(B):
                # Get features and reliability scores for selected markers
                sample_mask = marker_mask[b]
                sample_features = marker_features[b, sample_mask]
                sample_weights = reliability_scores[b, sample_mask].unsqueeze(-1)
                
                # Skip if no markers selected for this sample
                if not sample_mask.any():
                    batch_features.append(torch.zeros(self.feature_dim, device=marker_values.device))
                    continue
                
                # Weight features by reliability and aggregate
                weighted_features = sample_features * sample_weights
                aggregated = weighted_features.sum(dim=0) / (sample_weights.sum() + 1e-8)
                
                batch_features.append(aggregated)
            
            # Stack features for all samples
            cell_type_features.append(torch.stack(batch_features))
        
        # Combine features for all cell types
        combined_features = torch.cat(cell_type_features, dim=1)
        
        # Step 4: Get presence probabilities
        presence_probs, presence_logits = self.get_presence_probs(marker_values, coverage)
        
        # Step 5: Combine features with presence information
        decoder_input = torch.cat([combined_features, presence_probs], dim=1)
        
        # Step 6: Predict cell type proportions
        cell_props_raw = F.relu(self.cell_type_decoder(decoder_input))
        
        # Step 7: Apply presence gating
        cell_props_gated = cell_props_raw * presence_probs
        
        # Step 8: Normalize to sum to 1
        sum_props = torch.sum(cell_props_gated, dim=1, keepdim=True) + 1e-8
        cell_props = cell_props_gated / sum_props
        
        # Step 9: Reconstruct marker values
        reconstructed = self.reconstructor(cell_props)
        
        return cell_props, reconstructed, valid_mask, presence_probs, presence_logits
    
