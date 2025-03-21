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
                 feature_dim=32, coverage_threshold=5.0, reliability_alpha=0.7):
        """
        Efficient cell type deconvolution model with coverage-based marker reliability.
        
        Args:
            num_markers: Total number of markers in the atlas
            num_cell_types: Number of cell types to predict
            target_ids: Marker to cell type mapping array
            presence_models_dir: Directory containing pre-trained presence models
            feature_dim: Feature dimension for marker encoding
            coverage_threshold: Threshold for considering a marker reliable
            reliability_alpha: Weight parameter for reliability calculation (higher = more emphasis on coverage)
        """
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim
        self.coverage_threshold = coverage_threshold
        self.reliability_alpha = reliability_alpha

        # Store cell-type assignment for each marker
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)
        
        # Calculate marker informativeness based on uniqueness and specificity
        # Can be learned during training or preset based on atlas information
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

        # Cell type encoder with coverage-awareness
        self.celltype_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LeakyReLU(),
                nn.Linear(feature_dim, feature_dim)
            ) for _ in range(num_cell_types)
        ])

        # Cell type decoder with presence information
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim * num_cell_types + num_cell_types, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_cell_types)
        )
        
        # Reconstruction decoder (optional)
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

    def calculate_marker_reliability(self, coverage):
        """
        Calculate reliability scores for markers based on coverage.
        
        Args:
            coverage: [B, M] coverage values
            
        Returns:
            reliability: [B, M] reliability scores between 0 and 1
        """
        # Base reliability is a sigmoid function of coverage
        # This creates a smooth transition around the threshold
        reliability = torch.sigmoid((coverage - self.coverage_threshold) / 2.0)
        
        # Apply marker informativeness
        reliability = reliability * self.marker_informativeness.unsqueeze(0)
        
        # Apply coverage-dependent scaling
        # This makes the reliability more sensitive to coverage differences at lower values
        coverage_scale = 1.0 - torch.exp(-0.1 * coverage)
        reliability = reliability * (self.reliability_alpha + (1.0 - self.reliability_alpha) * coverage_scale)
        
        return reliability

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
                
                # Skip if all coverage is zero
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
        Forward pass with coverage-aware marker reliability.
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
        
        # Calculate reliability scores for all markers based on coverage
        reliability = self.calculate_marker_reliability(coverage)
        
        # Extract features from all markers
        marker_features = self.marker_encoder(marker_values_safe.unsqueeze(-1))  # [B, M, feature_dim]
        
        # For each cell type, extract features from its markers with reliability weighting
        cell_type_features = []
        
        for ct_idx in range(C):
            # Get mask for this cell type's markers
            ct_mask = (self.target_ids == ct_idx)
            
            # Skip if no markers for this cell type
            if not ct_mask.any():
                cell_type_features.append(torch.zeros(B, self.feature_dim, device=marker_values.device))
                continue
            
            # Get only this cell type's markers
            ct_features = marker_features[:, ct_mask]  # [B, M_ct, feature_dim]
            ct_reliability = reliability[:, ct_mask].unsqueeze(-1)  # [B, M_ct, 1]
            ct_valid = valid_mask[:, ct_mask].unsqueeze(-1)  # [B, M_ct, 1]
            
            # Apply reliability weighting to features
            weighted_features = ct_features * ct_reliability * ct_valid.float()
            
            # Calculate normalizing factor (sum of weights)
            normalizer = torch.sum(ct_reliability * ct_valid.float(), dim=1, keepdim=True) + 1e-8
            
            # Aggregate features with weighted average
            aggregated_features = torch.sum(weighted_features, dim=1) / normalizer.squeeze(-1)
            
            # Apply cell type specific encoding
            encoded_features = self.celltype_encoder[ct_idx](aggregated_features)
            
            cell_type_features.append(encoded_features)
        
        # Combine features from all cell types
        combined_features = torch.cat(cell_type_features, dim=1)
        
        # Get presence probabilities
        presence_probs, presence_logits = self.get_presence_probs(marker_values, coverage)
        
        # Combine with presence information
        decoder_input = torch.cat([combined_features, presence_probs], dim=1)
        
        # Predict cell type proportions
        cell_props_raw = F.relu(self.decoder(decoder_input))
        
        # Apply presence gating
        cell_props_gated = cell_props_raw * presence_probs
        
        # Normalize to sum to 1
        sum_props = torch.sum(cell_props_gated, dim=1, keepdim=True) + 1e-8
        cell_props = cell_props_gated / sum_props
        
        # Reconstruct marker values (optional)
        reconstructed = self.reconstructor(cell_props)
        
        return cell_props, reconstructed, valid_mask, presence_probs, presence_logits, reliability
    
