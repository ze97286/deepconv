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
    """
    Simplified deconvolution model with direct coverage weighting
    """
    def __init__(self, num_markers, num_cell_types, target_ids, presence_models_dir=None, feature_dim=32):
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim

        # Store cell-type assignment for each marker
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)

        # Load separate presence models if provided
        self.presence_models = None
        if presence_models_dir:
            self.presence_models = nn.ModuleList()
            
            for cell_type_idx in range(num_cell_types):
                model_path = Path(presence_models_dir) / f"presence_model_{cell_type_idx}.pt"
                
                if not model_path.exists():
                    raise FileNotFoundError(f"Presence model not found at {model_path}")
                
                # Load the model
                checkpoint = torch.load(model_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    presence_model = SingleCellTypePresenceModel()
                    presence_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    presence_model = checkpoint
                
                presence_model.eval()
                self.presence_models.append(presence_model)

        # Simple marker to feature mapping
        self.marker_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Cell type decoder
        self.cell_type_decoder = nn.Sequential(
            nn.Linear(feature_dim * num_cell_types, 128),
            nn.ReLU(),
            nn.Linear(128, num_cell_types)
        )

    def forward(self, marker_values, coverage):
        """
        Forward pass with direct coverage weighting
        """
        B, M = marker_values.shape
        C = self.num_celltypes

        # Create valid mask for markers with coverage > 0
        valid_mask = (coverage > 0)
        
        # Handle cases with no valid data
        if not valid_mask.any():
            zeros = torch.zeros((B, C), device=marker_values.device)
            return zeros, zeros, valid_mask, zeros, zeros

        # Replace invalid values with zeros
        marker_values_safe = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))
        
        # Weight marker values by coverage (NNLS-style approach)
        # Scale by mean coverage to keep values in reasonable range
        mean_coverage = coverage.mean()
        coverage_weight = coverage / (mean_coverage + 1e-8)
        weighted_markers = marker_values_safe * coverage_weight
        
        # Process each marker to extract features
        marker_features = self.marker_encoder(weighted_markers.unsqueeze(-1))  # [B, M, feature_dim]
        
        # Aggregate features by cell type using target_ids
        cell_type_features = []
        for ct in range(self.num_celltypes):
            # Create mask for this cell type's markers
            ct_mask = (self.target_ids == ct).expand(B, -1)
            
            # Apply cell type mask and valid mask
            combined_mask = ct_mask & valid_mask
            
            # Skip cell types with no valid markers
            if not combined_mask.any():
                cell_type_features.append(torch.zeros(B, self.feature_dim, device=marker_values.device))
                continue
            
            # Get coverage for this cell type's markers
            ct_coverage = torch.where(combined_mask, coverage, torch.zeros_like(coverage))
            
            # Calculate coverage weights for normalization
            ct_coverage_sum = ct_coverage.sum(dim=1, keepdim=True) + 1e-8
            ct_coverage_weights = ct_coverage.unsqueeze(-1) / ct_coverage_sum.unsqueeze(-1)
            
            # Use coverage-weighted average of features for this cell type
            ct_weighted_features = marker_features * ct_coverage_weights
            ct_features = ct_weighted_features.sum(dim=1)  # [B, feature_dim]
            
            cell_type_features.append(ct_features)
        
        # Concatenate all cell type features
        combined_features = torch.cat(cell_type_features, dim=1)  # [B, C*feature_dim]
        
        # Predict cell type proportions
        cell_props_raw = F.relu(self.cell_type_decoder(combined_features))
        
        # Get presence probabilities if presence models are available
        presence_probs = torch.ones((B, C), device=marker_values.device)
        presence_logits = torch.zeros((B, C), device=marker_values.device)
        
        if self.presence_models:
            for cell_type_idx, presence_model in enumerate(self.presence_models):
                with torch.no_grad():
                    # Get markers for this cell type
                    ct_mask = (self.target_ids == cell_type_idx)
                    ct_marker_values = marker_values[:, ct_mask]
                    ct_coverage = coverage[:, ct_mask]
                    
                    # Skip if no markers for this cell type
                    if ct_marker_values.shape[1] == 0:
                        continue
                    
                    # Get presence logits and probabilities
                    logits, _ = presence_model(ct_marker_values, ct_coverage)
                    probs = torch.sigmoid(logits)
                    
                    # Store results
                    presence_logits[:, cell_type_idx] = logits.squeeze(-1)
                    presence_probs[:, cell_type_idx] = probs.squeeze(-1)
        
        # Apply presence gating
        cell_props_gated = cell_props_raw * presence_probs
        
        # Normalize to sum to 1
        sum_props = torch.sum(cell_props_gated, dim=1, keepdim=True) + 1e-8
        cell_props = cell_props_gated / sum_props
        
        # Predict marker values from cell type proportions (reconstruction)
        # This is a simplified reconstruction that doesn't rely on a learned decoder
        reconstructed = torch.zeros_like(marker_values)
        
        # For each marker, use the corresponding cell type's contribution
        for m in range(M):
            cell_type = self.target_ids[m].item()
            reconstructed[:, m] = cell_props[:, cell_type]
        
        return cell_props, reconstructed, valid_mask, presence_probs, presence_logits

    def predict(self, marker_values, coverage, batch_size=256, device=None):
        """
        Makes predictions using the model in evaluation mode.
        
        Args:
            marker_values: Marker methylation values [N, M]
            coverage: Coverage values [N, M]
            batch_size: Batch size for processing
            device: Device to run inference on (defaults to model's device)
            
        Returns:
            numpy.ndarray: Cell type proportions [N, C]
        """
        import numpy as np
        
        # Decide which device to use (CPU/GPU)
        if device is None:
            device = next(self.parameters()).device
        
        # Convert inputs (X, coverage) to Torch tensors if needed
        if not isinstance(marker_values, torch.Tensor):
            marker_values = torch.tensor(marker_values, dtype=torch.float32)
        if not isinstance(coverage, torch.Tensor):
            coverage = torch.tensor(coverage, dtype=torch.float32)
        
        # Ensure both inputs have a batch dimension
        if len(marker_values.shape) == 1:
            marker_values = marker_values.unsqueeze(0)
        if len(coverage.shape) == 1:
            coverage = coverage.unsqueeze(0)
        
        self.eval()
        predictions_list = []
        
        # Process the data in batches
        num_samples = marker_values.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                batch_X = marker_values[start_idx:end_idx].to(device)
                batch_coverage = coverage[start_idx:end_idx].to(device)
                
                # Forward pass through the model - use only the proportions result
                props, *_ = self.forward(batch_X, batch_coverage)
                
                # Move to CPU numpy and store
                predictions_list.append(props.cpu().numpy())
                
                # Optional GPU memory cleanup
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Combine all batch results
        if len(predictions_list) == 0:
            # Edge case: empty input
            return np.zeros((num_samples, self.num_celltypes))
        
        # Return a single array of shape [N, C]
        return np.vstack(predictions_list)