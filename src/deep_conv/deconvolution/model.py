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
    A neural network for predicting cell-type proportions from cfDNA methylation data.
    
    This version integrates pre-trained SingleCellTypePresenceModel instances as features
    rather than using them as binary gates.
    """
    def __init__(self, num_markers, num_cell_types, target_ids, presence_models_dir, feature_dim=32):
        """
        Initialize the cell type deconvolution model with separate presence models.
        
        Args:
            num_markers (int): Total number of markers (M).
            num_cell_types (int): Number of cell types (C).
            target_ids (array-like): Mapping of each marker to its target cell type index.
            presence_models_dir (str): Directory containing pre-trained presence models.
            feature_dim (int): Dimensionality of the marker feature space.
        """
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim

        # Store cell-type assignment for each marker (not trainable, but placed on same device)
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)

        # Load separate presence models
        self.presence_models = nn.ModuleList()
        
        for cell_type_idx in range(num_cell_types):
            model_path = Path(presence_models_dir) / f"presence_model_{cell_type_idx}.pt"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Presence model not found at {model_path}")
            
            # Load the model (handling different saving formats)
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                from deep_conv.presence.model import SingleCellTypePresenceModel
                presence_model = SingleCellTypePresenceModel()
                presence_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                presence_model = checkpoint  # Direct model object
            
            presence_model.eval()  # Set to evaluation mode
            self.presence_models.append(presence_model)

        # ----- Marker Feature Extractor -----
        # Transforms each (scalar) methylation value into a learned feature space
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # ----- Encoder (Proportion Prediction) -----
        # Updated to take both aggregated features AND presence probabilities
        self.encoder = nn.Sequential(
            nn.Linear(num_cell_types * feature_dim + num_cell_types, 128),  # +C for presence probs
            nn.LeakyReLU(),
            nn.Linear(128, 128),  # Additional layer for more expressive capacity
            nn.LeakyReLU(),
            nn.Linear(128, num_cell_types)
        )

        # ----- Decoder (Marker Reconstruction) -----
        self.decoder = nn.Sequential(
            nn.Linear(num_cell_types, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_markers)
        )

    def apply_presence_gating(self, props, probs):
        """
        Apply cell-type specific presence scaling based on empirical data.
        """
        # Cell-type specific thresholds and slopes
        if not hasattr(self, "presence_thresholds"):
            thresholds = torch.ones(self.num_celltypes, device=props.device) * 0.5
            slopes = torch.ones(self.num_celltypes, device=props.device) * 10
            
            # Special handling for OAC based on actual data
            oac_index = 9  # Adjust to actual OAC index
            thresholds[oac_index] = 0.3  # Center sigmoid at 0.3 for OAC
            slopes[oac_index] = 15       # Steeper slope for OAC
            
            self.register_buffer("presence_thresholds", thresholds)
            self.register_buffer("presence_slopes", slopes)
        
        # Apply scaling - vector operation across all cell types at once
        scaling = torch.sigmoid(
            self.presence_slopes.unsqueeze(0) * (probs - self.presence_thresholds.unsqueeze(0))
        )
        
        scaled_props = props * scaling
        
        # Normalize to ensure sum to 1
        sum_props = torch.sum(scaled_props, dim=1, keepdim=True) + 1e-8
        gated_props = scaled_props / sum_props
        
        return gated_props

    def predict_presence_with_separate_models(self, marker_values, coverage):
        """
        Use the separate pre-trained presence models to predict 
        presence probabilities for each cell type.
        
        Each presence model receives only the markers that correspond to its cell type.
        
        Args:
            marker_values (FloatTensor): [B, M], fractional methylation
            coverage (FloatTensor): [B, M], read coverage
            
        Returns:
            presence_probs (FloatTensor): [B, C], presence probability for each cell type
            presence_logits (FloatTensor): [B, C], raw logits before sigmoid
        """
        B = marker_values.shape[0]
        C = self.num_celltypes
        
        # Initialize output tensors
        presence_probs = torch.zeros(B, C, device=marker_values.device)
        presence_logits = torch.zeros(B, C, device=marker_values.device)
        
        # For each cell type, use its dedicated presence model
        for cell_type_idx, presence_model in enumerate(self.presence_models):
            with torch.no_grad():  # No gradients needed for frozen presence models
                # Create a mask for the markers that belong to this cell type
                cell_type_marker_mask = (self.target_ids == cell_type_idx)
                
                # Skip if no markers for this cell type
                if not cell_type_marker_mask.any():
                    continue
                
                # Filter marker_values and coverage to only include markers for this cell type
                cell_type_marker_values = marker_values[:, cell_type_marker_mask]
                cell_type_coverage = coverage[:, cell_type_marker_mask]
                
                # Pass only the relevant markers to the presence model
                logits, _ = presence_model(cell_type_marker_values, cell_type_coverage)
                probs = torch.sigmoid(logits)
                
                # Store results
                presence_logits[:, cell_type_idx] = logits.squeeze(-1)
                presence_probs[:, cell_type_idx] = probs.squeeze(-1)
                
        return presence_probs, presence_logits

    def forward(self, marker_values: torch.Tensor, coverage: torch.Tensor):
        """
        Forward pass to predict cell-type proportions from methylation + coverage.

        Steps:
            1) Identify valid markers (coverage>0).
            2) Extract features for each valid marker via `marker_feature_extractor`.
            3) Aggregate marker features per cell type, weighting by coverage.
            4) Predict presence_prob for each cell type using separate models.
            5) Combine aggregated features with presence information for proportion prediction.
            6) Apply soft presence-informed scaling and normalize.
            7) Reconstruct marker methylation from the final proportions.

        Args:
            marker_values (FloatTensor): [B, M], fractional methylation (NaN if coverage=0).
            coverage (FloatTensor): [B, M], read coverage.

        Returns:
            celltype_props (FloatTensor): [B, C], predicted proportion for each cell type.
            reconstructed (FloatTensor): [B, M], the model's reconstruction of marker methylation.
            valid_mask (BoolTensor): [B, M], True where coverage>0.
            presence_probs (FloatTensor): [B, C], presence probability for each cell type.
            presence_logits (FloatTensor): [B, C], raw logits before sigmoid.
        """
        B, M = marker_values.shape
        C = self.num_celltypes

        # valid_mask indicates coverage>0
        valid_mask = (coverage > 0)

        # Flatten coverage & marker_values for efficient indexing
        coverage_flat = coverage.view(-1)
        marker_values_flat = marker_values.view(-1)

        # Indices of non-zero coverage
        valid_inds = torch.nonzero(coverage_flat, as_tuple=False).squeeze(1)

        # Handle all-zero-coverage case
        if valid_inds.numel() == 0:
            # Provide a fallback (assign 1.0 to the first cell type, 0 to others)
            celltype_props = coverage.new_zeros(B, C)
            celltype_props[:, 0] = 1.0
            reconstructed = coverage.new_zeros(B, M)
            presence_probs = coverage.new_zeros(B, C)
            presence_logits = coverage.new_zeros(B, C)
            return celltype_props, reconstructed, valid_mask, presence_probs, presence_logits

        # Extract coverage & marker_values for valid coverage
        coverage_valid = coverage_flat[valid_inds]
        marker_values_valid = marker_values_flat[valid_inds]

        # Compute the batch index and marker index from flattened indices
        batch_idx = valid_inds // M
        marker_idx = valid_inds % M

        # Each marker is known to correspond to a specific cell type (via self.target_ids)
        celltype_idx = self.target_ids[marker_idx]

        # ----- 1) Marker Feature Extraction -----
        marker_values_valid_2d = marker_values_valid.unsqueeze(1)  # [N, 1]
        features_valid = self.marker_feature_extractor(marker_values_valid_2d)  # [N, feature_dim]

        # ----- 2) Aggregate features by cell type -----
        # aggregator shape: [B, C, feature_dim], coverage_sum shape: [B, C]
        aggregator = coverage.new_zeros(B, C, self.feature_dim)
        coverage_sum = coverage.new_zeros(B, C)

        # We'll do index_add_ on a flattened [B*C, feature_dim]
        aggregator_2d = aggregator.view(B*C, self.feature_dim)
        coverage_sum_1d = coverage_sum.view(B*C)

        # Flatten to [N], so bc_index is each valid coverage row's (batch, celltype)
        bc_index = batch_idx * C + celltype_idx
        weighted_feats = coverage_valid.unsqueeze(1) * features_valid  # shape [N, feature_dim]

        # Scatter-add
        aggregator_2d.index_add_(0, bc_index, weighted_feats)
        coverage_sum_1d.index_add_(0, bc_index, coverage_valid)

        # Reshape back
        aggregator = aggregator_2d.view(B, C, self.feature_dim)
        coverage_sum = coverage_sum_1d.view(B, C)

        # Avoid divide-by-zero
        mask_cov = (coverage_sum == 0)
        coverage_sum[mask_cov] = 1.0
        aggregator = aggregator / coverage_sum.unsqueeze(-1)

        # Flatten aggregator for encoder
        agg_flat = aggregator.view(B, -1)  # [B, C*feature_dim]

        # ----- 3) Presence detection using separate models -----
        presence_probs, presence_logits = self.predict_presence_with_separate_models(marker_values, coverage)

        # ----- 4) Integrate presence information with aggregated features -----
        combined_features = torch.cat([agg_flat, presence_probs], dim=1)

        # ----- 5) Proportion Prediction with integrated presence -----
        logits = self.encoder(combined_features)  # [B, C]
        celltype_props_raw = F.relu(logits)  # ensure >=0
        
        # Apply soft gating that preserves proportion relationships
        celltype_props_gated = self.apply_presence_gating(celltype_props_raw, presence_probs)
        
        # Normalize to ensure sum to 1
        sum_props = torch.sum(celltype_props_gated, dim=1, keepdim=True)
        celltype_props = celltype_props_gated / (sum_props + 1e-8)

        # ----- 6) Marker reconstruction -----
        reconstructed = self.decoder(celltype_props)  # [B, M]

        return celltype_props, reconstructed, valid_mask, presence_probs, presence_logits
        
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
    
    def predict_with_details(self, marker_values, coverage, batch_size=256, device=None):
        """
        Extended prediction function that returns additional details.
        
        Args:
            marker_values: Marker methylation values [N, M]
            coverage: Coverage values [N, M]
            batch_size: Batch size for processing
            device: Device to run inference on (defaults to model's device)
            
        Returns:
            tuple: (cell_props, presence_probs, reconstructed_markers)
        """
        import numpy as np
        
        # Decide which device to use (CPU/GPU)
        if device is None:
            device = next(self.parameters()).device
        
        # Convert inputs to Torch tensors if needed
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
        presence_probs_list = []
        reconstructed_list = []
        
        # Process the data in batches
        num_samples = marker_values.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                batch_X = marker_values[start_idx:end_idx].to(device)
                batch_coverage = coverage[start_idx:end_idx].to(device)
                
                # Forward pass through the model
                props, reconstructed, _, presence_probs, _ = self.forward(batch_X, batch_coverage)
                
                # Move to CPU numpy and store
                predictions_list.append(props.cpu().numpy())
                presence_probs_list.append(presence_probs.cpu().numpy())
                reconstructed_list.append(reconstructed.cpu().numpy())
        
        # Combine all batch results
        if len(predictions_list) == 0:
            # Edge case: empty input
            return (np.zeros((num_samples, self.num_celltypes)), 
                    np.zeros((num_samples, self.num_celltypes)),
                    np.zeros((num_samples, self.num_markers)))
        
        # Return the combined results
        return (
            np.vstack(predictions_list),
            np.vstack(presence_probs_list),
            np.vstack(reconstructed_list)
        )