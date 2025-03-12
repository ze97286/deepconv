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
    
    The model uses pre-trained SingleCellTypePresenceModel instances for each cell type
    instead of an embedded presence detector network.

    Key inputs at forward pass:
      - marker_values: Fractional methylation values [B, M]. May be NaN where coverage=0.
      - coverage: Read coverage [B, M], used for weighting valid markers.
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
            
            # Load the presence model
            checkpoint = torch.load(os.path.join(model_path, f"presence_model_{model_path}.pt"))
            presence_model = SingleCellTypePresenceModel()
            presence_model.load_state_dict(checkpoint['model_state_dict'])
            presence_model.eval()  
            self.presence_models.append(presence_model)

        # ----- Marker Feature Extractor -----
        # Transforms each (scalar) methylation value into a learned feature space of dimension `feature_dim`.
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # ----- Encoder (Proportion Prediction) -----
        # Input: aggregated features (C * feature_dim).
        # Output: raw concentration logits for each cell type, ReLU => non-negative.
        self.encoder = nn.Sequential(
            nn.Linear(num_cell_types * feature_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_cell_types)
        )

        # ----- Decoder (Marker Reconstruction) -----
        # Input: predicted cell-type proportions [B, C].
        # Output: predicted marker methylation [B, M].
        self.decoder = nn.Sequential(
            nn.Linear(num_cell_types, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_markers)
        )

    def smooth_gating(self, props, probs, min_threshold=0.1, max_threshold=0.8):
        """
        Create a smoother gating transition between presence and absence.

        Instead of hard "multiplication by presence_prob",
        this function uses a sigmoid-like scaling around `min_threshold` -> `max_threshold`.

        Args:
            props (Tensor): [B, C] raw proportions (>= 0)
            probs (Tensor): [B, C] presence probabilities (0..1)
            min_threshold (float): Below this probability, props are heavily reduced
            max_threshold (float): Above this probability, props are minimally reduced

        Returns:
            gated_props (Tensor): [B, C] after smooth gating
        """
        # Clip presence probabilities to [0, 1]
        normalised_probs = torch.clamp(probs, min=0.0, max=1.0)

        # Sharpening factor: when (probs - min_threshold) is large, scaling ~1
        scaling_factor = torch.sigmoid(
            (normalised_probs - min_threshold) * 10 / (max_threshold - min_threshold)
        )

        # Apply the scaling to the original props
        gated_props = props * scaling_factor
        
        return gated_props

    def concentration_aware_gating(self, props, probs):
        """
        Apply concentration-dependent gating.

        The idea: high concentrations can tolerate lower presence confidence, 
        whereas very low concentrations require higher confidence to remain non-zero.

        Args:
            props (Tensor): [B, C] raw proportions
            probs (Tensor): [B, C] presence probabilities

        Returns:
            gated_props (Tensor): [B, C], re-scaled by a concentration-based confidence margin.
        """
        # Baseline confidence needed for each concentration:
        # if props is big, we lower the required confidence
        base_confidence = torch.clamp(0.8 - props * 4.0, min=0.2, max=0.8)

        # How much does actual presence_prob exceed the required confidence?
        confidence_margin = torch.clamp(probs - base_confidence, min=0.0)

        # Convert that margin into a scale factor from [0.1..1.0]
        scaling = 0.1 + 0.9 * (confidence_margin / (1.0 - base_confidence + 1e-8))

        # Multiply raw proportions by the scale factor
        gated_props = props * scaling

        return gated_props

    def enhanced_gating(self, props, probs):
        """
        Combine both smooth gating and concentration-aware gating.

        Steps:
         (1) Apply a smooth gating to avoid abrupt cutoff at certain presence_prob.
         (2) Apply a concentration-aware gating, so large props can survive 
             with slightly lower presence_prob, while tiny props need high presence_prob.

        Finally, we ensure a small floor to avoid exact zeros, and re-normalise so each sample sums to 1.

        Args:
            props (Tensor): [B, C] raw (non-negative) proportions
            probs (Tensor): [B, C] presence probabilities
        Returns:
            result (Tensor): [B, C], final gated & normalised proportions
        """
        # 1) Smooth gating
        smoothed = self.smooth_gating(props, probs)

        # 2) Concentration-aware gating
        result = self.concentration_aware_gating(smoothed, probs)

        # Avoid exact zero => maintain some gradient signal for rarely present cell types
        result = torch.max(result, torch.ones_like(result) * 1e-5)

        # Re-normalise across cell types (sum to 1)
        result = result / (torch.sum(result, dim=1, keepdim=True) + 1e-8)

        return result
        
    def predict_presence_with_separate_models(self, marker_values, coverage):
        """
        Use the separate pre-trained presence models to predict 
        presence probabilities for each cell type.
        
        Each presence model was trained on a subset of markers relevant to its target cell type.
        We need to filter the input data to provide only the relevant markers to each model.
        
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
            with torch.no_grad():  # No gradients needed as these models are pre-trained
                # Create a mask for the markers that belong to this cell type
                cell_type_marker_mask = (self.target_ids == cell_type_idx)
                
                # If no markers for this cell type, skip
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
            4) Predict presence_prob for each cell type (using separate models).
            5) Predict raw proportions (encoder) => ReLU => gating by presence_prob => normalised.
            6) Reconstruct marker methylation from the final proportions (decoder).

        Args:
            marker_values (FloatTensor): [B, M], fractional methylation (NaN if coverage=0).
            coverage (FloatTensor): [B, M], read coverage.

        Returns:
            celltype_props (FloatTensor): [B, C], predicted proportion for each cell type.
            reconstructed (FloatTensor): [B, M], the model's reconstruction of marker methylation.
            valid_mask (BoolTensor): [B, M], True where coverage>0.
            presence_probs (FloatTensor): [B, C], presence probability for each cell type.
            presence_logits (FloatTensor): [B, C], raw logits before sigmoid in presence_probs.
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
        # For valid marker values, get a learned feature vector
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

        # ----- 3) Presence detection -----
        # Use the dedicated presence models
        presence_probs, presence_logits = self.predict_presence_with_separate_models(marker_values, coverage)

        # ----- 4) Proportion Prediction -----
        logits = self.encoder(agg_flat)  # [B, C]
        celltype_props = F.relu(logits)  # ensure >=0

        # Enhanced gating: presence-based gating + concentration-based gating
        celltype_props = self.enhanced_gating(celltype_props, presence_probs)

        # ----- 5) Marker reconstruction -----
        reconstructed = self.decoder(celltype_props)  # [B, M]

        return celltype_props, reconstructed, valid_mask, presence_probs, presence_logits
    
