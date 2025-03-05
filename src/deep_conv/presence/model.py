import torch
import torch.nn as nn
from torch.utils.data import Dataset


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


import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleCellTypePresenceModel(nn.Module):
    """
    A simplified model that predicts the presence of a single cell type.
    """
    def __init__(self, num_markers, target_ids, target_cell_type, feature_dim=32):
        super().__init__()
        self.num_markers = num_markers
        self.target_cell_type = target_cell_type
        self.feature_dim = feature_dim

        # Store marker-to-cell-type mapping
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)

        # Feature extraction
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Simple presence detection network
        self.presence_detector = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize the final layer with negative bias
        self.presence_detector[-1].bias.data.fill_(-1.0)

    def forward(self, marker_values, coverage):
        """
        Forward pass to predict presence of the target cell type.
        
        Args:
            marker_values: [B, M] Methylation values
            coverage: [B, M] Coverage values
            
        Returns:
            presence_logit: [B, 1] Logit for presence prediction
            valid_mask: [B, M] Mask of valid markers
        """
        B, M = marker_values.shape
        
        # Valid mask indicates coverage>0
        valid_mask = (coverage > 0)
        
        # Flatten for efficient processing
        coverage_flat = coverage.view(-1)
        marker_values_flat = marker_values.view(-1)
        valid_inds = torch.nonzero(coverage_flat, as_tuple=False).squeeze(1)
        
        if valid_inds.numel() == 0:
            return torch.zeros(B, 1, device=marker_values.device), valid_mask
            
        # Extract valid data
        coverage_valid = coverage_flat[valid_inds]
        marker_values_valid = marker_values_flat[valid_inds]
        batch_idx = valid_inds // M
        marker_idx = valid_inds % M
        
        # Only use markers for the target cell type
        target_mask = (self.target_ids[marker_idx] == self.target_cell_type)
        if not target_mask.any():
            return torch.zeros(B, 1, device=marker_values.device), valid_mask
            
        marker_values_target = marker_values_valid[target_mask]
        coverage_target = coverage_valid[target_mask]
        batch_idx_target = batch_idx[target_mask]
        
        # Extract features for target markers
        marker_values_2d = marker_values_target.unsqueeze(1)
        features = self.marker_feature_extractor(marker_values_2d)
        
        # Aggregate features per batch (weighted by coverage)
        aggregator = marker_values.new_zeros(B, self.feature_dim)
        coverage_sum = marker_values.new_zeros(B)
        
        # Use scatter_add for efficient aggregation
        for i in range(len(batch_idx_target)):
            b = batch_idx_target[i]
            aggregator[b] += features[i] * coverage_target[i]
            coverage_sum[b] += coverage_target[i]
        
        # Avoid divide-by-zero
        mask_cov = (coverage_sum == 0)
        coverage_sum[mask_cov] = 1.0
        aggregator = aggregator / coverage_sum.unsqueeze(-1)
        
        # Predict presence
        presence_logit = self.presence_detector(aggregator)
        
        return presence_logit, valid_mask

# class CellTypePresenceModel(nn.Module):
    # """
    # A neural network that predicts presence/absence of cell types from methylation data.
    # Presence is defined as a cell type concentration of 0.05% or higher.
    # """
    # def __init__(self, num_markers, num_cell_types, target_ids, feature_dim=32, dropout_rate=0.2):
    #     super().__init__()
    #     self.num_markers = num_markers
    #     self.num_celltypes = num_cell_types
    #     self.feature_dim = feature_dim

    #     # Store cell-type mapping
    #     target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
    #     self.register_buffer("target_ids", target_ids_t)

    #     # Feature extraction with higher dimension and more dropout
    #     self.marker_feature_extractor = nn.Sequential(
    #         nn.Linear(1, feature_dim),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout_rate),
    #         nn.Linear(feature_dim, feature_dim),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout_rate/2)
    #     )

    #     # Deeper presence detection network
    #     self.presence_detector = nn.Sequential(
    #         nn.Linear(num_cell_types * feature_dim, 256),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout_rate),
    #         nn.Linear(256, 128),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout_rate/2),
    #         nn.Linear(128, 64),
    #         nn.LeakyReLU(),
    #         nn.Linear(64, num_cell_types)
    #     )
    #     self.presence_detector[-1].bias.data.fill_(-3.0)


    # def forward(self, marker_values, coverage):
    #     """
    #     Predict presence/absence probabilities for each cell type.
        
    #     Args:
    #         marker_values (Tensor): [B, M] methylation values
    #         coverage (Tensor): [B, M] coverage values
            
    #     Returns:
    #         presence_logits (Tensor): [B, C] logits for presence prediction
    #         valid_mask (Tensor): [B, M] boolean mask of valid markers
    #     """
    #     B, M = marker_values.shape
    #     C = self.num_celltypes

    #     # Identify valid markers (coverage > 0)
    #     valid_mask = (coverage > 0)

    #     # Handle empty case
    #     if not valid_mask.any():
    #         return torch.zeros(B, C, device=marker_values.device), valid_mask

    #     # Prepare data for feature extraction
    #     coverage_flat = coverage.view(-1)
    #     marker_values_flat = marker_values.view(-1)
        
    #     # Get valid indices
    #     valid_inds = torch.nonzero(coverage_flat, as_tuple=False).squeeze(1)
    #     if valid_inds.numel() == 0:
    #         return torch.zeros(B, C, device=marker_values.device), valid_mask
            
    #     coverage_valid = coverage_flat[valid_inds]
    #     marker_values_valid = marker_values_flat[valid_inds]
    #     batch_idx = valid_inds // M
    #     marker_idx = valid_inds % M
    #     celltype_idx = self.target_ids[marker_idx]
        
    #     # Extract features
    #     marker_values_valid_2d = marker_values_valid.unsqueeze(1)
    #     features_valid = self.marker_feature_extractor(marker_values_valid_2d)
        
    #     # Aggregate features by cell type (coverage-weighted)
    #     aggregator = marker_values.new_zeros(B, C, self.feature_dim)
    #     coverage_sum = marker_values.new_zeros(B, C)
        
    #     # Flatten for efficient indexing
    #     aggregator_2d = aggregator.view(B*C, self.feature_dim)
    #     coverage_sum_1d = coverage_sum.view(B*C)
        
    #     # Calculate indices and weighted features
    #     bc_index = batch_idx * C + celltype_idx
    #     weighted_feats = coverage_valid.unsqueeze(1) * features_valid
        
    #     # Aggregate
    #     aggregator_2d.index_add_(0, bc_index, weighted_feats)
    #     coverage_sum_1d.index_add_(0, bc_index, coverage_valid)
        
    #     # Reshape and normalize
    #     aggregator = aggregator_2d.view(B, C, self.feature_dim)
    #     coverage_sum = coverage_sum_1d.view(B, C)
        
    #     # Avoid divide-by-zero
    #     mask_cov = (coverage_sum == 0)
    #     coverage_sum[mask_cov] = 1.0
    #     aggregator = aggregator / coverage_sum.unsqueeze(-1)
        
    #     # Flatten for presence detection
    #     agg_flat = aggregator.view(B, -1)
        
    #     # Predict presence logits
    #     presence_logits = self.presence_detector(agg_flat)
        
    #     return presence_logits, valid_mask