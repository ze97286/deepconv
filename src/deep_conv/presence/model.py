import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TissueDeconvolutionDataset(Dataset):
    """
    An optimized PyTorch Dataset for loading cfDNA methylation data.
    
    Key optimizations:
    1. Precomputes and caches target cell type masks
    2. Handles NaN values more efficiently
    3. Uses pin_memory for faster data transfer to GPU
    """
    def __init__(self, fraction, coverage, atlas, y=None, 
                target_ids=None, target_cell_type=None, 
                precompute_targets=True, presence_threshold=0.0005):
        # Convert to tensors if they're not already
        self.fraction = torch.tensor(fraction, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.atlas = torch.tensor(atlas, dtype=torch.float32)
        
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None
            
        # Precompute masks for target cell type if provided
        self.target_masks = {}
        
        # Only precompute for the specific target cell type if specified
        if precompute_targets and target_ids is not None and target_cell_type is not None:
            # Create mask for this cell type's markers
            target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
            cell_mask = (target_ids_t == target_cell_type)
            self.target_masks['target_markers'] = cell_mask
            
            # Precompute presence labels if ground truth is available
            if self.y is not None and presence_threshold is not None:
                presence = (self.y[:, target_cell_type] > presence_threshold).float()
                self.target_masks['target_presence'] = presence

    def __len__(self):
        return self.fraction.size(0)

    def __getitem__(self, idx):
        """
        Return a dictionary containing:
            'X': The methylation fraction row for this sample
            'coverage': The coverage row for this sample
            'y': The ground-truth proportions, if available
        """
        # Get the base item
        item = {
            'X': self.fraction[idx],
            'coverage': self.coverage[idx],
        }
        
        # Add ground truth if available
        if self.y is not None:
            item['y'] = self.y[idx]
            
        # Add any precomputed target masks for this sample
        if 'target_presence' in self.target_masks:
            item['target_presence'] = self.target_masks['target_presence'][idx]
                
        return item



class SingleCellTypePresenceModel(nn.Module):
    """
    A model that predicts the presence of a single cell type.
    """
    def __init__(self, num_markers, target_ids, target_cell_type, feature_dim=32):
        super().__init__()
        self.num_markers = num_markers
        self.target_cell_type = target_cell_type
        self.feature_dim = feature_dim

        # Store marker-to-cell-type mapping
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)
        
        # Create a mask for the target cell type markers (precomputed)
        target_mask = (target_ids_t == target_cell_type)
        self.register_buffer("target_markers_mask", target_mask)

        # Feature extraction
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Presence detection network
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
        
        # Create valid mask (coverage > 0)
        valid_mask = (coverage > 0)
        
        # Only focus on markers for the target cell type (using precomputed mask)
        target_markers_valid = valid_mask & self.target_markers_mask.expand(B, -1)
        
        # If no valid target markers for any sample, return zeros
        if not target_markers_valid.any():
            return torch.zeros(B, 1, device=marker_values.device), valid_mask
        
        # Create a safe version of marker_values where NaNs are replaced with zeros
        # (these will be ignored in the aggregation step)
        safe_markers = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))
        
        # Process only the target markers that are valid
        # We'll expand the mask to find valid samples
        samples_with_valid_targets = target_markers_valid.any(dim=1)
        
        # If no sample has valid target markers, return zeros
        if not samples_with_valid_targets.any():
            return torch.zeros(B, 1, device=marker_values.device), valid_mask
        
        # Extract features for all markers in one operation
        marker_values_2d = safe_markers.unsqueeze(2)  # [B, M, 1]
        all_features = self.marker_feature_extractor(marker_values_2d)  # [B, M, feature_dim]
        
        # Use the target marker mask to zero out non-target markers
        target_mask_expanded = self.target_markers_mask.view(1, M, 1).expand(B, -1, self.feature_dim)
        valid_mask_expanded = valid_mask.unsqueeze(2).expand(-1, -1, self.feature_dim)
        
        # Zero out features for non-target or invalid markers
        masked_features = all_features * target_mask_expanded * valid_mask_expanded
        
        # Weight the features by coverage
        coverage_expanded = coverage.unsqueeze(2).expand(-1, -1, self.feature_dim)
        weighted_features = masked_features * coverage_expanded
        
        # Aggregate features per sample
        summed_features = weighted_features.sum(dim=1)  # [B, feature_dim]
        
        # Calculate the sum of coverage for normalization
        target_coverage = coverage * target_markers_valid
        coverage_sum = target_coverage.sum(dim=1, keepdim=True)  # [B, 1]
        
        # Avoid divide-by-zero (replace zeros with ones for safe division)
        safe_coverage_sum = torch.where(
            coverage_sum > 0, 
            coverage_sum, 
            torch.ones_like(coverage_sum)
        )
        
        # Normalize features by coverage
        normalized_features = summed_features / safe_coverage_sum.expand(-1, self.feature_dim)
        
        # Predict presence
        presence_logit = self.presence_detector(normalized_features)
        
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