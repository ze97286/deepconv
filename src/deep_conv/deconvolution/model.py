import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os 
import pandas as pd
from pathlib import Path
from deep_conv.presence.model import SingleCellTypePresenceModel


def coverage_matched_augmentation(marker_values, coverage, target_dist_params, augmentation_prob=0.7):
    """
    Augmentation using target_dist_params to match coverage profile.
    - zero_rate: Fraction of markers with zero coverage (1 - presence_prob).
    - quantiles: Distribution to estimate fraction of non-zero markers >= 5.
    Assumes marker_values is NaN where coverage == 0.
    Works with both NumPy arrays and PyTorch tensors.
    """
    # Determine if input is NumPy or PyTorch
    is_torch = isinstance(marker_values, torch.Tensor)
    
    # Clone/copy based on input type
    if is_torch:
        augmented_values = marker_values.clone()
        augmented_coverage = coverage.clone()
        device = marker_values.device
        num_samples, num_markers = marker_values.shape
        # PyTorch operations
        zeros_fn = lambda shape: torch.zeros(shape, device=device)
        rand_fn = lambda shape: torch.rand(shape, device=device)
        normal_fn = lambda mean, std, shape: torch.normal(mean=mean, std=std, size=shape, device=device)
        clamp_fn = torch.clamp
        nan_to_num_fn = torch.nan_to_num
        binomial_fn = lambda n, p: torch.distributions.binomial.Binomial(n, p).sample()
        where_fn = torch.where
    else:
        augmented_values = marker_values.copy()
        augmented_coverage = coverage.copy()
        num_samples, num_markers = marker_values.shape
        # NumPy operations
        zeros_fn = np.zeros
        rand_fn = np.random.random
        normal_fn = np.random.normal
        clamp_fn = np.clip
        nan_to_num_fn = np.nan_to_num
        binomial_fn = np.random.binomial
        where_fn = np.where
    
    # Generate augmentation mask
    augment_mask = rand_fn((num_samples,)) < augmentation_prob
    
    # Initial sanitization: where coverage == 0, set marker_values to 0 (NaN -> 0)
    zero_coverage_mask = augmented_coverage == 0
    augmented_values[zero_coverage_mask] = 0  # Overwrites NaN where coverage is 0
    # For non-zero coverage, clamp to [0, 1] and replace any remaining NaN with 0
    non_zero_mask = ~zero_coverage_mask
    augmented_values[non_zero_mask] = clamp_fn(augmented_values[non_zero_mask], 0, 1)
    augmented_values[non_zero_mask] = nan_to_num_fn(augmented_values[non_zero_mask], nan=0.0)
    
    # Precompute quantiles for reliable probability
    q_values = [target_dist_params['quantiles'][k] for k in ['5%', '25%', '50%', '75%', '95%']]
    q_probs = [0.05, 0.25, 0.5, 0.75, 0.95]
    base_reliable_prob = 1 - np.interp(5, q_values, q_probs)  # Fraction >= 5
    
    for i in range(num_samples):
        if not augment_mask[i]:
            continue
            
        # Use zero_rate to determine presence probability
        presence_prob = 1 - target_dist_params['zero_rate']
        # Generate mask where True means non-zero coverage (present)
        non_zero_mask = rand_fn((num_markers,)) < presence_prob
        augmented_coverage[i, ~non_zero_mask] = 0  # Set to 0 where not present
        augmented_values[i, ~non_zero_mask] = 0    # Ensure marker_values is 0 where coverage is 0
        
        # For non-zero markers, adjust coverage based on quantiles
        num_non_zero = non_zero_mask.sum() if is_torch else non_zero_mask.sum()
        num_non_zero = num_non_zero.item() if is_torch else num_non_zero  # Convert to scalar
        if num_non_zero > 0:
            # Introduce variability in reliable_prob per sample
            reliable_prob = normal_fn(base_reliable_prob, 0.1, (1,))
            reliable_prob = clamp_fn(reliable_prob, 0, 1)  # Ensure valid probability
            reliable_prob = reliable_prob.item() if is_torch else reliable_prob  # Convert to scalar
            reliable_count = max(1, int(reliable_prob * num_non_zero))  # Ensure at least 1
            
            # Randomly select markers to get reliable coverage
            non_zero_indices = where_fn(non_zero_mask)[0]
            if is_torch:
                perm = torch.randperm(len(non_zero_indices), device=device)
                reliable_indices = non_zero_indices[perm[:reliable_count]]
                low_cov_indices = non_zero_indices[perm[reliable_count:]]
            else:
                indices = np.random.permutation(len(non_zero_indices))
                reliable_indices = non_zero_indices[indices[:reliable_count]]
                low_cov_indices = non_zero_indices[indices[reliable_count:]]
            
            # Vectorized assignment for reliable markers (5-20)
            if len(reliable_indices) > 0:
                new_coverage = rand_fn((len(reliable_indices),)) * (20 - 5) + 5  # Uniform 5-20
                augmented_coverage[i, reliable_indices] = new_coverage
                n = new_coverage.to(torch.int) if is_torch else new_coverage.astype(int)
                p = marker_values[i, reliable_indices]
                p = nan_to_num_fn(p, nan=0.0)  # Replace NaN with 0
                p = clamp_fn(p, 0, 1)  # Ensure valid p
                # Replace NaN with random values
                nan_mask = p == 0  # After nan_to_num, NaN becomes 0
                if nan_mask.any():
                    p[nan_mask] = rand_fn((nan_mask.sum().item() if is_torch else nan_mask.sum(),))
                successes = binomial_fn(n, p)
                augmented_values[i, reliable_indices] = successes / new_coverage
            
            # Vectorized assignment for low coverage markers (1-4)
            if len(low_cov_indices) > 0:
                new_coverage = rand_fn((len(low_cov_indices),)) * (4 - 1) + 1  # Uniform 1-4
                augmented_coverage[i, low_cov_indices] = new_coverage
                n = new_coverage.to(torch.int) if is_torch else new_coverage.astype(int)
                p = marker_values[i, low_cov_indices]
                p = nan_to_num_fn(p, nan=0.0)  # Replace NaN with 0
                p = clamp_fn(p, 0, 1)  # Ensure valid p
                nan_mask = p == 0  # After nan_to_num, NaN becomes 0
                if nan_mask.any():
                    p[nan_mask] = rand_fn((nan_mask.sum().item() if is_torch else nan_mask.sum(),))
                successes = binomial_fn(n, p)
                augmented_values[i, low_cov_indices] = successes / new_coverage
    
    # Final consistency: where coverage == 0, marker_values must be 0
    augmented_values[augmented_coverage == 0] = 0
    
    return augmented_values, augmented_coverage


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


class AugmentedTissueDataset(TissueDeconvolutionDataset):
    """
    Enhanced dataset with coverage-matched augmentation for clinical scenarios.
    
    This extends the base TissueDeconvolutionDataset by adding coverage augmentation
    to better simulate real-world clinical data distributions.
    """
    def __init__(self, 
                 fraction, 
                 coverage, 
                 atlas, 
                 y=None, 
                 target_dist_params=None,
                 augmentation_probability=0.5,
                 enable_augmentation=True):
        # Initialise the parent class
        super().__init__(fraction, coverage, atlas, y)
        
        # Store augmentation parameters
        self.target_dist_params = target_dist_params
        self.augmentation_probability = augmentation_probability
        self.enable_augmentation = enable_augmentation
        
        # Convert numpy arrays to tensors if needed
        if not isinstance(self.fraction, torch.Tensor):
            self.fraction = torch.tensor(self.fraction, dtype=torch.float32)
        if not isinstance(self.coverage, torch.Tensor):
            self.coverage = torch.tensor(self.coverage, dtype=torch.float32)
    
    def __getitem__(self, idx):
        """
        Get a dataset item with optional augmentation.
        """
        # Get the base item from parent class
        item = super().__getitem__(idx)
        
        # Apply augmentation during training if enabled
        if self.enable_augmentation and self.training and self.y is not None:
            # Convert to numpy for augmentation
            fraction_np = item['X'].numpy().reshape(1, -1)
            coverage_np = item['coverage'].numpy().reshape(1, -1)
            
            # Apply augmentation with some probability
            if np.random.random() < self.augmentation_probability:
                # Augment the data
                aug_fraction, aug_coverage = coverage_matched_augmentation(
                   fraction_np, 
                   coverage_np, 
                   self.target_dist_params, 
                   augmentation_prob=1.0
                )
                
                # Update item with augmented data
                item['X'] = torch.tensor(aug_fraction[0], dtype=torch.float32)
                item['coverage'] = torch.tensor(aug_coverage[0], dtype=torch.float32)
        
        return item
    
    def set_training(self, training=True):
        """Enable/disable training mode for augmentation"""
        self.training = training

        
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
    def __init__(self, num_markers, num_cell_types, target_ids, presence_models_dir, feature_dim=32):
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim

        # Store cell-type assignment for each marker
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)

        # Load separate presence models
        self.presence_models = nn.ModuleList()
        
        for cell_type_idx in range(num_cell_types):
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
            self.presence_models.append(presence_model)

        # Marker Feature Extractor: Takes marker values and log(coverage + 1) as input
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(2, feature_dim),  # Input: [marker_value, log(coverage + 1)]
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Encoder with presence input
        self.encoder = nn.Sequential(
            nn.Linear(num_cell_types * feature_dim + num_cell_types, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_cell_types)
        )

        # Decoder: Reconstructs marker values from proportions
        self.decoder = nn.Sequential(
            nn.Linear(num_cell_types, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_markers)
        )
        
        # Initialize presence gating parameters
        thresholds = torch.ones(num_cell_types) * 0.5
        slopes = torch.ones(num_cell_types) * 10
        
        # Special handling for OAC
        oac_index = 9  # Adjust to your actual OAC index
        thresholds[oac_index] = 0.3
        slopes[oac_index] = 15
        
        self.register_buffer("presence_thresholds", thresholds)
        self.register_buffer("presence_slopes", slopes)

    def apply_presence_gating(self, props, probs, coverage):
        """
        Apply cell-type specific presence scaling based on empirical data.
        Adjusted to use proportion of markers with coverage >= 5 for threshold adjustment.
        """
        # Calculate proportion of markers with reliable coverage (>= 5) per sample
        reliable_mask = (coverage >= 5)  # [B, M]
        reliable_ratio = reliable_mask.float().mean(dim=1, keepdim=True)  # [B, 1]
        # Adjust threshold: increase when reliable coverage is low (e.g., < 60%)
        coverage_factor = torch.clamp((0.6 - reliable_ratio) * 0.5, 0.0, 0.3)  # Max increase 0.3
        adjusted_thresholds = self.presence_thresholds + coverage_factor  # [B, C]

        # Apply scaling
        scaling = torch.sigmoid(
            self.presence_slopes.unsqueeze(0) * (probs - adjusted_thresholds)
        )
        
        scaled_props = props * scaling
        
        # Normalize to sum to 1
        sum_props = torch.sum(scaled_props, dim=1, keepdim=True) + 1e-8
        gated_props = scaled_props / sum_props
        
        return gated_props

    def predict_presence_with_separate_models(self, marker_values, coverage):
        """
        Use the separate pre-trained presence models to predict 
        presence probabilities for each cell type.
        """
        B = marker_values.shape[0]
        C = self.num_celltypes
        
        presence_probs = torch.zeros(B, C, device=marker_values.device)
        presence_logits = torch.zeros(B, C, device=marker_values.device)
        
        for cell_type_idx, presence_model in enumerate(self.presence_models):
            with torch.no_grad():
                cell_type_marker_mask = (self.target_ids == cell_type_idx)
                
                if not cell_type_marker_mask.any():
                    continue
                
                cell_type_marker_values = marker_values[:, cell_type_marker_mask]
                cell_type_coverage = coverage[:, cell_type_marker_mask]
                
                logits, _, _ = presence_model(cell_type_marker_values, cell_type_coverage)
                _, adaptive_probs, _ = presence_model.adaptive_predict(cell_type_marker_values, cell_type_coverage)
                
                presence_logits[:, cell_type_idx] = logits.squeeze(-1)
                presence_probs[:, cell_type_idx] = adaptive_probs.squeeze(-1)
                
        return presence_probs, presence_logits

    def forward(self, marker_values: torch.Tensor, coverage: torch.Tensor):
        """
        Forward pass to predict cell-type proportions from methylation + coverage.
        """
        B, M = marker_values.shape
        C = self.num_celltypes

        valid_mask = (coverage > 0)
        log_coverage = torch.log1p(coverage)
        marker_input = torch.stack([marker_values, log_coverage], dim=2)  # [B, M, 2]
        marker_input_flat = marker_input.view(-1, 2)
        coverage_flat = coverage.view(-1)
        valid_inds = torch.nonzero(coverage_flat, as_tuple=False).squeeze(1)

        if valid_inds.numel() == 0:
            celltype_props = coverage.new_zeros(B, C)
            celltype_props[:, 0] = 1.0
            reconstructed = coverage.new_zeros(B, M)
            presence_probs = coverage.new_zeros(B, C)
            presence_logits = coverage.new_zeros(B, C)
            return celltype_props, reconstructed, valid_mask, presence_probs, presence_logits

        coverage_valid = coverage_flat[valid_inds]
        marker_input_valid = marker_input_flat[valid_inds]
        batch_idx = valid_inds // M
        marker_idx = valid_inds % M
        celltype_idx = self.target_ids[marker_idx]
        features_valid = self.marker_feature_extractor(marker_input_valid)

        aggregator = coverage.new_zeros(B, C, self.feature_dim)
        coverage_sum = coverage.new_zeros(B, C)
        aggregator_2d = aggregator.view(B * C, self.feature_dim)
        coverage_sum_1d = coverage_sum.view(B * C)
        bc_index = batch_idx * C + celltype_idx
        weighted_feats = coverage_valid.unsqueeze(1) * features_valid
        aggregator_2d.index_add_(0, bc_index, weighted_feats)
        coverage_sum_1d.index_add_(0, bc_index, coverage_valid)
        aggregator = aggregator_2d.view(B, C, self.feature_dim)
        coverage_sum = coverage_sum_1d.view(B, C)
        mask_cov = (coverage_sum == 0)
        coverage_sum[mask_cov] = 1.0
        aggregator = aggregator / coverage_sum.unsqueeze(-1)
        agg_flat = aggregator.view(B, -1)

        presence_probs, presence_logits = self.predict_presence_with_separate_models(marker_values, coverage)
        combined_features = torch.cat([agg_flat, presence_probs], dim=1)
        logits = self.encoder(combined_features)
        celltype_props_raw = F.relu(logits)
        celltype_props_gated = self.apply_presence_gating(celltype_props_raw, presence_probs, coverage)
        sum_props = torch.sum(celltype_props_gated, dim=1, keepdim=True)
        celltype_props = celltype_props_gated / (sum_props + 1e-8)
        reconstructed = self.decoder(celltype_props)

        return celltype_props, reconstructed, valid_mask, presence_probs, presence_logits

    def predict(self, marker_values, coverage, batch_size=256, device=None):
        if device is None:
            device = next(self.parameters()).device
        if not isinstance(marker_values, torch.Tensor):
            marker_values = torch.tensor(marker_values, dtype=torch.float32)
        if not isinstance(coverage, torch.Tensor):
            coverage = torch.tensor(coverage, dtype=torch.float32)
        if len(marker_values.shape) == 1:
            marker_values = marker_values.unsqueeze(0)
        if len(coverage.shape) == 1:
            coverage = coverage.unsqueeze(0)
        self.eval()
        predictions_list = []
        num_samples = marker_values.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                batch_X = marker_values[start_idx:end_idx].to(device)
                batch_coverage = coverage[start_idx:end_idx].to(device)
                props, *_ = self.forward(batch_X, batch_coverage)
                predictions_list.append(props.cpu().numpy())
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        return np.vstack(predictions_list) if predictions_list else np.zeros((num_samples, self.num_celltypes))