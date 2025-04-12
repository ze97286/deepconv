import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os 
import pandas as pd
from pathlib import Path
from deep_conv.presence.model import SingleCellTypePresenceModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def coverage_matched_augmentation(marker_values, coverage, target_dist_params, augmentation_prob=1.0):
    """
    Augmentation to match the clinical dataset's coverage profile using target_dist_params.
    - zero_rate: Fraction of markers at 0.
    - 62% of markers in 5-<max> (adjusted to achieve mean ~9.1).
    - Remaining non-zero markers in 1-4.
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
    
    # Initial sanitization: where coverage == 0, set marker_values to 0 (NaN -> 0)
    zero_coverage_mask = augmented_coverage == 0
    augmented_values[zero_coverage_mask] = zeros_fn(zero_coverage_mask.sum().item() if is_torch else zero_coverage_mask.sum())
    # For non-zero coverage, clamp to [0, 1] and replace any remaining NaN with 0
    non_zero_mask = ~zero_coverage_mask
    augmented_values[non_zero_mask] = clamp_fn(augmented_values[non_zero_mask], 0.0, 1.0)
    augmented_values[non_zero_mask] = nan_to_num_fn(augmented_values[non_zero_mask], nan=0.0)
    
    # Extract zero_rate from target_dist_params and convert to tensor if needed
    zero_fraction = target_dist_params['zero_rate']  # 0.03 for low
    if is_torch:
        zero_fraction = torch.tensor(zero_fraction, device=device)
    
    # Estimate upper bound from quantiles (extrapolate to 99th percentile)
    q_values = [target_dist_params['quantiles'][k] for k in ['5%', '25%', '50%', '75%', '95%']]
    q_probs = [0.05, 0.25, 0.5, 0.75, 0.95]
    assumed_max = 30.0  # Based on observed tail in the plot
    slope = (assumed_max - q_values[-1]) / (1.0 - q_probs[-1])
    target_prob = 0.99
    max_coverage = q_values[-1] + slope * (target_prob - q_probs[-1])
    max_coverage = min(max_coverage, 30.0)  # Cap at 30 based on observed tail
    max_coverage = 22.25
    
    # Target fractions
    high_cov_fraction = 0.62
    low_cov_fraction = 1.0 - zero_fraction - high_cov_fraction
    if is_torch:
        high_cov_fraction = torch.tensor(high_cov_fraction, device=device)
        low_cov_fraction = torch.tensor(low_cov_fraction, device=device)
    
    # Generate augmentation mask for all samples
    augment_mask = rand_fn((num_samples,)) < augmentation_prob
    
    # Only process samples where augment_mask is True
    if augment_mask.any():
        # Introduce sample-level variability in fractions for all samples
        delta = normal_fn(0, 0.6, (num_samples,))
        delta = clamp_fn(delta, -0.6, 0.6)
        zero_frac = clamp_fn(zero_fraction + delta, 0.0, 0.15)  # Shape: (num_samples,)
        low_cov_frac = clamp_fn(low_cov_fraction - delta / 2, 0.0, 0.77)  # Adjusted to allow more variability
        high_cov_frac = 1.0 - zero_frac - low_cov_frac  # Shape: (num_samples,)
          
        # Generate random values for all samples and markers
        rand_vals = rand_fn((num_samples, num_markers))  # Shape: (num_samples, num_markers)
        
        # Compute cumulative thresholds for each sample
        zero_threshold = zero_frac.unsqueeze(1) if is_torch else zero_frac[:, np.newaxis]  # Shape: (num_samples, 1)
        low_cov_threshold = (zero_frac + low_cov_frac).unsqueeze(1) if is_torch else (zero_frac + low_cov_frac)[:, np.newaxis]  # Shape: (num_samples, 1)
        
        # Generate masks for each coverage range
        zero_mask = rand_vals < zero_threshold  # Shape: (num_samples, num_markers)
        low_cov_mask = (rand_vals >= zero_threshold) & (rand_vals < low_cov_threshold)  # Shape: (num_samples, num_markers)
        high_cov_mask = rand_vals >= (1.0 - high_cov_frac.unsqueeze(1) if is_torch else high_cov_frac[:, np.newaxis])  # Shape: (num_samples, num_markers)
        
        # Apply augmentation only to samples where augment_mask is True
        samples_to_augment = where_fn(augment_mask, True, False)  # Shape: (num_samples,)
        samples_to_augment = samples_to_augment.unsqueeze(1) if is_torch else samples_to_augment[:, np.newaxis]  # Shape: (num_samples, 1)
        
        # Zero coverage
        augmented_coverage = where_fn(zero_mask & samples_to_augment, zeros_fn((num_samples, num_markers)), augmented_coverage)
        augmented_values = where_fn(zero_mask & samples_to_augment, zeros_fn((num_samples, num_markers)), augmented_values)
        
        # 1-4 range: Uniform 1-4
        new_coverage_low = rand_fn((num_samples, num_markers)) * (4 - 1) + 1  # Shape: (num_samples, num_markers)
        augmented_coverage = where_fn(low_cov_mask & samples_to_augment, new_coverage_low, augmented_coverage)
        n_low = new_coverage_low.to(torch.int) if is_torch else new_coverage_low.astype(int)
        p_low = where_fn(low_cov_mask & samples_to_augment, marker_values, zeros_fn((num_samples, num_markers)))
        p_low = nan_to_num_fn(p_low, nan=0.0)
        p_low = clamp_fn(p_low, 0.0, 1.0)
        nan_mask_low = p_low == 0
        if nan_mask_low.any():
            p_low = where_fn(nan_mask_low, rand_fn((num_samples, num_markers)), p_low)
        successes_low = binomial_fn(n_low, p_low)
        new_values_low = successes_low / new_coverage_low
        augmented_values = where_fn(low_cov_mask & samples_to_augment, new_values_low, augmented_values)
        
        # 5-<max> range: Uniform 5-<max>
        new_coverage_high = rand_fn((num_samples, num_markers)) * (max_coverage - 5) + 5  # Shape: (num_samples, num_markers)
        augmented_coverage = where_fn(high_cov_mask & samples_to_augment, new_coverage_high, augmented_coverage)
        n_high = new_coverage_high.to(torch.int) if is_torch else new_coverage_high.astype(int)
        p_high = where_fn(high_cov_mask & samples_to_augment, marker_values, zeros_fn((num_samples, num_markers)))
        p_high = nan_to_num_fn(p_high, nan=0.0)
        p_high = clamp_fn(p_high, 0.0, 1.0)
        nan_mask_high = p_high == 0
        if nan_mask_high.any():
            p_high = where_fn(nan_mask_high, rand_fn((num_samples, num_markers)), p_high)
        successes_high = binomial_fn(n_high, p_high)
        new_values_high = successes_high / new_coverage_high
        augmented_values = where_fn(high_cov_mask & samples_to_augment, new_values_high, augmented_values)
    
    # Final consistency: where coverage == 0, marker_values must be 0
    augmented_values = where_fn(augmented_coverage == 0, zeros_fn(augmented_coverage.shape), augmented_values)
        
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
        item = {
            'X': self.fraction[idx],
            'coverage': self.coverage[idx],
        }
        if self.y is not None:
            item['y'] = self.y[idx]
        return item  

class AugmentedTissueDataset(TissueDeconvolutionDataset):
    def __init__(self, 
                 fraction, 
                 coverage, 
                 atlas, 
                 y=None, 
                 target_dist_params=None,
                 augmentation_probability=0.5,
                 enable_augmentation=True):
        super().__init__(fraction, coverage, atlas, y)
        self.target_dist_params = target_dist_params
        self.augmentation_probability = augmentation_probability
        self.enable_augmentation = enable_augmentation
        if not isinstance(self.fraction, torch.Tensor):
            self.fraction = torch.tensor(self.fraction, dtype=torch.float32)
        if not isinstance(self.coverage, torch.Tensor):
            self.coverage = torch.tensor(self.coverage, dtype=torch.float32)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # Initialize augmentation flag
        item['is_augmented'] = False
        
        if self.enable_augmentation and self.training and self.y is not None:
            fraction_np = item['X'].numpy().reshape(1, -1)
            coverage_np = item['coverage'].numpy().reshape(1, -1)
            
            # Apply augmentation with some probability
            if np.random.random() < self.augmentation_probability:
                aug_fraction, aug_coverage = coverage_matched_augmentation(
                    fraction_np, 
                    coverage_np, 
                    self.target_dist_params, 
                    augmentation_prob=1.0
                )
                item['X'] = torch.tensor(aug_fraction[0], dtype=torch.float32)
                item['coverage'] = torch.tensor(aug_coverage[0], dtype=torch.float32)
                item['is_augmented'] = True
        
        return item
    
    def set_training(self, training=True):
        self.training = training

class PreAugmentedTissueDataset(TissueDeconvolutionDataset):
    def __init__(self, fraction, coverage, atlas, y=None):
        super().__init__(fraction, coverage, atlas, y)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['is_augmented'] = idx >= len(self.fraction) // 2 
        return item

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.relu = nn.LeakyReLU()
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + residual
        out = self.relu(out)
        return out

class CellTypeDeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types, target_ids, presence_models_dir, feature_dim=128):
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim

        # Store cell-type assignment for each marker
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)

        # Load separate presence models
        self.presence_models = nn.ModuleList()

        print("\nLoading Presence Models:")
        for cell_type_idx in range(num_cell_types):
            model_path = Path(presence_models_dir) / f"presence_model_{cell_type_idx}.pt"

            if not model_path.exists():
                raise FileNotFoundError(f"Presence model not found at {model_path}")
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                from deep_conv.presence.model import SingleCellTypePresenceModel
                presence_model = SingleCellTypePresenceModel()
                presence_model.load_state_dict(checkpoint['model_state_dict'])
                presence_model.load_threshold(checkpoint)
                print(f"Cell Type {cell_type_idx}: Loaded Specificity Threshold = {presence_model.specificity_threshold:.4f}")
            else:
                presence_model = checkpoint
                if not hasattr(presence_model, 'specificity_threshold'):
                    presence_model.specificity_threshold = 0.5
                print(f"Cell Type {cell_type_idx}: Loaded Specificity Threshold = {presence_model.specificity_threshold:.4f} (from model)")
            self.presence_models.append(presence_model)

        # Marker Feature Extractor: Processes marker values and normalized log(coverage)
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(2, feature_dim),  # Input: [marker_value, log_coverage]
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Encoder with presence input
        self.encoder = nn.Sequential(
            nn.Linear(num_cell_types * feature_dim + num_cell_types, 256),
            nn.LeakyReLU(),
            ResidualBlock(256, 256),
            nn.Linear(256, num_cell_types)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(num_cell_types, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_markers)
        )

        # Apply Kaiming initialization to all linear layers
        self._initialize_weights()

        # Initialise presence gating parameters
        thresholds = torch.ones(num_cell_types) * 0.5
        slopes = torch.ones(num_cell_types) * 10
        oac_index = 9  
        tcells_index = 11  
        thresholds[oac_index] = 0.4  
        thresholds[tcells_index] = 0.3
        slopes[oac_index] = 15
        slopes[tcells_index] = 10
        self.register_buffer("presence_thresholds", thresholds)
        self.register_buffer("presence_slopes", slopes)

        # Maximum log(coverage + 1) for normalization (max coverage of 100)
        self.register_buffer("max_log_coverage", torch.tensor(4.615))  # log(101)

        # Maximum coverage for normalizing mean_coverage in apply_presence_gating
        self.register_buffer("max_coverage", torch.tensor(100.0))

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def apply_presence_gating(self, props, probs, coverage):
        mean_coverage = coverage.mean(dim=1)  # [B]
        # Normalize mean_coverage to [0, 1]
        mean_coverage_normalized = mean_coverage / self.max_coverage  # [B]
        coverage_adjustment = torch.clamp((0.5 - mean_coverage_normalized) * 0.4, -0.1, 0.2)  # [B]
        adjusted_thresholds = self.presence_thresholds.unsqueeze(0) + coverage_adjustment.unsqueeze(1)  # [B, C]
        scaling = torch.sigmoid(
            self.presence_slopes.unsqueeze(0) * (probs - adjusted_thresholds)
        )
        scaled_props = props * scaling
        sum_props = torch.sum(scaled_props, dim=1, keepdim=True) + 1e-8
        gated_props = scaled_props / sum_props
        return gated_props

    def predict_presence_with_separate_models(self, marker_values, coverage):
        B = marker_values.shape[0]
        C = self.num_celltypes
        M = self.num_markers
        device = marker_values.device

        # Initialize output tensors
        presence_probs = torch.zeros(B, C, device=device)
        presence_logits = torch.zeros(B, C, device=device)

        # Create a mask for markers per cell type
        cell_type_masks = torch.zeros(C, M, dtype=torch.bool, device=device)
        for cell_type_idx in range(C):
            cell_type_masks[cell_type_idx] = (self.target_ids == cell_type_idx)

        # Stack markers for all cell types
        # Shape: [B, C, M]
        marker_values_expanded = marker_values.unsqueeze(1).expand(-1, C, -1)
        coverage_expanded = coverage.unsqueeze(1).expand(-1, C, -1)
        cell_type_masks_expanded = cell_type_masks.unsqueeze(0).expand(B, -1, -1)

        # Mask markers not belonging to each cell type
        marker_values_masked = torch.where(cell_type_masks_expanded, marker_values_expanded, torch.zeros_like(marker_values_expanded))
        coverage_masked = torch.where(cell_type_masks_expanded, coverage_expanded, torch.zeros_like(coverage_expanded))

        # Process each cell type in parallel
        for cell_type_idx, presence_model in enumerate(self.presence_models):
            # Shape: [B, M_cell_type]
            cell_type_marker_values = marker_values_masked[:, cell_type_idx, cell_type_masks[cell_type_idx]]
            cell_type_coverage = coverage_masked[:, cell_type_idx, cell_type_masks[cell_type_idx]]

            if cell_type_marker_values.shape[1] == 0:  # Skip if no markers
                continue

            logits, _, _ = presence_model(cell_type_marker_values, cell_type_coverage)
            _, adaptive_probs, _ = presence_model.adaptive_predict(cell_type_marker_values, cell_type_coverage)
            presence_logits[:, cell_type_idx] = logits.squeeze(-1)
            presence_probs[:, cell_type_idx] = adaptive_probs.squeeze(-1)

        return presence_probs, presence_logits
    
    def forward(self, marker_values: torch.Tensor, coverage: torch.Tensor):
        B, M = marker_values.shape
        C = self.num_celltypes

        # Debug: Check for nan in inputs
        if torch.isnan(coverage).any() or torch.isinf(coverage).any():
            print("Warning: coverage contains nan or inf values")

        # Valid mask indicates coverage>0
        valid_mask = (coverage > 0)

        # Compute log(coverage + 1) and normalize to [0, 1]
        log_coverage = torch.log(coverage + 1)  # [B, M]
        log_coverage_normalized = log_coverage / self.max_log_coverage  # [B, M]

        # Debug: Check for nan in log_coverage
        if torch.isnan(log_coverage_normalized).any():
            print("Warning: log_coverage_normalized contains nan values")

        # Flatten coverage, marker_values, and log_coverage for efficient indexing
        coverage_flat = coverage.view(-1)
        marker_values_flat = marker_values.view(-1)
        log_coverage_flat = log_coverage_normalized.view(-1)

        # Indices of non-zero coverage
        valid_inds = torch.nonzero(coverage_flat, as_tuple=False).squeeze(1)

        # Handle all-zero-coverage case
        if valid_inds.numel() == 0:
            celltype_props = coverage.new_zeros(B, C)
            celltype_props[:, 0] = 1.0
            reconstructed = coverage.new_zeros(B, M)
            presence_probs = coverage.new_zeros(B, C)
            presence_logits = coverage.new_zeros(B, C)
            return celltype_props, reconstructed, valid_mask, presence_probs, presence_logits

        # Extract coverage, marker_values, and log_coverage for valid coverage
        coverage_valid = coverage_flat[valid_inds]
        marker_values_valid = marker_values_flat[valid_inds]
        log_coverage_valid = log_coverage_flat[valid_inds]

        # Compute the batch index and marker index from flattened indices
        batch_idx = valid_inds // M
        marker_idx = valid_inds % M

        # Each marker corresponds to a specific cell type (via self.target_ids)
        celltype_idx = self.target_ids[marker_idx]

        # ----- 1) Marker Feature Extraction -----
        features_input = torch.stack([marker_values_valid, log_coverage_valid], dim=1)  # [N, 2]
        features_valid = self.marker_feature_extractor(features_input)  # [N, feature_dim]

        # ----- 2) Aggregate features by cell type -----
        aggregator = coverage.new_zeros(B, C, self.feature_dim)
        coverage_sum = coverage.new_zeros(B, C)

        aggregator_2d = aggregator.view(B*C, self.feature_dim)
        coverage_sum_1d = coverage_sum.view(B*C)

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

        # ----- 3) Presence detection using separate models -----
        presence_probs, presence_logits = self.predict_presence_with_separate_models(marker_values, coverage)

        # ----- 4) Integrate presence information with aggregated features -----
        combined_features = torch.cat([agg_flat, presence_probs], dim=1)

        # ----- 5) Proportion Prediction with integrated presence -----
        logits = self.encoder(combined_features)
        celltype_props_raw = F.relu(logits)
        celltype_props_gated = self.apply_presence_gating(celltype_props_raw, presence_probs, coverage)
        sum_props = torch.sum(celltype_props_gated, dim=1, keepdim=True)
        celltype_props = celltype_props_gated / (sum_props + 1e-8)

        # Debug: Check for nan in celltype_props
        if torch.isnan(celltype_props).any():
            print("Warning: celltype_props contains nan values")

        # ----- 6) Marker reconstruction -----
        reconstructed = self.decoder(celltype_props)

        # Debug: Check for nan in reconstructed
        if torch.isnan(reconstructed).any():
            print("Warning: reconstructed contains nan values")

        return celltype_props, reconstructed, valid_mask, presence_probs, presence_logits
    
    def predict_with_adaptive_threshold(self, marker_values, coverage, base_threshold=0.5):
        logits, _ = self.forward(marker_values, coverage)
        probabilities = torch.sigmoid(logits).squeeze(-1)
        
        mean_coverage = coverage.mean(dim=1)
        coverage_adjustment = torch.clamp((20.0 - mean_coverage) / 40.0, -0.1, 0.2)
        adjusted_thresholds = base_threshold + coverage_adjustment
        
        predictions = (probabilities >= adjusted_thresholds).float()
        
        return predictions, probabilities
    
    def predict(self, marker_values, coverage, batch_size=256, device=None):
        import numpy as np
        
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
        
        if len(predictions_list) == 0:
            return np.zeros((num_samples, self.num_celltypes))
        
        return np.vstack(predictions_list)
    
    def predict_with_details(self, marker_values, coverage, batch_size=256, device=None):
        import numpy as np
        
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
        presence_probs_list = []
        reconstructed_list = []
        
        num_samples = marker_values.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                batch_X = marker_values[start_idx:end_idx].to(device)
                batch_coverage = coverage[start_idx:end_idx].to(device)
                
                props, reconstructed, _, presence_probs, _ = self.forward(batch_X, batch_coverage)
                
                predictions_list.append(props.cpu().numpy())
                presence_probs_list.append(presence_probs.cpu().numpy())
                reconstructed_list.append(reconstructed.cpu().numpy())
        
        if len(predictions_list) == 0:
            return (np.zeros((num_samples, self.num_celltypes)), 
                    np.zeros((num_samples, self.num_celltypes)),
                    np.zeros((num_samples, self.num_markers)))
        
        return (
            np.vstack(predictions_list),
            np.vstack(presence_probs_list),
            np.vstack(reconstructed_list)
        )