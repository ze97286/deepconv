from deep_conv.benchmark.nnls import run_weighted_nnls
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
from torch.utils.data import DataLoader, TensorDataset
import tqdm

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
    A PyTorch Dataset for loading cfDNA methylation data, optional labels, and NNLS predictions.
    
    Each sample in this dataset includes:
      - `fraction`: Methylation fractions across markers, in [0..1] (may contain NaNs if coverage=0).
      - `coverage`: Read coverage array of the same shape as `fraction`.
      - `atlas`: Reference atlas or additional data, stored for convenience.
      - `y`: Ground-truth cell-type proportions for training/validation, if available.
      - `x_nnls`: Precomputed NNLS predictions, if available.
      
    Args:
        fraction (ndarray or Tensor): Shape [num_samples, num_markers].
        coverage (ndarray or Tensor): Shape [num_samples, num_markers].
        atlas (ndarray or Tensor): Reference atlas data.
        y (ndarray or Tensor, optional): Shape [num_samples, num_cell_types].
        x_nnls (ndarray or Tensor, optional): Shape [num_samples, num_cell_types].
            Precomputed NNLS predictions for regularization.
    """
    def __init__(self, fraction, coverage, atlas, y=None, x_nnls=None):
        self.fraction = torch.tensor(fraction, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.atlas = torch.tensor(atlas, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None
        if x_nnls is not None:
            self.x_nnls = torch.tensor(x_nnls, dtype=torch.float32)
        else:
            self.x_nnls = None

    def __len__(self):
        return self.fraction.size(0)

    def __getitem__(self, idx):
        item = {
            'X': self.fraction[idx],
            'coverage': self.coverage[idx],
        }
        if self.y is not None:
            item['y'] = self.y[idx]
        if self.x_nnls is not None:
            item['x_nnls'] = self.x_nnls[idx]
        return item

class AugmentedTissueDataset(TissueDeconvolutionDataset):
    def __init__(self, 
                 fraction, 
                 coverage, 
                 atlas, 
                 y=None, 
                 x_nnls=None,
                 target_dist_params=None,
                 augmentation_probability=0.5,
                 enable_augmentation=True):
        super().__init__(fraction, coverage, atlas, y, x_nnls)
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
    def __init__(self, fraction, coverage, atlas, y=None, x_nnls=None, presence_models=None, target_ids=None, batch_size=1024):
        """
        Dataset with precomputed augmentation and presence probabilities.
        
        Args:
            fraction: Methylation fractions [N, M]
            coverage: Read coverage [N, M]
            atlas: Reference atlas [M, C]
            y: Ground truth labels [N, C]
            x_nnls: NNLS predictions [N, C]
            presence_models: List of presence models
            target_ids: Tensor mapping markers to cell types
            batch_size: Batch size for presence probability computation
        """
        super().__init__(fraction, coverage, atlas, y, x_nnls)
        
        if presence_models is not None:
            # Precompute presence probabilities in batches
            num_samples = self.fraction.size(0)
            num_cell_types = len(presence_models)
            device = next(presence_models[0].parameters()).device
            
            presence_probs = torch.zeros(num_samples, num_cell_types, device=device)
            
            # Create a temporary DataLoader for batching
            temp_dataset = TensorDataset(self.fraction, self.coverage)
            temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
            
            print("Computing presence probabilities in batches...")
            
            # Create a mask for markers per cell type if target_ids is provided
            target_ids_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
            M = self.fraction.size(1)
            C = num_cell_types
                    
            cell_type_masks = torch.zeros(C, M, dtype=torch.bool, device=device)
            for cell_type_idx in range(C):
                cell_type_masks[cell_type_idx] = (target_ids_tensor == cell_type_idx)
            
            for batch_idx, (batch_fraction, batch_coverage) in enumerate(tqdm.tqdm(temp_loader, desc="Computing presence probabilities")):
                batch_fraction = batch_fraction.to(device)
                batch_coverage = batch_coverage.to(device)
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                # Create valid mask and clean marker values
                valid_mask = (batch_coverage > 0)
                batch_fraction_clean = torch.where(
                    valid_mask, 
                    batch_fraction, 
                    torch.zeros_like(batch_fraction)
                )
                
                if cell_type_masks is not None:
                    # Process each cell type with cell-specific markers
                    B = batch_fraction.size(0)
                    
                    # Stack markers for all cell types
                    marker_values_expanded = batch_fraction_clean.unsqueeze(1).expand(-1, C, -1)
                    coverage_expanded = batch_coverage.unsqueeze(1).expand(-1, C, -1)
                    cell_type_masks_expanded = cell_type_masks.unsqueeze(0).expand(B, -1, -1)
                    
                    # Mask markers not belonging to each cell type
                    marker_values_masked = torch.where(
                        cell_type_masks_expanded, 
                        marker_values_expanded, 
                        torch.zeros_like(marker_values_expanded)
                    )
                    coverage_masked = torch.where(
                        cell_type_masks_expanded, 
                        coverage_expanded, 
                        torch.zeros_like(coverage_expanded)
                    )
                    
                    # Process each cell type
                    for cell_type_idx, presence_model in enumerate(presence_models):
                        cell_type_mask = cell_type_masks[cell_type_idx]
                        if not cell_type_mask.any():
                            presence_probs[start_idx:end_idx, cell_type_idx] = 0.5
                            continue
                            
                        # Extract only markers for this cell type
                        cell_type_marker_values = marker_values_masked[:, cell_type_idx, cell_type_mask]
                        cell_type_coverage = coverage_masked[:, cell_type_idx, cell_type_mask]
                        
                        if cell_type_marker_values.shape[1] == 0:
                            presence_probs[start_idx:end_idx, cell_type_idx] = 0.5
                            continue
                        
                        # Predict presence using cell type-specific model
                        try:
                            with torch.no_grad():
                                _, adaptive_probs, _ = presence_model.adaptive_predict(
                                    cell_type_marker_values, cell_type_coverage
                                )
                            presence_probs[start_idx:end_idx, cell_type_idx] = adaptive_probs.squeeze(-1)
                        except Exception as e:
                            print(f"Error in presence model {cell_type_idx}: {e}")
                            presence_probs[start_idx:end_idx, cell_type_idx] = 0.5
                else:
                    # No target_ids, use all markers for each model
                    for cell_type_idx, presence_model in enumerate(presence_models):
                        with torch.no_grad():
                            _, adaptive_probs, _ = presence_model.adaptive_predict(
                                batch_fraction_clean, batch_coverage
                            )
                        presence_probs[start_idx:end_idx, cell_type_idx] = adaptive_probs.squeeze(-1)
            
            print("Finished computing presence probabilities")
            self.presence_probs = presence_probs.cpu()  # Move to CPU for storage
        else:
            self.presence_probs = None

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['is_augmented'] = idx >= len(self.fraction) // 2
        if self.presence_probs is not None:
            item['presence_probs'] = self.presence_probs[idx]
        return item

class CellTypeDeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types, target_ids, presence_models_dir, feature_dim=64):
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
                presence_model = SingleCellTypePresenceModel()
                presence_model.load_state_dict(checkpoint['model_state_dict'])
                presence_model.load_threshold(checkpoint)  # NEW: Load threshold if available
            else:
                presence_model = checkpoint
            
            presence_model.eval()
            self.presence_models.append(presence_model)

        # Marker Feature Extractor (unchanged from old model)
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
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

        # Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(num_cell_types, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_markers)
        )
        
        # Initialize presence gating parameters 
        thresholds = torch.ones(num_cell_types) * 0.5
        slopes = torch.ones(num_cell_types) * 10
        
        # Special handling for OAC
        oac_index = 9  
        thresholds[oac_index] = 0.3
        slopes[oac_index] = 15
        
        self.register_buffer("presence_thresholds", thresholds)
        self.register_buffer("presence_slopes", slopes)
        
        # Learnable weight for combining DL props and x_nnls
        self.combination_weight = nn.Parameter(torch.full((num_cell_types,), 0.5))

        # Initialize weights
        self._initialise_weights()

    def _initialise_weights(self):
        """Initialise weights using Kaiming normalization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def apply_presence_gating(self, props, probs):
        """
        Apply cell-type specific presence scaling based on empirical data .
        """
        # Apply scaling - vector operation across all cell types at once
        scaling = torch.sigmoid(
            self.presence_slopes.unsqueeze(0) * (probs - self.presence_thresholds.unsqueeze(0))
        )
        
        scaled_props = props * scaling
        
        # Normalise to ensure sum to 1
        sum_props = torch.sum(scaled_props, dim=1, keepdim=True) + 1e-8
        gated_props = scaled_props / sum_props
        
        return gated_props

    def predict_presence_with_separate_models(self, marker_values, coverage):
        """
        Use the separate pre-trained presence models to predict 
        presence probabilities for each cell type .
        
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
                    presence_probs[:, cell_type_idx] = 0.5
                    presence_logits[:, cell_type_idx] = 0.0
                    continue
                
                # Filter marker_values and coverage to only include markers for this cell type
                cell_type_marker_values = marker_values[:, cell_type_marker_mask]
                cell_type_coverage = coverage[:, cell_type_marker_mask]
                
                # Pass only the relevant markers to the presence model
                logits, _, _ = presence_model(cell_type_marker_values, cell_type_coverage)
                _, adaptive_probs, _ = presence_model.adaptive_predict(cell_type_marker_values, cell_type_coverage)
                # Store results
                presence_logits[:, cell_type_idx] = logits.squeeze(-1)
                presence_probs[:, cell_type_idx] = adaptive_probs.squeeze(-1)
                
        return presence_probs, presence_logits

    def forward(self, marker_values: torch.Tensor, coverage: torch.Tensor, x_nnls=None, presence_probs=None):
        """
        Forward pass to predict cell-type proportions from methylation + coverage (restored with NEW x_nnls and presence_probs).

        Args:
            marker_values (FloatTensor): [B, M], fractional methylation (NaN if coverage=0).
            coverage (FloatTensor): [B, M], read coverage.
            x_nnls (FloatTensor, optional): [B, C], NNLS predictions for ensembling.
            presence_probs (FloatTensor, optional): [B, C], precomputed presence probabilities.

        Returns:
            props (FloatTensor): [B, C], final predicted proportions (ensembled if x_nnls provided).
            presence_probs (FloatTensor): [B, C], presence probabilities (precomputed or computed).
            x_nnls (FloatTensor): [B, C], input x_nnls (passed through).
            dl_props (FloatTensor): [B, C], DL-only proportions before ensembling.
            reconstructed (FloatTensor): [B, M], reconstructed marker methylation.
            valid_mask (BoolTensor): [B, M], True where coverage>0.
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
            props = coverage.new_zeros(B, C)
            props[:, 0] = 1.0
            reconstructed = coverage.new_zeros(B, M)
            presence_probs_out = coverage.new_zeros(B, C)
            dl_props = props.clone()
            return props, presence_probs_out, x_nnls, dl_props, reconstructed, valid_mask

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
        if presence_probs is None:
            presence_probs, _ = self.predict_presence_with_separate_models(marker_values, coverage)
        else:
            # NEW: Use precomputed presence_probs if provided
            if presence_probs.shape[0] != B:
                if presence_probs.shape[0] == 1:
                    presence_probs = presence_probs.expand(B, -1)
                else:
                    raise ValueError(f"Presence probs batch size {presence_probs.shape[0]} != input batch size {B}")

        # ----- 4) Integrate presence information with aggregated features -----
        combined_features = torch.cat([agg_flat, presence_probs], dim=1)

        # ----- 5) Proportion Prediction with integrated presence -----
        logits = self.encoder(combined_features)  # [B, C]
        dl_props = F.relu(logits)  # NEW: DL-only proportions (before gating)
        
        # Apply soft gating that preserves proportion relationships
        dl_props_gated = self.apply_presence_gating(dl_props, presence_probs)
        
        # Normalize to ensure sum to 1
        sum_props = torch.sum(dl_props_gated, dim=1, keepdim=True)
        dl_props_out = dl_props_gated / (sum_props + 1e-8)  # NEW: Gated DL proportions

        # ----- 6) Marker reconstruction -----
        reconstructed = self.decoder(dl_props_out)  # [B, M]

        # NEW: Ensemble with x_nnls (from current model)
        props = dl_props_out.clone()
        if x_nnls is not None:
            if x_nnls.shape != props.shape:
                if x_nnls.shape[0] == 1 and props.shape[0] > 1:
                    x_nnls = x_nnls.expand(props.shape[0], -1)
                elif props.shape[0] == 1 and x_nnls.shape[0] > 1:
                    props = props.expand(x_nnls.shape[0], -1)
                if x_nnls.shape[1] != props.shape[1]:
                    raise ValueError(f"x_nnls has {x_nnls.shape[1]} features but props has {props.shape[1]}")
            weight = torch.sigmoid(self.combination_weight).unsqueeze(0)
            props = weight * props + (1 - weight) * x_nnls
            row_sums = props.sum(dim=1, keepdim=True)
            valid_rows = row_sums > 0
            if valid_rows.any():
                normalization_factor = torch.where(
                    valid_rows, 1.0 / row_sums, torch.ones_like(row_sums)
                )
                props = props * normalization_factor

        return props, presence_probs, x_nnls, dl_props_out, reconstructed, valid_mask

    def get_combination_weights(self):
        """NEW: Return the current combination weights (from current model)."""
        return torch.sigmoid(self.combination_weight).detach().cpu().numpy()

    def predict(self, marker_values, coverage, batch_size=256, device=None, atlas=None):
        """
        Makes predictions using the model in evaluation mode (adapted for NEW inputs).
        
        Args:
            marker_values: Marker methylation values [N, M]
            coverage: Coverage values [N, M]
            batch_size: Batch size for processing
            device: Device to run inference on (defaults to model's device)
            atlas: Optional atlas for NNLS computation
            
        Returns:
            numpy.ndarray: Cell type proportions [N, C]
        """
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

                x_nnls = None
                if atlas is not None:
                    x_nnls_np = run_weighted_nnls(batch_X.cpu().numpy(), batch_coverage.cpu().numpy(), atlas)
                    x_nnls = torch.tensor(x_nnls_np, dtype=torch.float32, device=device)

                props, _, _, _, _, _ = self.forward(batch_X, batch_coverage, x_nnls)
                predictions_list.append(props.cpu().numpy())

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        if len(predictions_list) == 0:
            return np.zeros((num_samples, self.num_celltypes))

        return np.vstack(predictions_list)