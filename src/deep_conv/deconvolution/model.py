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
    def __init__(self, num_markers, num_cell_types, presence_models_dir, target_ids=None, feature_dim=128, dropout_rate=0.1):
        """
        Initialize the Cell Type Deconvolution Model.

        Args:
            num_markers (int): Number of marker genes.
            num_cell_types (int): Number of cell types to deconvolve.
            presence_models_dir (str): Directory containing pre-trained presence models.
            target_ids (torch.Tensor, optional): Tensor mapping markers to cell types.
                If None, will be inferred from presence models.
            feature_dim (int): Dimension of the feature extraction layer.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim

        # Load pre-trained presence models
        self.presence_models = nn.ModuleList()
        print("\nLoading Presence Models:")
        for cell_type_idx in range(num_cell_types):
            model_path = Path(presence_models_dir) / f"presence_model_{cell_type_idx}.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Presence model not found at {model_path}")
            checkpoint = torch.load(model_path)
            presence_model = SingleCellTypePresenceModel()
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                presence_model.load_state_dict(checkpoint['model_state_dict'])
                presence_model.load_threshold(checkpoint)
            else:
                presence_model = checkpoint
            self.presence_models.append(presence_model)

        self.target_ids = torch.tensor(target_ids, dtype=torch.long)

        # Feature extractor: process marker values and coverage
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_markers * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Encoder: predict proportions from features and presence probabilities
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + num_cell_types, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_cell_types)
        )

        # Initialize weights
        self._initialize_weights()

        # Learnable weight for combining deep learning props and x_nnls
        self.combination_weight = nn.Parameter(torch.tensor(0.5))

    def _initialize_weights(self):
        """Initialize weights using Kaiming normalization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def predict_presence(self, marker_values, coverage):
        """
        Predict presence probabilities for each cell type using separate models.
        Each model only receives the markers specific to that cell type.
        
        Args:
            marker_values: [B, M] tensor of marker values
            coverage: [B, M] tensor of coverage values
            
        Returns:
            tuple: (presence_probs, presence_logits)
                - presence_probs: [B, C] tensor of presence probabilities
                - presence_logits: [B, C] tensor of presence logits
        """
        B = marker_values.shape[0]
        C = self.num_celltypes
        M = self.num_markers
        device = marker_values.device

        # Create a valid mask - markers with coverage > 0 should be considered
        valid_mask = (coverage > 0)
        
        # Replace NaN values in marker_values with 0 where coverage = 0
        marker_values_clean = torch.where(
            valid_mask,
            marker_values,
            torch.zeros_like(marker_values)
        )
        
        # Initialize output tensors
        presence_probs = torch.zeros(B, C, device=device)
        presence_logits = torch.zeros(B, C, device=device)

        # Create a mask for markers per cell type
        cell_type_masks = torch.zeros(C, M, dtype=torch.bool, device=device)
        for cell_type_idx in range(C):
            cell_type_masks[cell_type_idx] = (self.target_ids == cell_type_idx)

        # Stack markers for all cell types
        # Shape: [B, C, M]
        marker_values_expanded = marker_values_clean.unsqueeze(1).expand(-1, C, -1)
        coverage_expanded = coverage.unsqueeze(1).expand(-1, C, -1)
        cell_type_masks_expanded = cell_type_masks.unsqueeze(0).expand(B, -1, -1)

        # Mask markers not belonging to each cell type
        marker_values_masked = torch.where(cell_type_masks_expanded, marker_values_expanded, torch.zeros_like(marker_values_expanded))
        coverage_masked = torch.where(cell_type_masks_expanded, coverage_expanded, torch.zeros_like(coverage_expanded))

        # Process each cell type in parallel
        for cell_type_idx, presence_model in enumerate(self.presence_models):
            # Extract only markers for this cell type
            cell_type_mask = cell_type_masks[cell_type_idx]
            if not cell_type_mask.any():
                print(f"Warning: No markers assigned to cell type {cell_type_idx}")
                presence_probs[:, cell_type_idx] = 0.5
                presence_logits[:, cell_type_idx] = 0.0
                continue
                
            # Shape: [B, M_cell_type]
            cell_type_marker_values = marker_values_masked[:, cell_type_idx, cell_type_mask]
            cell_type_coverage = coverage_masked[:, cell_type_idx, cell_type_mask]

            if cell_type_marker_values.shape[1] == 0:  # Skip if no markers
                presence_probs[:, cell_type_idx] = 0.5
                presence_logits[:, cell_type_idx] = 0.0
                continue

            # Predict presence using the cell type-specific model
            try:
                logits, _, _ = presence_model(cell_type_marker_values, cell_type_coverage)
                _, adaptive_probs, _ = presence_model.adaptive_predict(cell_type_marker_values, cell_type_coverage)
                presence_logits[:, cell_type_idx] = logits.squeeze(-1)
                presence_probs[:, cell_type_idx] = adaptive_probs.squeeze(-1)
            except Exception as e:
                print(f"Error in presence model {cell_type_idx}: {e}")
                # Use default value of 0.5 for this cell type
                presence_probs[:, cell_type_idx] = 0.5
                presence_logits[:, cell_type_idx] = 0.0

        return presence_probs, presence_logits

    def forward(self, marker_values, coverage, x_nnls=None, presence_probs=None):
        """
        Forward pass with proper handling of NaN values and cell type-specific presence prediction.
        
        Args:
            marker_values: [B, M] tensor of marker values (can contain NaNs where coverage=0)
            coverage: [B, M] tensor of coverage values
            x_nnls: Optional [B, C] tensor of NNLS predictions
            presence_probs: Optional [B, C] tensor of precomputed presence probabilities
            
        Returns:
            tuple: (props, presence_probs, x_nnls)
                - props: [B, C] tensor of predicted proportions
                - presence_probs: [B, C] tensor of presence probabilities
                - x_nnls: Original NNLS predictions or None
        """
        B = marker_values.shape[0]
        
        # Create a valid mask - markers with coverage > 0 should be considered
        valid_mask = (coverage > 0)
        
        # Replace NaN values in marker_values with 0 where coverage = 0
        marker_values_clean = torch.where(
            valid_mask,
            marker_values,
            torch.zeros_like(marker_values)
        )
        
        # Use precomputed presence probs if provided, otherwise compute them
        if presence_probs is None:
            # Call the predict_presence method which now returns a tuple
            presence_tuple = self.predict_presence(marker_values_clean, coverage)
            # Extract just the presence probabilities (first element of tuple)
            presence_probs = presence_tuple[0] if isinstance(presence_tuple, tuple) else presence_tuple
        else:
            presence_probs = presence_probs.to(marker_values.device)
            
            # Ensure presence_probs has the right shape
            if presence_probs.shape[0] != B:
                if presence_probs.shape[0] == 1:
                    # Expand singleton batch dimension to match input batch size
                    presence_probs = presence_probs.expand(B, -1)
                else:
                    # This is a more serious problem - raise an error
                    raise ValueError(f"Precomputed presence_probs has batch size {presence_probs.shape[0]} but input has batch size {B}")
        
        # Use log1p for coverage which is more stable for small values
        log_coverage = torch.log1p(coverage) / 4.615  # Assuming max_coverage=100
        
        # Extract features using only valid marker values and coverage
        features_input = torch.cat([marker_values_clean, log_coverage], dim=1)  # [B, M*2]
        features = self.feature_extractor(features_input)  # [B, feature_dim]
        
        # Combine features with presence probabilities
        combined = torch.cat([features, presence_probs], dim=1)  # [B, feature_dim + C]
        
        # Predict proportions using the deep learning model
        logits = self.encoder(combined)
        props = F.softmax(logits, dim=1)  # [B, C], deep learning proportions
        
        # Ensemble with x_nnls if provided
        if x_nnls is not None:
            # Ensure x_nnls has compatible shape with props
            if x_nnls.shape != props.shape:
                # If batch dimensions don't match but can be expanded
                if x_nnls.shape[0] == 1 and props.shape[0] > 1:
                    x_nnls = x_nnls.expand(props.shape[0], -1)
                elif props.shape[0] == 1 and x_nnls.shape[0] > 1:
                    props = props.expand(x_nnls.shape[0], -1)
                
                # If feature dimensions don't match, this is more serious
                if x_nnls.shape[1] != props.shape[1]:
                    raise ValueError(f"x_nnls has {x_nnls.shape[1]} features but props has {props.shape[1]} features")
            
            # Use a bounded weight via sigmoid to ensure it's between 0 and 1
            weight = torch.sigmoid(self.combination_weight)
            props = weight * props + (1 - weight) * x_nnls
            
            # Ensure proportions sum to 1
            row_sums = props.sum(dim=1, keepdim=True)  # Shape: [B, 1]
            valid_rows = row_sums > 0  # Shape: [B, 1]

            if valid_rows.any():
                normalization_factor = torch.where(
                    valid_rows,  # Shape: [B, 1]
                    1.0 / row_sums,  # Shape: [B, 1]
                    torch.ones_like(row_sums)  # Shape: [B, 1]
                )
                props = props * normalization_factor  # Shape: [B, C] * [B, 1] -> [B, C]
        
        return props, presence_probs, x_nnls

    def predict(self, marker_values, coverage, batch_size=256, device=None, atlas=None):
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
                
                # Compute x_nnls if atlas is provided
                x_nnls = None
                if atlas is not None:
                    x_nnls_np = run_weighted_nnls(
                        batch_X.cpu().numpy(),
                        batch_coverage.cpu().numpy(),
                        atlas
                    )
                    x_nnls = torch.tensor(x_nnls_np, dtype=torch.float32, device=device)
                
                props, _, _ = self.forward(batch_X, batch_coverage, x_nnls)
                
                predictions_list.append(props.cpu().numpy())
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        if len(predictions_list) == 0:
            return np.zeros((num_samples, self.num_celltypes))
        
        return np.vstack(predictions_list)
