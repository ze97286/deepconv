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
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def coverage_matched_augmentation(marker_values, coverage, clinical_coverage_params=None, augmentation_prob=1.0):
    """Enhanced augmentation focused on extreme low coverage simulation, adapted for pre-computation."""
    # Determine if input is NumPy or PyTorch
    is_torch = isinstance(marker_values, torch.Tensor)
    
    # Clone/copy based on input type
    if is_torch:
        augmented_values = marker_values.clone()
        augmented_coverage = coverage.clone()
        device = marker_values.device
        num_samples, num_markers = marker_values.shape
        zeros_fn = lambda shape: torch.zeros(shape, device=device)
        rand_fn = lambda shape: torch.rand(shape, device=device)
        normal_fn = lambda mean, std, shape: torch.normal(mean=mean, std=std, size=shape, device=device)
        clamp_fn = torch.clamp
        binomial_fn = lambda n, p: torch.distributions.binomial.Binomial(n, p).sample()
        where_fn = torch.where
    else:
        augmented_values = marker_values.copy()
        augmented_coverage = coverage.copy()
        num_samples, num_markers = marker_values.shape
        zeros_fn = np.zeros
        rand_fn = np.random.random
        normal_fn = np.random.normal
        clamp_fn = np.clip
        binomial_fn = np.random.binomial
        where_fn = np.where
    
    # Initial sanitization: where coverage == 0, set marker_values to 0
    zero_coverage_mask = augmented_coverage == 0
    augmented_values[zero_coverage_mask] = zeros_fn(zero_coverage_mask.sum().item() if is_torch else zero_coverage_mask.sum())
    
    # Process all samples
    for i in range(num_samples):
        # Get current coverage per marker
        current_cov = augmented_coverage[i]
            
        # Use clinical_coverage_params for target coverage
        if clinical_coverage_params:
            target_mean = clinical_coverage_params['mean']
            target_std = clinical_coverage_params['std']
            target_cov = normal_fn(target_mean, target_std, (1,)) if is_torch else np.random.normal(target_mean, target_std)
            target_cov = clamp_fn(target_cov, 0.5, target_mean + 2 * target_std) if is_torch else min(max(target_cov, 0.5), target_mean + 2 * target_std)
        else:
            # Fallback to previous behavior if params not provided
            if rand_fn(()) < 0.8:
                target_cov = normal_fn(1.5, 0.5, (1,)) if is_torch else np.random.exponential(1.5)
                target_cov = clamp_fn(target_cov, 0.5, 5.0) if is_torch else min(max(target_cov, 0.5), 5.0)
            else:
                target_cov = torch.exp(normal_fn(1.0, 0.7, (1,))) if is_torch else np.exp(np.random.normal(1.0, 0.7))
        
        # Calculate scaling and apply per marker
        scale = target_cov / (current_cov + 1e-8)  # Avoid division by zero
        augmented_coverage[i] = augmented_coverage[i] * scale
        
        # Add noise for very low coverage
        if (scale < 0.2).any():
            noise_level = 0.3
            noise = normal_fn(0, noise_level, (num_markers,))
            augmented_values[i] = clamp_fn(augmented_values[i] + noise, 0, 1)
            
            # Aggressive marker zeroing using zero_rate from clinical_coverage_params
            missing_prob = clinical_coverage_params['zero_rate'] if clinical_coverage_params else 0.4
            missing_mask = rand_fn((num_markers,)) < missing_prob
            augmented_coverage[i][missing_mask] = zeros_fn((missing_mask.sum().item() if is_torch else missing_mask.sum(),))
            augmented_values[i][missing_mask] = zeros_fn((missing_mask.sum().item() if is_torch else missing_mask.sum(),))
            
            # Quantize marker values for low read counts
            for j in range(num_markers):
                if augmented_coverage[i, j] > 0:
                    read_count = max(1, int(augmented_coverage[i, j]))
                    if read_count < 5:
                        successes = binomial_fn(read_count, augmented_values[i, j])
                        augmented_values[i, j] = successes / read_count
    
    # Final consistency: where coverage == 0, marker_values must be 0
    augmented_values = where_fn(augmented_coverage == 0, zeros_fn(augmented_coverage.shape), augmented_values)
    
    return augmented_values, augmented_coverage

class TissueDeconvolutionDataset(Dataset):
    """
    A PyTorch Dataset for loading cfDNA methylation data, optional labels, and NNLS predictions.
    
    Each sample in this dataset includes:
      - `fraction`: Methylation fractions across markers, in [0..1] (may contain NaNs if coverage=0).
      - `coverage`: Read coverage array of the same shape as `fraction`.
      - `y`: Ground-truth cell-type proportions for training/validation, if available.
      - `x_nnls`: Precomputed NNLS predictions, if available.
      
    Args:
        fraction (ndarray or Tensor): Shape [num_samples, num_markers].
        coverage (ndarray or Tensor): Shape [num_samples, num_markers].
        y (ndarray or Tensor, optional): Shape [num_samples, num_cell_types].
        x_nnls (ndarray or Tensor, optional): Shape [num_samples, num_cell_types].
            Precomputed NNLS predictions for regularization.
    """
    def __init__(self, fraction, coverage, y=None, x_nnls=None):
        self.fraction = torch.tensor(fraction, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
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
                 y=None, 
                 x_nnls=None,
                 target_dist_params=None,
                 augmentation_probability=0.5,
                 enable_augmentation=True):
        super().__init__(fraction, coverage, y, x_nnls)
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
                    augmentation_prob=1.0
                )
                item['X'] = torch.tensor(aug_fraction[0], dtype=torch.float32)
                item['coverage'] = torch.tensor(aug_coverage[0], dtype=torch.float32)
                item['is_augmented'] = True
        
        return item
    
    def set_training(self, training=True):
        self.training = training

class PreAugmentedTissueDataset(TissueDeconvolutionDataset):
    def __init__(self, fraction, coverage, y=None, x_nnls=None, model=None, batch_size=1024, is_clinical_like=False, is_augmented=None):
        """
        Dataset with precomputed augmentation and presence probabilities.
        
        Args:
            fraction: Methylation fractions [N, M]
            coverage: Read coverage [N, M]
            y: Ground truth labels [N, C]
            x_nnls: NNLS predictions [N, C]
            model: The entire deconvolution model (to use its presence prediction method)
            batch_size: Batch size for presence probability computation
            is_clinical_like: Flag indicating if the dataset is clinical-like (all samples augmented)
            is_augmented: Array of booleans [N] indicating which samples are augmented
        """
        super().__init__(fraction, coverage, y, x_nnls)
        self.is_clinical_like = is_clinical_like
        if is_augmented is not None:
            self.is_augmented = torch.tensor(is_augmented, dtype=torch.bool)
        else:
            self.is_augmented = None
        
        if model is not None:
            # Precompute presence probabilities in batches
            num_samples = self.fraction.size(0)
            num_cell_types = model.num_celltypes
            device = next(model.parameters()).device
            
            presence_probs = torch.zeros(num_samples, num_cell_types)
            presence_logits = torch.zeros(num_samples, num_cell_types)
            
            # Create a temporary DataLoader for batching
            temp_dataset = TensorDataset(self.fraction, self.coverage)
            temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
            
            print("Computing presence probabilities in batches...")
            print("Warning: Presence probabilities are precomputed with model in eval mode; ensure model weights are stable.")
            
            model.eval()  # Ensure model is in evaluation mode
            
            with torch.no_grad():
                for batch_idx, (batch_fraction, batch_coverage) in enumerate(tqdm(temp_loader, desc="Computing presence probabilities")):
                    batch_fraction = batch_fraction.to(device)
                    batch_coverage = batch_coverage.to(device)
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    
                    # Use the model's own method for presence prediction
                    batch_presence_probs, batch_presence_logits = model.predict_presence_with_separate_models(
                        batch_fraction, batch_coverage
                    )
                    
                    # Store the results
                    presence_probs[start_idx:end_idx] = batch_presence_probs.cpu()
                    presence_logits[start_idx:end_idx] = batch_presence_logits.cpu()
            
            print("Finished computing presence probabilities")
            self.presence_probs = presence_probs
            self.presence_logits = presence_logits
        else:
            self.presence_probs = None
            self.presence_logits = None

    def __getitem__(self, idx):
        """
        Get a sample from the dataset with its precomputed values.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
                - X: Marker methylation values
                - coverage: Read coverage
                - y: Ground truth labels (if available)
                - x_nnls: NNLS predictions (if available)
                - presence_probs: Precomputed presence probabilities (if available)
                - presence_logits: Precomputed presence logits (if available)
                - is_augmented: Flag indicating if the sample is augmented
        """
        item = super().__getitem__(idx)
        
        # Set is_augmented flag
        if self.is_clinical_like:
            item['is_augmented'] = True
        elif self.is_augmented is not None:
            item['is_augmented'] = self.is_augmented[idx].item()
        else:
            # Fallback if not provided
            item['is_augmented'] = idx >= len(self.fraction) // 2
        
        # Add precomputed presence probabilities if available
        if self.presence_probs is not None:
            item['presence_probs'] = self.presence_probs[idx]
        
        # Add precomputed presence logits if available
        if self.presence_logits is not None:
            item['presence_logits'] = self.presence_logits[idx]
            
        return item

class CellTypeDeconvolutionModel(nn.Module):
    """A neural network model for deconvolving cell-type proportions from cfDNA methylation data.

    This model predicts cell-type proportions by integrating methylation marker values, read coverage,
    pre-trained presence probabilities, and optional non-negative least squares (NNLS) predictions.
    It employs a per-marker feature extractor, an encoder with presence and NNLS inputs, and a decoder
    to reconstruct marker values. Learnable weights prioritise informative markers, select sparse markers,
    and weight coverage for robust low-coverage performance. Only markers with positive coverage contribute
    to computations, handling NaN values appropriately.

    Attributes:
        num_markers (int): Number of methylation markers (M).
        num_celltypes (int): Number of cell types to predict (C).
        feature_dim (int): Dimensionality of marker feature embeddings.
        use_x_nnls (bool): Whether to enable NNLS ensembling.
        target_ids (torch.Tensor): Long tensor mapping markers to cell type indices [M].
        marker_quality_weights (torch.Tensor): Learnable weights for marker importance [M].
        marker_selection (torch.Tensor): Learnable weights for sparse marker selection [M].
        presence_models (nn.ModuleList): Pre-trained models for predicting cell-type presence.
        marker_feature_extractor (nn.Sequential): Neural network to extract features from markers.
        encoder (nn.Sequential): Neural network to predict cell-type proportions.
        decoder (nn.Sequential): Neural network to reconstruct marker methylation values.
        presence_thresholds (torch.Tensor): Sigmoid thresholds for presence gating [C].
        presence_slopes (torch.Tensor): Sigmoid slopes for presence gating [C].
        combination_weight (nn.Parameter): Weights for ensembling DL and NNLS predictions [C].
    """

    def __init__(self, num_markers, num_cell_types, target_ids, presence_models_dir, feature_dim=64, use_x_nnls=False, initialise_weights=False):
        """Initialise the model with specified parameters.

        Args:
            num_markers (int): Number of methylation markers.
            num_cell_types (int): Number of cell types to predict.
            target_ids (array-like): Indices mapping markers to cell types.
            presence_models_dir (str): Directory containing pre-trained presence model files.
            feature_dim (int, optional): Dimensionality of marker feature embeddings. Defaults to 64.
            use_x_nnls (bool, optional): Enable NNLS ensembling. Defaults to False.
            initialise_weights (bool, optional): Use Kaiming initialisation if True. Defaults to False.
        """
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim
        self.use_x_nnls = use_x_nnls

        # Store cell-type assignments as a buffer
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)

        # Initialise weights to prioritise informative markers
        self.marker_quality_weights = nn.Parameter(torch.ones(num_markers) + 0.1 * torch.randn(num_markers))

        # Initialise weights for sparse marker selection
        self.marker_selection = nn.Parameter(torch.ones(num_markers) + 0.1 * torch.randn(num_markers))

        # Load pre-trained presence models for each cell type
        self.presence_models = nn.ModuleList()
        for cell_type_idx in range(num_cell_types):
            model_path = Path(presence_models_dir) / f"presence_model_{cell_type_idx}.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Presence model not found at {model_path}")
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                presence_model = SingleCellTypePresenceModel()
                presence_model.load_state_dict(checkpoint['model_state_dict'])
                presence_model.load_threshold(checkpoint)
            else:
                presence_model = checkpoint
            presence_model.eval()
            self.presence_models.append(presence_model)

        # Define marker feature extractor to transform single marker values into feature vectors
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Define encoder to predict proportions from aggregated features, presence probabilities, and NNLS predictions
        self.encoder = nn.Sequential(
            nn.Linear(num_cell_types * feature_dim + num_cell_types * 2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_cell_types)
        )

        # Define decoder to reconstruct marker methylation values from predicted proportions
        self.decoder = nn.Sequential(
            nn.Linear(num_cell_types, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_markers)
        )

        # Initialise presence gating parameters for scaling proportions based on presence probabilities
        thresholds = torch.ones(num_cell_types) * 0.5
        slopes = torch.ones(num_cell_types) * 10
        oac_index = 9
        thresholds[oac_index] = 0.3
        slopes[oac_index] = 15
        self.register_buffer("presence_thresholds", thresholds)
        self.register_buffer("presence_slopes", slopes)

        # Initialise ensembling weights, strongly biased toward DeepConv for low-SNR cell types
        combination_weight = torch.full((num_cell_types,), 0.5)
        for idx in [11]:
            combination_weight[idx] = -1.0
        self.combination_weight = nn.Parameter(combination_weight)

        if initialise_weights:
            self._initialise_weights()

    def _initialise_weights(self):
        """Initialise weights using Kaiming normalisation for ReLU-based networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def apply_presence_gating(self, props, probs):
        """Apply cell-type-specific scaling to proportions based on presence probabilities.

        Uses a sigmoid function parameterised by cell-type-specific thresholds and slopes to scale
        predicted proportions, enhancing predictions for cell types likely to be present.

        Args:
            props (torch.FloatTensor): Predicted proportions [B, C].
            probs (torch.FloatTensor): Presence probabilities [B, C].

        Returns:
            torch.FloatTensor: Scaled and normalised proportions [B, C].
        """
        scaling = torch.sigmoid(
            self.presence_slopes.unsqueeze(0) * (probs - self.presence_thresholds.unsqueeze(0))
        )
        scaled_props = props * scaling
        sum_props = torch.sum(scaled_props, dim=1, keepdim=True) + 1e-8
        gated_props = scaled_props / sum_props
        return gated_props

    def predict_presence_with_separate_models(self, marker_values, coverage):
        """Predict presence probabilities for each cell type using pre-trained models.

        Each cell type has a dedicated presence model that processes only the markers associated
        with that cell type, improving detection accuracy for low-SNR cell types.

        Args:
            marker_values (torch.FloatTensor): Fractional methylation values [B, M].
            coverage (torch.FloatTensor): Read coverage values [B, M].

        Returns:
            tuple:
                - torch.FloatTensor: Presence probabilities [B, C].
                - torch.FloatTensor: Raw logits before sigmoid [B, C].
        """
        B, C = marker_values.shape[0], self.num_celltypes
        presence_probs = torch.zeros(B, C, device=marker_values.device)
        presence_logits = torch.zeros(B, C, device=marker_values.device)
        valid_mask = coverage > 0
        marker_values_clean = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))

        for cell_type_idx, presence_model in enumerate(self.presence_models):
            cell_type_marker_mask = (self.target_ids == cell_type_idx)
            if not cell_type_marker_mask.any():
                continue
            cell_type_marker_values = marker_values_clean[:, cell_type_marker_mask]
            cell_type_coverage = coverage[:, cell_type_marker_mask]
            with torch.no_grad():
                logits, _, _ = presence_model(cell_type_marker_values, cell_type_coverage)
                _, adaptive_probs, _ = presence_model.adaptive_predict(cell_type_marker_values, cell_type_coverage)
                presence_logits[:, cell_type_idx] = logits.squeeze(-1)
                presence_probs[:, cell_type_idx] = adaptive_probs.squeeze(-1)

        return presence_probs, presence_logits

    def forward(self, marker_values, coverage, x_nnls=None, presence_probs=None):
        """Perform a forward pass to predict cell-type proportions and reconstruct marker values.

        This method processes input methylation marker values and coverage to predict cell-type
        proportions, integrating pre-trained presence probabilities and optional NNLS predictions.
        It applies learnable weights to prioritise informative markers, select sparse markers, and
        weight coverage for robust low-coverage performance. The encoder combines aggregated marker
        features with presence probabilities and NNLS predictions to produce proportions, which are
        scaled by presence probabilities and optionally ensembled with NNLS. A decoder reconstructs
        marker values to ensure consistency with input data. Only markers with positive coverage
        contribute to computations, handling NaN values appropriately.

        Args:
            marker_values (torch.FloatTensor): Fractional methylation values [B, M], NaN where coverage=0.
            coverage (torch.FloatTensor): Read coverage values [B, M].
            x_nnls (torch.FloatTensor, optional): NNLS predictions [B, C] for ensembling.
            presence_probs (torch.FloatTensor, optional): Precomputed presence probabilities [B, C].

        Returns:
            tuple:
                - torch.FloatTensor: Final predicted proportions [B, C], ensembled if x_nnls provided.
                - torch.FloatTensor: Presence probabilities [B, C].
                - torch.FloatTensor: Input x_nnls [B, C], or None if not provided.
                - torch.FloatTensor: Deep learning-only proportions before ensembling [B, C].
                - torch.FloatTensor: Reconstructed marker methylation values [B, M].
                - torch.BoolTensor: Mask indicating valid markers (coverage > 0) [B, M].
                - torch.FloatTensor: Marker quality weights [M].
                - torch.FloatTensor: Marker selection weights [M].
        """
        if not self.use_x_nnls:
            x_nnls = None

        B, M, C = marker_values.shape[0], marker_values.shape[1], self.num_celltypes
        valid_mask = coverage > 0

        # Extract valid markers to exclude NaN values
        coverage_flat = coverage.view(-1)
        marker_values_flat = marker_values.view(-1)
        valid_inds = torch.nonzero(coverage_flat, as_tuple=False).squeeze(1)

        if valid_inds.numel() == 0:
            props = coverage.new_zeros(B, C)
            props[:, 0] = 1.0
            reconstructed = coverage.new_zeros(B, M)
            presence_probs_out = coverage.new_zeros(B, C)
            dl_props = props.clone()
            return props, presence_probs_out, x_nnls, dl_props, reconstructed, valid_mask, self.marker_quality_weights, self.marker_selection

        coverage_valid = coverage_flat[valid_inds]
        marker_values_valid = marker_values_flat[valid_inds]
        batch_idx = valid_inds // M
        marker_idx = valid_inds % M
        celltype_idx = self.target_ids[marker_idx]

        # Apply learnable weights to prioritise informative markers
        marker_quality = torch.sigmoid(self.marker_quality_weights)
        marker_quality_valid = marker_quality[marker_idx]
        marker_values_weighted = marker_values_valid * marker_quality_valid

        # Apply weights for sparse marker selection to handle low coverage
        marker_scores = torch.sigmoid(self.marker_selection)
        marker_scores_valid = marker_scores[marker_idx]
        marker_values_weighted = marker_values_weighted * marker_scores_valid

        # Normalise coverage and weight valid markers to enhance robustness to low coverage
        log_coverage = torch.log1p(coverage) / torch.log1p(torch.tensor(1000.0, device=coverage.device))
        coverage_weights_valid = torch.sigmoid(log_coverage.view(-1)[valid_inds] / 10.0)
        marker_values_weighted = marker_values_weighted * coverage_weights_valid

        # Extract features from valid markers
        marker_values_valid_2d = marker_values_weighted.unsqueeze(1)
        features_valid = self.marker_feature_extractor(marker_values_valid_2d)

        # Aggregate features by cell type, weighted by coverage
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
        mask_cov = coverage_sum == 0
        coverage_sum[mask_cov] = 1.0
        aggregator = aggregator / coverage_sum.unsqueeze(-1)
        agg_flat = aggregator.view(B, -1)

        # Obtain presence probabilities
        if presence_probs is None:
            presence_probs, _ = self.predict_presence_with_separate_models(marker_values, coverage)
        else:
            if presence_probs.shape[0] != B:
                if presence_probs.shape[0] == 1:
                    presence_probs = presence_probs.expand(B, -1)
                else:
                    raise ValueError(f"Presence probs batch size {presence_probs.shape[0]} != input batch size {B}")

        # Combine aggregated features with presence probabilities and NNLS predictions
        if x_nnls is not None:
            combined_features = torch.cat([agg_flat, presence_probs, x_nnls], dim=1)
        else:
            combined_features = torch.cat([agg_flat, presence_probs, torch.zeros(B, C, device=marker_values.device)], dim=1)

        # Predict proportions
        logits = self.encoder(combined_features)
        dl_props = F.relu(logits)
        dl_props_gated = self.apply_presence_gating(dl_props, presence_probs)
        sum_props = torch.sum(dl_props_gated, dim=1, keepdim=True)
        dl_props_out = dl_props_gated / (sum_props + 1e-8)

        # Reconstruct marker values
        reconstructed = self.decoder(dl_props_out)

        # Ensemble with NNLS predictions if provided
        props = dl_props_out.clone()
        if x_nnls is not None:
            weight = torch.sigmoid(self.combination_weight).unsqueeze(0)
            props = weight * props + (1 - weight) * x_nnls
            row_sums = props.sum(dim=1, keepdim=True)
            valid_rows = row_sums > 0
            if valid_rows.any():
                normalization_factor = torch.where(
                    valid_rows, 1.0 / row_sums, torch.ones_like(row_sums)
                )
                props = props * normalization_factor

        return props, presence_probs, x_nnls, dl_props_out, reconstructed, valid_mask, self.marker_quality_weights, self.marker_selection

    def get_combination_weights(self):
        """Retrieve the current ensembling weights for combining DL and NNLS predictions.

        Returns:
            numpy.ndarray: Sigmoid-transformed combination weights [C].
        """
        return torch.sigmoid(self.combination_weight).detach().cpu().numpy()

    def predict(self, marker_values, coverage, batch_size=256, device=None, atlas=None, precompute_presence=True):
        """
        Generate cell-type proportion predictions in evaluation mode.

        Args:
            marker_values (array-like): Marker methylation values [N, M].
            coverage (array-like): Coverage values [N, M].
            batch_size (int, optional): Batch size for processing. Defaults to 256.
            device (torch.device, optional): Device for inference. Defaults to model's device.
            atlas (array-like, optional): Reference atlas for NNLS computation if use_x_nnls is True.
            precompute_presence (bool, optional): If True, precompute presence probabilities. Defaults to False.

        Returns:
            numpy.ndarray: Predicted cell-type proportions [N, C].
        """
        if device is None:
            device = next(self.parameters()).device

        marker_values = torch.as_tensor(marker_values, dtype=torch.float32)
        coverage = torch.as_tensor(coverage, dtype=torch.float32)
        if marker_values.dim() == 1:
            marker_values = marker_values.unsqueeze(0)
        if coverage.dim() == 1:
            coverage = coverage.unsqueeze(0)

        self.eval()
        num_samples = marker_values.shape[0]

        # Optionally precompute presence probabilities
        presence_probs = None
        if precompute_presence:
            num_cell_types = self.num_celltypes
            presence_probs = torch.zeros(num_samples, num_cell_types)
            num_batches_presence = (num_samples + batch_size - 1) // batch_size
            temp_dataset = TensorDataset(marker_values, coverage)
            temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch_idx, (batch_fraction, batch_coverage) in enumerate(temp_loader):
                    batch_fraction = batch_fraction.to(device)
                    batch_coverage = batch_coverage.to(device)
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_presence_probs, _ = self.predict_presence_with_separate_models(batch_fraction, batch_coverage)
                    presence_probs[start_idx:end_idx] = batch_presence_probs.cpu()

        predictions_list = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                batch_X = marker_values[start_idx:end_idx].to(device)
                batch_coverage = coverage[start_idx:end_idx].to(device)
                x_nnls = None
                if self.use_x_nnls and atlas is not None:
                    x_nnls_np = run_weighted_nnls(batch_X.cpu().numpy(), batch_coverage.cpu().numpy(), atlas)
                    x_nnls = torch.tensor(x_nnls_np, dtype=torch.float32, device=device)
                batch_presence_probs = presence_probs[start_idx:end_idx].to(device) if precompute_presence else None
                props, _, _, _, _, _, _, _ = self.forward(batch_X, batch_coverage, x_nnls, batch_presence_probs)
                predictions_list.append(props.cpu().numpy())
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        return np.vstack(predictions_list) if predictions_list else np.zeros((num_samples, self.num_celltypes))