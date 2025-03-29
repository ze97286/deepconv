import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os 
import pandas as pd
from pathlib import Path
from deep_conv.presence.model import SingleCellTypePresenceModel


def coverage_matched_augmentation(marker_values, coverage, target_dist_params=None, augmentation_probability=0.5):
    """
    Augment methylation and coverage data to better match clinical coverage distributions.
    
    Args:
        marker_values: Original marker values [samples, markers]
        coverage: Original coverage values [samples, markers]
        target_dist_params: Dict with distribution parameters for target coverage:
            - mean: Target mean coverage
            - std: Target standard deviation
            - log_params: Parameters for log-normal distribution
            - quantiles: Distribution quantiles (5%, 25%, 50%, 75%, 95%)
        augmentation_probability: Probability of applying augmentation to a sample
            
    Returns:
        augmented_values: Augmented marker values
        augmented_coverage: Augmented coverage values
    """
    # Default parameters for clinical-like coverage if not provided
    if target_dist_params is None:
        target_dist_params = {
            'mean': 5.0,  # Lower mean coverage for clinical samples
            'std': 4.0,
            'log_params': {
                'mean': 1.2,  # ln(mean) ~ 1.2
                'std': 0.8    # High variability
            },
            'quantiles': {
                '5%': 0.5,
                '25%': 2.0,
                '50%': 4.0,
                '75%': 7.0,
                '95%': 12.0
            }
        }
    
    # Create copies to modify
    augmented_values = marker_values.copy()
    augmented_coverage = coverage.copy()
    
    # Determine which samples to augment
    num_samples = len(marker_values)
    samples_to_augment = np.random.random(num_samples) < augmentation_probability
    
    for i in range(num_samples):
        if not samples_to_augment[i]:
            continue  # Skip samples not selected for augmentation
        
        # Get sample coverage statistics
        sample_coverage = coverage[i]
        current_mean = np.mean(sample_coverage)
        
        # Skip if current coverage is already very low
        if current_mean < 1.0:
            continue
        
        # Generate a target coverage level from distribution
        if np.random.random() < 0.7:
            # 70% of time: Use log-normal distribution for realistic coverage profile
            log_mean = target_dist_params['log_params']['mean']
            log_std = target_dist_params['log_params']['std']
            target_coverage = np.exp(np.random.normal(log_mean, log_std))
        else:
            # 30% of time: Use a very low coverage to simulate extreme cases
            target_coverage = np.random.uniform(0.5, 2.0)
        
        # Calculate scaling factor to match target coverage
        scaling_factor = target_coverage / (current_mean + 1e-8)
        
        # Apply coverage scaling
        augmented_coverage[i] = sample_coverage * scaling_factor
        
        # Higher noise for low scaling factors
        if scaling_factor < 0.5:
            # Calculate noise level (more reduction = more noise)
            noise_level = np.clip(0.3 * (1.0 - scaling_factor), 0.05, 0.3)
            
            # Add noise to marker values
            noise = np.random.normal(0, noise_level, size=marker_values[i].shape)
            augmented_values[i] = np.clip(marker_values[i] + noise, 0, 1)
            
            # Simulate missing marker data
            missing_prob = np.clip(1.0 - 2 * scaling_factor, 0, 0.5)  # Up to 50% missing
            missing_mask = np.random.random(marker_values[i].shape) < missing_prob
            
            # Zero out coverage for missing markers
            augmented_coverage[i, missing_mask] = 0
            
            # Handle marker values for missing data
            augmented_values[i, missing_mask] = 0  # Or NaN if your model handles it
    
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
        # Initialize the parent class
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
                    augmentation_probability=1.0  # Always augment since we already decided to
                )
                
                # Update item with augmented data
                item['X'] = torch.tensor(aug_fraction[0], dtype=torch.float32)
                item['coverage'] = torch.tensor(aug_coverage[0], dtype=torch.float32)
        
        return item
    
    def set_training(self, training=True):
        """Enable/disable training mode for augmentation"""
        self.training = training
        

class CellTypeDeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types, target_ids, presence_models_dir=None, feature_dim=32, disable_presence_gating_in_training=True):
        super().__init__()
        self.disable_presence_gating_in_training = disable_presence_gating_in_training
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store cell-type assignment for each marker
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)
        
        # Load presence models if directory provided
        self.presence_models = None
        if presence_models_dir:
            self.load_presence_models(presence_models_dir)
            
        # Try to load calibration table if it exists
        self.calibration_table = None
        if presence_models_dir:
            calib_path = os.path.join(presence_models_dir, "calibration.pt")
            if os.path.exists(calib_path):
                self.calibration_table = torch.load(calib_path)
        
        # Marker Feature Extractor
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Coverage pathway - explicitly model coverage
        self.coverage_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # High coverage branch
        self.high_cov_encoder = nn.Sequential(
            nn.Linear(num_cell_types * feature_dim + feature_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),  # Light dropout for high coverage
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_cell_types)
        )
        
        # Low coverage branch
        self.low_cov_encoder = nn.Sequential(
            nn.Linear(num_cell_types * feature_dim + feature_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),  # More dropout for low coverage
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_cell_types)
        )
        
        # Coverage gating mechanism
        self.coverage_gate = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        
        # Temperature parameter for sigmoid
        self.temp = nn.Parameter(torch.ones(1))
        
        # Decoder for marker reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(num_cell_types, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_markers)
        )
        
        # Initialize weights properly
        self._init_weights()
        
        # Store loss hyperparameters
        self.loss_params = {
            'alpha': 1.0,             
            'beta': 0.05,             
            'gamma': 0.02,            
            'coverage_weight_enabled': False,  
            'coverage_weight_scale': 0.2,      
            'cov_min_weight': 0.2,
            'cov_max_weight': 1.5,
            'cov_norm_factor': 20.0,
            'min_frac_weight': 0.5,
            'max_frac_weight': 1.5,
            'cov_threshold': 10.0,
            'sparsity_min': 1.0,
            'sparsity_max': 3.0
        }
    
    def _init_weights(self):
        """Apply proper initialization to all modules"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize coverage gate bias to produce values around 0.5
        if hasattr(self.coverage_gate[-1], 'bias'):
            self.coverage_gate[-1].bias.data.fill_(0.0)
    
    def load_presence_models(self, presence_models_dir):
        """Load pre-trained presence models"""
        self.presence_models = nn.ModuleList()
        
        for cell_type_idx in range(self.num_celltypes):
            model_path = os.path.join(presence_models_dir, f"presence_model_{cell_type_idx}.pt")
            
            if not os.path.exists(model_path):
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
    
    def predict_presence(self, marker_values, coverage):
        """Use pre-trained presence models to predict cell type presence"""
        B = marker_values.shape[0]
        C = self.num_celltypes
        
        # Initialize output tensors
        presence_probs = torch.zeros(B, C, device=marker_values.device)
        presence_logits = torch.zeros(B, C, device=marker_values.device)
        
        # Skip if no presence models loaded
        if self.presence_models is None:
            return presence_probs, presence_logits
        
        # For each cell type, use its dedicated presence model
        for cell_type_idx, presence_model in enumerate(self.presence_models):
            with torch.no_grad():  # No gradients needed for frozen presence models
                # Create a mask for markers belonging to this cell type
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
    
 
    def apply_presence_gating(self, props, probs, coverage):
        """Apply calibrated presence gating with coverage-dependent thresholds"""
        if self.training and self.disable_presence_gating_in_training:
            # Skip gating
            raw_relu = F.relu(props)
            sums = raw_relu.sum(dim=1, keepdim=True) + 1e-8
            return raw_relu / sums
        
        # Compute average coverage per sample
        avg_coverage = coverage.mean(dim=1, keepdim=True)
        
        # Dynamic thresholds based on coverage
        base_threshold = 0.5  # Default threshold
        coverage_factor = torch.clamp(10.0 / (avg_coverage + 5.0), 0.8, 1.5)
        adjusted_thresholds = base_threshold * coverage_factor
        
        # Apply calibration if available
        if self.calibration_table is not None:
            for cell_type in range(probs.size(1)):
                key = f"cell_{cell_type}"
                if key in self.calibration_table:
                    cal_factor = self.calibration_table[key]
                    adjusted_thresholds = adjusted_thresholds * cal_factor
        
        # Create sigmoid scaling factors with adaptive steepness
        slope_factor = torch.clamp(15.0 / torch.sqrt(avg_coverage + 1.0), 10.0, 30.0)
        
        # Apply scaled sigmoid gating
        scaling = torch.sigmoid(
            slope_factor * (probs - adjusted_thresholds)
        )
        
        # Apply gating to raw proportions
        gated_props = props * scaling
        
        # Re-normalize to sum to 1
        sum_props = torch.sum(gated_props, dim=1, keepdim=True) + 1e-8
        result = gated_props / sum_props
        
        # Handle extremely low coverage as a special case
        extremely_low_cov = avg_coverage < 1.0
        if extremely_low_cov.any():
            # Clone result to avoid in-place operations
            modified_result = result.clone()
            
            ext_low_indices = extremely_low_cov.squeeze().nonzero(as_tuple=True)[0]
            for idx in ext_low_indices:
                # Keep only top 1-2 predictions for extremely low coverage
                _, top_indices = torch.topk(probs[idx], k=2)
                
                # Create a mask of zeros with ones at top indices
                mask = torch.zeros_like(probs[idx], dtype=torch.bool)
                mask[top_indices] = True
                
                # Create a new row by zeroing out non-top values
                new_row = torch.zeros_like(modified_result[idx])
                new_row[mask] = modified_result[idx][mask]
                
                # Renormalize if needed
                row_sum = new_row.sum()
                if row_sum > 0:
                    # Create a normalized version without in-place operation
                    new_row = new_row / row_sum
                
                # Assign the new row
                modified_result[idx] = new_row
            
            # Use the modified result
            return modified_result
        
        # No extremely low coverage samples
        return result


    def forward(self, marker_values, coverage):
        """
        Forward pass with coverage-aware branching architecture
        
        Args:
            marker_values: [B, M] methylation values
            coverage: [B, M] coverage values
            
        Returns:
            celltype_props: [B, C] cell type proportions
            reconstructed: [B, M] reconstructed marker values
            valid_mask: [B, M] boolean mask where coverage > 0
            presence_probs: [B, C] presence probabilities
            presence_logits: [B, C] presence logits
        """
        B, M = marker_values.shape
        C = self.num_celltypes
        
        # Create valid mask (coverage > 0)
        valid_mask = (coverage > 0)
        
        # Flatten for easier processing
        coverage_flat = coverage.view(-1)
        marker_values_flat = marker_values.view(-1)
        
        # Valid indices (coverage > 0)
        valid_inds = torch.nonzero(coverage_flat, as_tuple=False).squeeze(1)
        
        # Handle case with no valid coverage
        if valid_inds.numel() == 0:
            celltype_props = coverage.new_zeros(B, C)
            celltype_props[:, 0] = 1.0  # Assign to first cell type as fallback
            reconstructed = coverage.new_zeros(B, M)
            presence_probs = coverage.new_zeros(B, C)
            presence_logits = coverage.new_zeros(B, C)
            return celltype_props, reconstructed, valid_mask, presence_probs, presence_logits
        
        # Extract values with valid coverage
        coverage_valid = coverage_flat[valid_inds]
        marker_values_valid = marker_values_flat[valid_inds]
        
        # Compute batch and marker indices
        batch_idx = valid_inds // M
        marker_idx = valid_inds % M
        
        # Get cell type for each marker
        celltype_idx = self.target_ids[marker_idx]
        
        # Extract marker features
        marker_values_valid_2d = marker_values_valid.unsqueeze(1)  # [N, 1]
        features_valid = self.marker_feature_extractor(marker_values_valid_2d)  # [N, feature_dim]
        
        # Aggregate features by cell type
        aggregator = coverage.new_zeros(B, C, self.feature_dim)
        coverage_sum = coverage.new_zeros(B, C)
        
        # Flatten for scatter operations
        aggregator_2d = aggregator.view(B*C, self.feature_dim)
        coverage_sum_1d = coverage_sum.view(B*C)
        
        # Calculate indices for aggregation
        bc_index = batch_idx * C + celltype_idx
        weighted_feats = coverage_valid.unsqueeze(1) * features_valid
        
        # Scatter-add features
        aggregator_2d.index_add_(0, bc_index, weighted_feats)
        coverage_sum_1d.index_add_(0, bc_index, coverage_valid)
        
        # Reshape back
        aggregator = aggregator_2d.view(B, C, self.feature_dim)
        coverage_sum = coverage_sum_1d.view(B, C)
        
        # Avoid divide-by-zero
        mask_cov = (coverage_sum == 0)
        coverage_sum[mask_cov] = 1.0
        aggregator = aggregator / coverage_sum.unsqueeze(-1)
        
        # Flatten aggregator
        agg_flat = aggregator.view(B, -1)  # [B, C*feature_dim]
        
        # Compute coverage features
        log_coverage = torch.log1p(coverage.mean(dim=1, keepdim=True))
        coverage_features = self.coverage_encoder(log_coverage)
        
        # Get presence predictions
        presence_probs, presence_logits = self.predict_presence(marker_values, coverage)
        
        # Combine aggregated features with coverage features
        combined_features = torch.cat([agg_flat, coverage_features], dim=1)
        
        # Process through both branches
        high_cov_output = self.high_cov_encoder(combined_features)  # [B, C]
        low_cov_output = self.low_cov_encoder(combined_features)  # [B, C]
        
        # Coverage gating
        gate_logits = self.coverage_gate(log_coverage)
        gate_value = torch.sigmoid(gate_logits / self.temp)
        
        # Combine branches based on coverage
        raw_props = gate_value * F.relu(high_cov_output) + (1 - gate_value) * F.relu(low_cov_output)
        
        # Apply presence gating
        celltype_props = self.apply_presence_gating(raw_props, presence_probs, coverage)
        
        # Reconstruct marker values
        reconstructed = self.decoder(celltype_props)  # [B, M]
        
        return celltype_props, reconstructed, valid_mask, presence_probs, presence_logits
    
    def init_temperature(self, loader, target_gate_std=0.2):
        """Initialize temperature parameter for balanced pathway usage"""
        self.eval()
        gate_values = []
        
        # Collect gate values
        with torch.no_grad():
            for batch in loader:
                marker_values = batch['X'].to(self.device)
                coverage = batch['coverage'].to(self.device)
                
                # Get gate logits
                log_coverage = torch.log1p(coverage.mean(dim=1, keepdim=True))
                gate_logits = self.coverage_gate(log_coverage)
                
                # Store values
                gate_values.append(gate_logits.cpu().numpy())
        
        # Calculate statistics
        gate_values = np.concatenate(gate_values, axis=0)
        gate_std = np.std(gate_values)
        
        # Adjust temperature
        if gate_std > 0:
            new_temp = self.temp.item() * (gate_std / target_gate_std)
            new_temp = max(0.5, min(2.0, new_temp))
            
            with torch.no_grad():
                self.temp.fill_(new_temp)
                
        self.train()
        return new_temp
    
    def predict(self, marker_values, coverage, batch_size=256, device=None):
        """
        Makes predictions using the model in evaluation mode.
        
        Args:
            marker_values: [N, M] methylation values
            coverage: [N, M] coverage values
            batch_size: Batch size for processing
            device: Device for computation
            
        Returns:
            numpy.ndarray: Cell type proportions [N, C]
        """
        if device is None:
            device = self.device
        
        # Convert inputs to torch tensors
        if not isinstance(marker_values, torch.Tensor):
            marker_values = torch.tensor(marker_values, dtype=torch.float32)
        if not isinstance(coverage, torch.Tensor):
            coverage = torch.tensor(coverage, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(marker_values.shape) == 1:
            marker_values = marker_values.unsqueeze(0)
        if len(coverage.shape) == 1:
            coverage = coverage.unsqueeze(0)
        
        self.eval()
        predictions_list = []
        
        # Process in batches
        num_samples = marker_values.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                batch_X = marker_values[start_idx:end_idx].to(device)
                batch_coverage = coverage[start_idx:end_idx].to(device)
                
                # Forward pass
                props, *_ = self.forward(batch_X, batch_coverage)
                
                # Store predictions
                predictions_list.append(props.cpu().numpy())
        
        # Combine results
        return np.vstack(predictions_list) if predictions_list else np.zeros((num_samples, self.num_celltypes))