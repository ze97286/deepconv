import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os 
import pandas as pd
from pathlib import Path
from deep_conv.presence.model import SingleCellTypePresenceModel


def coverage_matched_augmentation(marker_values, coverage, augmentation_prob=0.5):
    """
    Augment training data to better match clinical coverage patterns
    
    Args:
        marker_values: [N, M] Methylation values
        coverage: [N, M] Coverage values
        augmentation_prob: Probability of applying augmentation
        
    Returns:
        augmented_values: Augmented methylation values
        augmented_coverage: Augmented coverage values
    """
    # Create copies
    augmented_values = marker_values.copy()
    augmented_coverage = coverage.copy()
    
    num_samples = len(marker_values)
    augment_mask = np.random.random(num_samples) < augmentation_prob
    
    for i in range(num_samples):
        if not augment_mask[i]:
            continue
        
        # Get current coverage
        current_cov = coverage[i].mean()
        
        # Skip if already very low
        if current_cov < 2.0:
            continue
        
        # Generate target coverage (biased toward clinical patterns)
        if np.random.random() < 0.7:
            # Clinical-like coverage (log-normal distribution)
            target_cov = np.exp(np.random.normal(1.2, 0.8))
        else:
            # Extremely low coverage
            target_cov = np.random.uniform(0.5, 2.0)
        
        # Calculate scaling factor
        scale = target_cov / current_cov
        
        # Apply scaling
        augmented_coverage[i] = coverage[i] * scale
        
        # Add noise to marker values for low coverage
        if scale < 0.3:
            noise_level = np.clip((1.0 - scale) * 0.3, 0.05, 0.3)
            noise = np.random.normal(0, noise_level, size=marker_values[i].shape)
            augmented_values[i] = np.clip(marker_values[i] + noise, 0, 1)
            
            # Simulate missing markers
            missing_prob = np.clip(0.5 * (1.0 - scale), 0, 0.5)
            missing_mask = np.random.random(marker_values[i].shape) < missing_prob
            augmented_coverage[i, missing_mask] = 0
    
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
                    augmentation_prob=1.0  # Always augment since we already decided to
                )
                
                # Update item with augmented data
                item['X'] = torch.tensor(aug_fraction[0], dtype=torch.float32)
                item['coverage'] = torch.tensor(aug_coverage[0], dtype=torch.float32)
        
        return item
    
    def set_training(self, training=True):
        """Enable/disable training mode for augmentation"""
        self.training = training
        

class CellTypeDeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types, target_ids, cell_types, feature_dim=64):
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim
        self.cell_types = cell_types
        
        # Store marker-to-cell-type mapping
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        self.register_buffer("target_ids", target_ids_t)
        
        # Marker feature extraction with coverage context
        self.marker_extractor = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU()
        )
        
        # Sample-level coverage context encoder
        self.coverage_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU()
        )
        
        # Shared base for both decoders
        self.shared_decoder_base = nn.Sequential(
            nn.Linear(num_cell_types * feature_dim + feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        
        # High coverage specific layers
        self.high_cov_specific = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_cell_types)
        )
        
        # Low coverage specific layers
        self.low_cov_specific = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_cell_types)
        )
        
        # Coverage regime classifier - kept this from original model
        self.coverage_classifier = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Binary presence classifier - explicitly learns which cell types are present/absent
        self.presence_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_cell_types),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
        # Hyperparameters for inference
        self.min_detection_threshold = 0.001
        self.post_processing_enabled = True
        
        # Cell type specific thresholds for post-processing
        self.problematic_types = ['Colon', 'Gastric', 'Small-intestine', 'Esophagus']
        self.cell_type_thresholds = {
            ct: 0.05 if ct in self.problematic_types else 0.001 
            for ct in cell_types
        }

    def _init_weights(self):
        """Initialize weights with Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize the final layer of presence classifier with negative bias
        # This makes it tend toward predicting absence (0) by default
        if hasattr(self.presence_classifier[-2], 'bias'):  # Note: -2 since -1 is sigmoid
            self.presence_classifier[-2].bias.data.fill_(-2.0)
                    
    def aggregate_cell_type_features(self, marker_features, marker_values, coverage, valid_mask):
        """Aggregate features by cell type with quality assessment"""
        B, M, F = marker_features.shape  # Batch, Markers, Features
        C = self.num_celltypes
        
        # Initialize results
        aggregated = torch.zeros(B, C, F, device=marker_features.device)
        feature_quality = torch.zeros(B, C, device=marker_features.device)
        
        for cell_type_idx in range(C):
            # Get markers for this cell type
            cell_markers = (self.target_ids == cell_type_idx)
            
            if not cell_markers.any():
                continue
                
            # Extract cell-specific data
            ct_features = marker_features[:, cell_markers, :]
            ct_values = marker_values[:, cell_markers]
            ct_coverage = coverage[:, cell_markers]
            ct_valid = valid_mask[:, cell_markers]
            
            # Skip if no valid markers
            if not ct_valid.any():
                continue
            
            # Count valid markers per sample
            valid_count = ct_valid.sum(dim=1)
            
            # Calculate coverage-based confidence (0.1 to 1.0)
            confidence = torch.clamp(ct_coverage / (ct_coverage + 5.0), 0.1, 1.0)
            
            # Weighted aggregation
            weighted_features = ct_features * confidence.unsqueeze(-1)
            sum_features = (weighted_features * ct_valid.unsqueeze(-1)).sum(dim=1)
            sum_weights = (confidence * ct_valid).sum(dim=1, keepdim=True) + 1e-8
            
            # Normalize
            agg_features = sum_features / sum_weights
            
            # Calculate quality score based on:
            # 1. Fraction of valid markers
            # 2. Average coverage of valid markers
            valid_ratio = valid_count / cell_markers.sum()
            avg_coverage = torch.sum(ct_coverage * ct_valid, dim=1) / (valid_count + 1e-8)
            coverage_quality = torch.clamp(avg_coverage / 10.0, 0.1, 1.0)
            
            # Combined quality score
            quality = valid_ratio * coverage_quality
            
            # Store results
            aggregated[:, cell_type_idx] = agg_features
            feature_quality[:, cell_type_idx] = quality
            
        return aggregated, feature_quality
                
    def forward(self, marker_values, coverage):
        """Forward pass with dual pathway and explicit presence detection"""
        B, M = marker_values.shape
        C = self.num_celltypes
        
        # Create valid marker mask
        valid_mask = (coverage > 0)
        
        # Replace invalid values with zeros
        marker_values_safe = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))
        
        # Combine marker values with their coverage
        log_marker_coverage = torch.log1p(coverage + 1e-8)
        marker_inputs = torch.cat([
            marker_values_safe.unsqueeze(-1),
            log_marker_coverage.unsqueeze(-1)
        ], dim=-1)  # [B, M, 2]
        
        # Extract marker features with coverage context
        marker_features = self.marker_extractor(marker_inputs)  # [B, M, feature_dim]
        
        # Aggregate features by cell type
        agg_features, feature_quality = self.aggregate_cell_type_features(
            marker_features, marker_values_safe, coverage, valid_mask)
        
        # Calculate coverage context features
        avg_coverage = coverage.mean(dim=1, keepdim=True)
        log_coverage = torch.log1p(avg_coverage)  # log(1+x) for numerical stability
        coverage_features = self.coverage_encoder(log_coverage)
        
        # Determine coverage regime (high vs low)
        coverage_weight = self.coverage_classifier(log_coverage)  # 1.0 = high, 0.0 = low
        
        # Flatten aggregated features
        flat_features = agg_features.reshape(B, -1)  # [B, C*F]
        
        # Combine with coverage context
        combined_features = torch.cat([flat_features, coverage_features], dim=1)
        
        # Shared processing
        shared_features = self.shared_decoder_base(combined_features)
        
        # NEW: Presence prediction
        presence_probs = self.presence_classifier(shared_features)
        
        # Branch-specific processing
        high_cov_output = F.relu(self.high_cov_specific(shared_features))
        low_cov_output = F.relu(self.low_cov_specific(shared_features))
        
        # Blend based on coverage regime
        raw_props = coverage_weight * high_cov_output + (1 - coverage_weight) * low_cov_output
        
        # Apply presence-based gating
        # Multiply by presence probabilities to enforce zero where cell type is absent
        gated_props = raw_props * presence_probs
        
        # Apply quality-aware scaling with stronger quality impact
        quality_scaling = torch.pow(feature_quality, 3)  # Stronger quality impact
        scaled_props = gated_props * quality_scaling
        
        # Normalize to sum to 1
        sum_props = scaled_props.sum(dim=1, keepdim=True) + 1e-8
        cell_props = scaled_props / sum_props
        
        # Simple reconstruction for training feedback
        reconstruction = torch.zeros_like(marker_values)
        
        # Return results
        return cell_props, reconstruction, valid_mask, feature_quality, presence_probs

    def predict(self, marker_values, coverage, batch_size=128, device=None):
        """Make predictions with optional post-processing"""
        if device is None:
            device = next(self.parameters()).device
        
        # Convert inputs to tensors if needed
        if not isinstance(marker_values, torch.Tensor):
            marker_values = torch.tensor(marker_values, dtype=torch.float32)
        if not isinstance(coverage, torch.Tensor):
            coverage = torch.tensor(coverage, dtype=torch.float32)
        
        # Ensure batch dimension
        if len(marker_values.shape) == 1:
            marker_values = marker_values.unsqueeze(0)
        if len(coverage.shape) == 1:
            coverage = coverage.unsqueeze(0)
            
        # Process in batches
        self.eval()
        all_preds = []
        all_qualities = []
        all_presence = []
        
        with torch.no_grad():
            # Process in batches
            num_samples = marker_values.shape[0]
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                
                batch_markers = marker_values[i:end].to(device)
                batch_coverage = coverage[i:end].to(device)
                
                # Forward pass
                props, _, _, quality, presence = self.forward(batch_markers, batch_coverage)
                
                # Store results
                all_preds.append(props.cpu().numpy())
                all_qualities.append(quality.cpu().numpy())
                all_presence.append(presence.cpu().numpy())
        
        # Combine results
        predictions = np.vstack(all_preds)
        qualities = np.vstack(all_qualities)
        presence_probs = np.vstack(all_presence)
        
        # Post-processing for clinical samples
        if self.post_processing_enabled:
            predictions = self._clinical_post_processing(predictions, qualities, presence_probs)
        
        return predictions
        
    def _clinical_post_processing(self, raw_preds, qualities=None, presence_probs=None):
        """Enhanced post-processing with cell-type specific thresholds"""
        processed = raw_preds.copy()
        
        # Apply cell-type specific thresholds
        for i, cell_type in enumerate(self.cell_types):
            # Get appropriate threshold for this cell type
            threshold = self.cell_type_thresholds.get(cell_type, self.min_detection_threshold)
            
            # If we have presence probabilities, use them to make better decisions
            if presence_probs is not None:
                # For problematic cell types, require stronger evidence
                is_problematic = cell_type in self.problematic_types
                presence_threshold = 0.7 if is_problematic else 0.5
                
                # Zero out low presence probability cells
                processed[:, i] = np.where(
                    presence_probs[:, i] < presence_threshold,
                    0.0,
                    processed[:, i]
                )
            
            # Apply concentration threshold
            processed[:, i] = np.where(processed[:, i] < threshold, 0, processed[:, i])
            
        # Ensure at least one non-zero prediction per sample
        zero_samples = (processed.sum(axis=1) == 0)
        if np.any(zero_samples):
            # For samples with all zeros, keep the highest raw prediction
            for i in np.where(zero_samples)[0]:
                top_idx = np.argmax(raw_preds[i])
                processed[i, top_idx] = 1.0
        
        # Re-normalize to sum to 1
        row_sums = processed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        processed = processed / row_sums
        
        return processed