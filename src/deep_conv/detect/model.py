import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import math
import scipy.stats as stats
import numpy as np


class EnhancedCancerDetectionModel(nn.Module):
    """
    Enhanced deep learning model for cancer detection from cfDNA methylation markers.
    This model uses transformer architecture with attention mechanisms to estimate
    cell type concentration and provide calibrated confidence intervals.
    """
    def __init__(self, num_markers, feature_dim=96, num_heads=6, num_layers=2, 
                 dropout_rate=0.2, detection_thresholds=(0.001, 0.005, 0.01, 0.05),
                 background_level=0.05, min_reliable_coverage=5.0):
        """
        Initialize the EnhancedCancerDetectionModel.
        
        Args:
            num_markers: Number of methylation markers in input
            feature_dim: Dimension of feature representations
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer encoder layers
            dropout_rate: Dropout probability for regularization
            detection_thresholds: Concentration thresholds for binary detection
            background_level: Initial background level for correction
            min_reliable_coverage: Minimum coverage to consider a marker reliable
        """
        super().__init__()
        
        # Store configuration parameters
        self.detection_thresholds = detection_thresholds
        self.num_markers = num_markers
        self.min_reliable_coverage = min_reliable_coverage
            
        # Separate embeddings for marker values and coverage
        # These transform the raw inputs into higher-dimensional representations
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
        
        # Transformer encoder for learning marker interactions
        # Uses pre-norm for better training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout_rate,
            activation=F.gelu,  # GELU for smoother gradients
            batch_first=True,
            norm_first=True     # Pre-norm architecture
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention mechanism for marker importance weighting
        # Learns which markers are most informative
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Coverage reliability weighting mechanism
        # Produces weights that reduce the influence of low-coverage markers
        self.reliability_weight = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Concentration estimation head (mu)
        # Predicts the cell type concentration (0-1)
        self.mu_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation head
        # Predicts the uncertainty/confidence in the concentration estimate
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()  # Ensures positive uncertainty values
        )
        
        # Binary detection heads for different concentration thresholds
        # Each head predicts the probability that concentration exceeds the threshold
        self.detection_heads = nn.ModuleList()
        for _ in detection_thresholds:
            head = nn.Sequential(
                nn.Linear(feature_dim + 1, feature_dim // 2),  # +1 for uncertainty feature
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            )
            # Initialize with negative bias for lower initial predictions
            # This helps avoid excessive false positives early in training
            with torch.no_grad():
                head[-2].bias.data.fill_(-1.0)
            self.detection_heads.append(head)
        
        # Background correction parameter
        # Subtracts background signal from concentration estimates
        self.register_buffer('background_level', torch.tensor([background_level]))
        
        # Confidence interval calibration parameter
        # Scales uncertainty estimates to achieve desired CI coverage
        self.register_buffer('calibration', torch.ones(1))
        
        # Initialize mu_head bias to predict low values initially
        # This conservative bias helps reduce false positives early in training
        with torch.no_grad():
            self.mu_head[-2].bias.data.fill_(-2.0)
        
    def forward(self, marker_values, coverage):
        """
        Forward pass through the model.
        
        Args:
            marker_values: Tensor of shape [batch_size, num_markers] with methylation values
            coverage: Tensor of shape [batch_size, num_markers] with read coverage
            
        Returns:
            mu: Estimated cell type concentration
            uncertainty: Uncertainty in the estimation
            detection_probs: Probability of exceeding each detection threshold
            attention_weights: Learned importance weights for each marker
        """
        B, M = marker_values.shape
        
        # Create mask for missing/unreliable values
        # Zero coverage markers are completely ignored
        mask_missing = (coverage == 0)  # [B, M]
        # Low coverage markers are considered unreliable
        mask_low_cov = (coverage < self.min_reliable_coverage)  # [B, M]
        # Combined mask identifies all markers to be treated cautiously
        mask = mask_missing | mask_low_cov  # [B, M]
        
        # Transform coverage to log space for better numerical stability
        # Log transformation helps handle the wide range of coverage values
        log_coverage = torch.log1p(coverage).unsqueeze(-1)  # [B, M, 1]
        
        # Calculate reliability weights based on coverage
        # Higher coverage markers get higher reliability
        reliability_weight = self.reliability_weight(log_coverage)  # [B, M, 1]
        
        # Apply quadratic scaling for sharper dropoff below threshold
        # This creates a non-linear boundary that heavily downweights low coverage
        coverage_reliability = torch.pow(
            torch.clamp(coverage.unsqueeze(-1) / self.min_reliable_coverage, 0.0, 1.0), 
            2  # Quadratic power creates sharper falloff
        )  # [B, M, 1]
        
        # Combine learned reliability and coverage-based reliability
        reliability_weight = reliability_weight * coverage_reliability  # [B, M, 1]
        
        # Handle NaN values in marker_values (replacing with zeros)
        # NaNs typically come from markers with zero coverage
        marker_values = torch.nan_to_num(marker_values, nan=0.0)  # [B, M]
        
        # Embed marker values and coverage separately
        # This allows different representations for these different data types
        value_features = self.value_embedding(marker_values.unsqueeze(-1))  # [B, M, feature_dim//2]
        coverage_features = self.coverage_embedding(log_coverage)  # [B, M, feature_dim//2]
        
        # Combine and project features
        features = torch.cat([value_features, coverage_features], dim=-1)  # [B, M, feature_dim]
        features = self.feature_projection(features)  # [B, M, feature_dim]
        
        # Apply transformer with masking
        # The transformer learns interactions between markers
        # Masked markers are ignored by the self-attention mechanism
        transformer_output = self.transformer_encoder(
            features, 
            src_key_padding_mask=mask
        )  # [B, M, feature_dim]
        
        # Calculate attention scores for each marker
        # These scores determine how much each marker contributes to the final prediction
        attention_scores = self.attention(transformer_output).squeeze(-1)  # [B, M]
        
        # Apply reliability weights to attention scores
        # This ensures low-coverage markers have minimal influence
        attention_scores = attention_scores * reliability_weight.squeeze(-1)  # [B, M]
        
        # Set masked positions to large negative values
        # This ensures zero attention weight for unreliable markers
        attention_scores = attention_scores.masked_fill(mask, -1e9)  # [B, M]
        
        # Apply softmax to get normalized attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, M]
        
        # Compute weighted sum of features using attention weights
        # This creates a single feature vector representing the sample
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)  # [B, feature_dim]
        
        # Predict cell type concentration (mu)
        mu = self.mu_head(aggregated)  # [B, 1]
        
        # Apply background correction
        # Subtracts the estimated background signal from predictions
        mu = torch.clamp(mu - self.background_level, min=0.0)  # [B, 1]
        
        # Predict uncertainty in the concentration estimate
        uncertainty = self.uncertainty_head(aggregated)  # [B, 1]
        
        # Create enhanced features for detection heads
        # Includes both the aggregated features and the uncertainty estimate
        detection_features = torch.cat([aggregated, uncertainty], dim=1)  # [B, feature_dim+1]
        
        # Get detection probabilities for each threshold
        # Each head predicts whether concentration exceeds its threshold
        detection_probs = [head(detection_features) for head in self.detection_heads]  # List of [B, 1]
        
        return mu, uncertainty, detection_probs, attention_weights
    
    def compute_loss(self, mu, uncertainty, y_true, control_mask=None):
        """
        Compute the loss function for training.
        
        Args:
            mu: Predicted concentration values [batch_size, 1]
            uncertainty: Predicted uncertainty values [batch_size, 1]
            y_true: Ground truth concentration values [batch_size, 1]
            control_mask: Optional boolean mask identifying control samples
            
        Returns:
            total_loss: Combined loss value for optimization
        """
        # Basic MSE loss between predictions and ground truth
        mse_loss = F.mse_loss(mu, y_true, reduction='none')  # [batch_size, 1]
        
        # Apply focal weighting to focus more on lower concentration samples
        # This addresses class imbalance where high concentration samples are rare
        weight = 1.0 + 2.0 * (1.0 - y_true)  # [batch_size, 1]
        weighted_loss = (mse_loss * weight).mean()  # scalar
        
        # Add penalty for predicting non-zero concentration for zero-concentration samples
        # This helps avoid false positives for healthy samples
        epsilon = 1e-6
        zero_penalty = 0.0
        if (y_true < epsilon).sum() > 0:
            zero_penalty = 2.0 * (mu * (y_true < epsilon).float()).mean()  # scalar
        
        # Add penalty for control samples if provided
        # Control samples should have zero concentration
        control_loss = 0.0
        if control_mask is not None and control_mask.sum() > 0:
            control_loss = 10.0 * torch.mean(mu[control_mask])  # scalar
        
        # Combine all loss components
        total_loss = weighted_loss + zero_penalty + control_loss  # scalar
        
        return total_loss
    
    def get_estimate_and_ci(self, mu, uncertainty, ci_level=0.95):
        """
        Get point estimate and confidence interval for concentration.
        
        Args:
            mu: Predicted concentration values [batch_size, 1]
            uncertainty: Predicted uncertainty values [batch_size, 1]
            ci_level: Confidence interval level (default: 0.95 for 95% CI)
            
        Returns:
            mu: Point estimate of concentration
            ci: Lower and upper bounds of confidence interval [batch_size, 2]
            scaled_uncertainty: Calibrated uncertainty values
        """
        # Scale uncertainty by calibration factor
        # This ensures proper coverage of the confidence interval
        scaled_uncertainty = uncertainty * self.calibration  # [batch_size, 1]
        
        # Calculate z-score for desired confidence level
        # e.g., z_score=1.96 for 95% CI
        z_score = stats.norm.ppf((1 + ci_level) / 2)  # scalar
        
        # Calculate lower bound of CI, clipped to valid range
        lower = torch.clamp(mu - z_score * scaled_uncertainty, min=0.0)  # [batch_size, 1]
        
        # Calculate upper bound of CI, clipped to valid range
        upper = torch.clamp(mu + z_score * scaled_uncertainty, max=1.0)  # [batch_size, 1]
        
        # Combine into single tensor
        ci = torch.cat([lower, upper], dim=1)  # [batch_size, 2]
        
        return mu, ci, scaled_uncertainty
    
    def calibrate(self, val_loader, control_loader=None, device='cpu'):
        """
        Calibrate model confidence intervals and background level.
        
        Args:
            val_loader: DataLoader with validation data
            control_loader: Optional DataLoader with control samples
            device: Device to run calibration on ('cpu' or 'cuda')
            
        Returns:
            dict: Calibration results with calibration_factor and background_level
        """
        self.eval()
        
        # 1. Calibrate confidence intervals using validation data
        all_errors = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Handle both 3-element and 4-element returns
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask
                else:
                    marker_values, coverage, y_true = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                # Get model predictions
                mu, uncertainty, _, _ = self(marker_values, coverage)
                
                # Calculate absolute errors
                errors = torch.abs(mu - y_true)
                
                # Store results for later analysis
                all_errors.append(errors.cpu())
                all_uncertainties.append(uncertainty.cpu())
        
        # Concatenate collected data
        all_errors = torch.cat(all_errors)  # [total_samples, 1]
        all_uncertainties = torch.cat(all_uncertainties)  # [total_samples, 1]
        
        # Calculate calibration factor based on empirical error distribution
        # We aim for 95% of errors to be within the predicted uncertainty range
        error_95_percentile = torch.quantile(all_errors, 0.95)  # scalar
        avg_uncertainty = all_uncertainties.mean()  # scalar
        
        # Scale factor to achieve desired CI coverage
        # For 95% CI with normal distribution, z-score = 1.96
        calibration_factor = error_95_percentile / (1.96 * avg_uncertainty)  # scalar
        
        # 2. Calibrate background level using control samples if provided
        bg_level = self.background_level.item()  # Start with current value
        
        if control_loader is not None:
            all_control_preds = []
            
            with torch.no_grad():
                for batch_data in control_loader:
                    if len(batch_data) == 4:
                        marker_values, coverage, _, _ = batch_data
                    else:
                        marker_values, coverage, _ = batch_data
                    
                    marker_values = marker_values.to(device)
                    coverage = coverage.to(device)
                    
                    # Get predictions for control samples
                    mu, _, _, _ = self(marker_values, coverage)
                    all_control_preds.append(mu.cpu())
            
            # Use 95th percentile for conservative background correction
            # This helps avoid false positives in control samples
            all_control_preds = torch.cat(all_control_preds)  # [total_control_samples, 1]
            bg_level = float(torch.quantile(all_control_preds, 0.95))  # scalar
            
            # Ensure minimum background level for stability
            bg_level = max(bg_level, 0.05)
        
        # Apply calibration parameters to model
        with torch.no_grad():
            self.calibration.copy_(torch.tensor([calibration_factor]))
            self.background_level.copy_(torch.tensor([bg_level]))
        
        # Return calibration results
        return {
            'calibration_factor': calibration_factor.item(),
            'background_level': bg_level
        }
        
    def get_binary_prediction(self, mu, detection_probs, threshold_idx=1):
        """
        Get binary prediction for cancer detection
        
        Args:
            mu: Predicted concentration
            detection_probs: List of detection probabilities from forward pass
            threshold_idx: Which threshold to use (default: 1, which is 0.01 or 1%)
        
        Returns:
            Binary prediction (1 = cancer detected, 0 = no cancer detected)
        """
        # Convert predictions to detection score
        # Use both concentration estimate and detection head
        # This combines both approaches for more robust detection
        detection_score = detection_probs[threshold_idx]
        
        # Threshold for positive detection (can be calibrated)
        detection_threshold = 0.5
        
        return (detection_score >= detection_threshold).float()


class MarkerImportanceAnalyser:
    """
    Utility class to analyse the importance of different markers
    """
    def __init__(self, model):
        self.model = model
    
    def get_marker_importance(self, dataloader, top_k=20):
        """
        Analyse marker importance across the dataset
        """
        self.model.eval()
        all_attentions = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Handle both 3-element and 4-element returns
                if len(batch_data) == 4:
                    marker_values, coverage, _, _ = batch_data  # Unpack 4 elements, ignore y_true and control_mask
                else:
                    marker_values, coverage, _ = batch_data  # Unpack 3 elements, ignore y_true
                _, _, _, attention_weights = self.model(marker_values, coverage)
                all_attentions.append(attention_weights)
        
        # Average attention weights across batches
        avg_attention = torch.cat(all_attentions, dim=0).mean(dim=0)  # [M]
        
        # Get top-k markers by attention weight
        top_k_indices = torch.topk(avg_attention, k=min(top_k, len(avg_attention))).indices
        top_k_weights = avg_attention[top_k_indices]
        
        return top_k_indices.cpu().numpy(), top_k_weights.cpu().numpy()
    
    def analyse_detection_performance(self, dataloader, thresholds=None):
        """
        Analyse detection performance at different concentration thresholds
        """
        model_thresholds = self.model.detection_thresholds
                
        if thresholds is None:
            thresholds = model_thresholds
        
        self.model.eval()
        predictions = []
        ground_truth = []
        detection_probs = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Handle both 3-element and 4-element returns
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data  # Unpack 4 elements, ignore control_mask
                else:
                    marker_values, coverage, y_true = batch_data
                mu, _, det_probs, _ = self.model(marker_values, coverage)
                
                # Make sure to convert tensors to numpy arrays consistently
                predictions.append(mu.cpu().numpy())
                ground_truth.append(y_true.cpu().numpy())
                detection_probs.append([dp.cpu().numpy() for dp in det_probs])
        
        # Concatenate results - ensure these are numpy arrays, not lists
        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth).flatten()  # Make sure it's flattened
        detection_probs = [np.concatenate([dp[i] for dp in detection_probs]) for i in range(len(model_thresholds))]
        
        # Calculate detection metrics for each threshold
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        
        results = {}
        for i, threshold in enumerate(thresholds):
            if i >= len(detection_probs):
                continue  # Skip if threshold not in model_thresholds
                
            # Binary ground truth - ensure ground_truth is a numpy array
            y_binary = (ground_truth >= threshold).astype(int)
            
            # ROC curve and AUC
            fpr, tpr, roc_thresholds = roc_curve(y_binary, detection_probs[i])
            roc_auc = auc(fpr, tpr)
            
            # Find sensitivity at 95% specificity (5% FPR)
            idx_95spec = np.argmin(np.abs(fpr - 0.05))
            sens_at_95spec = tpr[idx_95spec]
            threshold_at_95spec = roc_thresholds[idx_95spec]
            
            precision, recall, pr_thresholds = precision_recall_curve(y_binary, detection_probs[i])
            ap = average_precision_score(y_binary, detection_probs[i])

            results[threshold] = {
                'auc': roc_auc,
                'sensitivity_at_95spec': sens_at_95spec,
                'threshold_at_95spec': threshold_at_95spec,
                'average_precision': ap,
                'precision_recall_data': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
                }
            }
        
        return results