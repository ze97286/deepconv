"""
Set Transformer implementation for cfDNA cancer detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import numpy as np
from scipy import stats
import math


class MultiheadAttentionBlock(nn.Module):
    """
    Multihead Attention Block (MAB) for Set Transformer
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y, mask=None):
        """
        Args:
            x: Query tensor [B, seq_len_q, dim]
            y: Key/Value tensor [B, seq_len_kv, dim]
            mask: Boolean mask for y [B, seq_len_kv]
        """
        attention_mask = None if mask is None else ~mask
        x_norm = self.ln1(x)
        y_norm = self.ln1(y)
        attention_output, attention_weights = self.attention(x_norm, y_norm, y_norm, key_padding_mask=attention_mask)
        x = x + self.dropout(attention_output)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x, attention_weights


class SetAttentionBlock(nn.Module):
    """
    Set Attention Block (SAB) for Set Transformer
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.mab = MultiheadAttentionBlock(dim, num_heads, dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [B, seq_len, dim]
            mask: Boolean mask [B, seq_len]
        """
        return self.mab(x, x, mask)


class PoolingByMultiheadAttention(nn.Module):
    """
    Pooling by Multihead Attention (PMA) for Set Transformer
    """
    def __init__(self, dim, num_heads, num_inds, dropout=0.1):
        super().__init__()
        self.inds = nn.Parameter(torch.randn(1, num_inds, dim))
        self.mab = MultiheadAttentionBlock(dim, num_heads, dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [B, seq_len, dim]
            mask: Boolean mask [B, seq_len]
        """
        batch_size = x.size(0)
        inds = self.inds.repeat(batch_size, 1, 1)
        pooled, attention_weights = self.mab(inds, x, mask)
        return pooled, attention_weights


class SetTransformerCancerDetection(nn.Module):
    """
    Set Transformer model for cancer detection from cfDNA methylation data
    
    Provides permutation-invariant processing of marker data with specialized
    components for accurate cancer concentration estimation and detection.
    """
    def __init__(self, num_markers, feature_dim=128, num_heads=8, num_inds=8, 
                 num_encoder_blocks=2, dropout_rate=0.2, 
                 detection_thresholds=(0.001, 0.01, 0.05),
                 focal_weight_factor=100, low_concentration_threshold=0.01):
        super().__init__()
        
        # Store configuration
        self.detection_thresholds = detection_thresholds
        self.num_markers = num_markers
        self.focal_weight_factor = focal_weight_factor
        self.low_concentration_threshold = low_concentration_threshold
        self.num_inds = num_inds
        
        # Initial embedding for marker values and coverage
        self.marker_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
        
        # Main encoder (Set Attention Blocks)
        self.encoder_blocks = nn.ModuleList([
            SetAttentionBlock(feature_dim, num_heads, dropout_rate) 
            for _ in range(num_encoder_blocks)
        ])
        
        # Low concentration specialist encoder
        self.low_conc_encoder_blocks = nn.ModuleList([
            SetAttentionBlock(feature_dim, num_heads, dropout_rate) 
            for _ in range(2)  # Smaller network for low concentration focus
        ])
        
        # Pooling mechanisms
        self.main_pooling = PoolingByMultiheadAttention(
            feature_dim, num_heads, num_inds, dropout_rate
        )
        
        self.low_conc_pooling = PoolingByMultiheadAttention(
            feature_dim, num_heads, num_inds, dropout_rate
        )
        
        # Main prediction components for concentration estimation
        self.mu_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.phi_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()  # Ensures positive concentration parameter
        )
        
        # Low concentration specialist head
        self.low_mu_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.low_phi_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()
        )
        
        # Binary detection heads for each threshold with enhanced features
        self.detection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + 1, feature_dim),  # +1 for uncertainty feature
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 1),
                nn.Sigmoid()
            ) for _ in detection_thresholds
        ])
        
        # Calibration components
        self.calibration = nn.Parameter(torch.ones(1))
        self.low_calibration = nn.Parameter(torch.ones(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, marker_values, coverage, y_true=None):
        """
        Forward pass through the Set Transformer Cancer Detection model
        
        Args:
            marker_values: Marker methylation values [batch_size, num_markers]
            coverage: Coverage values for each marker [batch_size, num_markers]
            y_true: Ground truth cancer concentration (optional) [batch_size, 1]
            
        Returns:
            mu: Estimated cancer concentration [batch_size, 1]
            phi: Concentration parameter for Beta distribution [batch_size, 1]
            detection_probs: List of detection probabilities for each threshold [batch_size, 1]
            attention_weights: Attention weights for markers [batch_size, num_markers]
            
            If y_true is provided, also returns y_true for loss calculation
        """
        B, M = marker_values.shape
        
        # Create mask for missing values (where coverage = 0)
        mask = (coverage == 0)  # [B, M]
        
        # Handle NaN values in marker_values by replacing only when coverage > 0
        # For markers with coverage = 0, the value doesn't matter as they'll be masked
        marker_values_processed = torch.where(mask, torch.zeros_like(marker_values), marker_values)
        marker_values_processed = torch.nan_to_num(marker_values_processed, nan=0.5)
        
        # Embed marker values and coverage separately
        value_features = self.marker_embedding(marker_values_processed.unsqueeze(-1))  # [B, M, feature_dim//2]
        coverage_features = self.coverage_embedding(
            torch.log1p(coverage).unsqueeze(-1)  # Log transform for better numerical stability
        )  # [B, M, feature_dim//2]
        
        # Combine features
        features = torch.cat([value_features, coverage_features], dim=-1)  # [B, M, feature_dim]
        features = self.feature_projection(features)  # [B, M, feature_dim]
        
        # Create attention mask (False = keep, True = mask out)
        attention_mask = mask  # [B, M]
        
        # Process through main encoder blocks
        x = features
        for encoder_block in self.encoder_blocks:
            x, _ = encoder_block(x, attention_mask)
        
        # Process through low concentration encoder blocks
        low_x = features  
        for encoder_block in self.low_conc_encoder_blocks:
            low_x, _ = encoder_block(low_x, attention_mask)
        
        # Apply pooling to get fixed-size representations
        pooled, main_attention_weights = self.main_pooling(x, attention_mask)  # [B, num_inds, feature_dim]
        low_pooled, _ = self.low_conc_pooling(low_x, attention_mask)  # [B, num_inds, feature_dim]
        
        # Average across inducing points
        main_features = pooled.mean(dim=1)  # [B, feature_dim]
        main_features = self.dropout(main_features)
        
        low_features = low_pooled.mean(dim=1)  # [B, feature_dim]
        low_features = self.dropout(low_features)
        
        # Predict parameters for Beta distribution from main encoder
        mu = self.mu_head(main_features)  # [B, 1]
        phi = self.phi_head(main_features) * self.calibration  # [B, 1], calibrated concentration
        
        # Predict parameters from low concentration encoder
        low_mu = self.low_mu_head(low_features)  # [B, 1]
        low_phi = self.low_phi_head(low_features) * self.low_calibration  # [B, 1]
        
        # Add safeguards for phi
        phi = torch.clamp(phi, min=1.0)  # Ensure phi is at least 1.0
        low_phi = torch.clamp(low_phi, min=1.0)  # Ensure low_phi is at least 1.0
        
        # Blend predictions based on predicted concentration
        # More weight to low_mu for low concentrations
        with torch.no_grad():
            blend_weight = torch.exp(-mu * 200)  # Weight decreases as concentration increases
        
        blended_mu = blend_weight * low_mu + (1 - blend_weight) * mu
        blended_phi = blend_weight * low_phi + (1 - blend_weight) * phi
        
        # Calculate uncertainty for detection heads
        _, _, uncertainty = self.get_estimate_and_ci(blended_mu, blended_phi)
        
        # Extract marker-level attention weights (average across heads and inducing points)
        # For visualization/interpretation of which markers are important
        # Shape of main_attention_weights from pooling layer: [batch, num_inds, num_markers]
        marker_attention = main_attention_weights.mean(dim=1)  # [B, num_markers]
        
        # Enhanced features for detection heads (including uncertainty)
        detection_features = torch.cat([main_features, uncertainty], dim=1)
        
        # Get detection probabilities for each threshold
        detection_probs_raw = [head(detection_features) for head in self.detection_heads]
        
        # Ensure all detection probabilities are properly bounded between 0 and 1
        detection_probs = [torch.clamp(dp, 0.0, 1.0) for dp in detection_probs_raw]
            
        if y_true is not None:
            return blended_mu, blended_phi, detection_probs, marker_attention, y_true
        
        return blended_mu, blended_phi, detection_probs, marker_attention
    
    def compute_loss(self, mu, phi, y_true, epsilon=1e-6):
        """
        Compute enhanced focal Beta negative log likelihood loss with threshold emphasis
        
        Args:
            mu: Predicted mean (concentration) [batch_size, 1]
            phi: Precision parameter [batch_size, 1]
            y_true: Ground truth concentration [batch_size, 1]
            epsilon: Small value for numerical stability
            
        Returns:
            Enhanced focal loss for optimization
        """
        y_clipped = torch.clamp(y_true, epsilon, 1 - epsilon)
        
        # Add safeguards for phi to ensure it's positive and not too small
        phi = torch.clamp(phi, min=1.0)  # Ensure phi is at least 1.0
        
        # Replace any NaN values in mu or phi
        mu = torch.nan_to_num(mu, nan=0.5)
        phi = torch.nan_to_num(phi, nan=1.0)
        
        # Calculate Beta distribution parameters with safeguards
        alpha = mu * phi  # [B, 1]
        beta = (1 - mu) * phi  # [B, 1]
        
        # Add safety margin to ensure alpha and beta are positive
        alpha = torch.clamp(alpha, min=epsilon)
        beta = torch.clamp(beta, min=epsilon)
        
        # Create Beta distribution with safety checks
        try:
            dist = Beta(alpha, beta)
            nll_loss = -dist.log_prob(y_clipped)
        except ValueError as e:
            # Fallback to MSE loss if Beta distribution fails
            print(f"Warning: Beta distribution failed, falling back to MSE loss. Error: {e}")
            print(f"mu range: {mu.min().item():.4f}-{mu.max().item():.4f}, phi range: {phi.min().item():.4f}-{phi.max().item():.4f}")
            nll_loss = F.mse_loss(mu, y_clipped, reduction='none')
        
        # Get focal weighting factor from args
        focal_factor = self.focal_weight_factor if hasattr(self, 'focal_weight_factor') else 100
        low_conc_threshold = self.low_concentration_threshold if hasattr(self, 'low_concentration_threshold') else 0.01
        
        # Enhanced focal weighting with configurable factor
        base_weight = torch.exp(-y_true * focal_factor) + 1.0
        
        # Additional weight for samples near thresholds
        threshold_weight = torch.zeros_like(y_true)
        for threshold in self.detection_thresholds:
            if threshold > 0:
                relative_distance = torch.abs(y_true - threshold) / max(threshold, epsilon)
                threshold_weight += torch.exp(-relative_distance * 5) * 2.0
        
        # Special handling for values below the low concentration threshold
        is_low_conc = (y_true <= low_conc_threshold).float()
        is_zero = (y_true < epsilon).float()
        
        # Extra weight for low but non-zero concentrations
        low_conc_weight = is_low_conc * (1 - is_zero) * 2.0
        
        # Combine weights (cap at 5x to prevent extreme values)
        focal_weight = torch.clamp(base_weight + threshold_weight + low_conc_weight, 1.0, 5.0)
        
        # Apply focal weighting
        focal_loss = nll_loss * focal_weight
        
        # Add regularization to prevent extremely confident predictions
        reg_loss = 0.01 * torch.abs(torch.log(torch.clamp(phi, min=epsilon))).mean()
        
        return focal_loss.mean() + reg_loss
    
    def get_estimate_and_ci(self, mu, phi, ci_level=0.95):
        """
        Get point estimate and confidence interval using scipy's beta ppf
        
        Args:
            mu: Predicted mean [batch_size, 1]
            phi: Precision parameter [batch_size, 1]
            ci_level: Confidence interval level (default: 0.95 for 95% CI)
            
        Returns:
            estimate: Point estimate [batch_size, 1]
            ci: Confidence interval bounds [batch_size, 2]
            uncertainty: Width of confidence interval [batch_size, 1]
        """
        # Add safeguards for phi
        phi = torch.clamp(phi, min=1.0)
        
        # Handle NaN values
        mu = torch.nan_to_num(mu, nan=0.5)
        phi = torch.nan_to_num(phi, nan=1.0)
        
        # Calculate parameters
        alpha = mu * phi
        beta = (1 - mu) * phi
        
        # Add safety margin
        alpha = torch.clamp(alpha, min=0.01)
        beta = torch.clamp(beta, min=0.01)
        
        # Move tensors to CPU and convert to numpy for scipy
        alpha_np = alpha.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        
        # Initialize tensors for results
        lower = torch.zeros_like(mu)
        upper = torch.zeros_like(mu)
        
        # Calculate CI bounds for each sample using scipy's beta ppf
        for i in range(len(alpha_np)):
            a_val = float(alpha_np[i])
            b_val = float(beta_np[i])
            
            # Handle potential numerical issues
            if a_val <= 0 or b_val <= 0 or np.isnan(a_val) or np.isnan(b_val):
                lower[i] = 0.0
                upper[i] = 1.0
            else:
                try:
                    # Use scipy.stats.beta for ppf (percent point function/quantile)
                    from scipy import stats
                    lower[i] = torch.tensor(stats.beta.ppf((1 - ci_level) / 2, a_val, b_val))
                    upper[i] = torch.tensor(stats.beta.ppf(1 - (1 - ci_level) / 2, a_val, b_val))
                    
                    # Handle any NaN results from scipy
                    if torch.isnan(lower[i]) or torch.isnan(upper[i]):
                        lower[i] = max(0.0, mu[i] - 0.1)
                        upper[i] = min(1.0, mu[i] + 0.1)
                except:
                    # Fallback if scipy calculation fails
                    lower[i] = max(0.0, mu[i] - 0.1)
                    upper[i] = min(1.0, mu[i] + 0.1)
        
        # Ensure bounds are valid
        lower = torch.clamp(lower, 0.0, 0.99)
        upper = torch.clamp(upper, 0.01, 1.0)
        
        # Move tensors back to the device of the input
        lower = lower.to(mu.device)
        upper = upper.to(mu.device)
        
        estimate = mu
        ci = torch.cat([lower, upper], dim=1)
        
        # Calculate uncertainty (width of CI)
        uncertainty = upper - lower
        
        return estimate, ci, uncertainty

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
    
def calculate_loss(model, mu, phi, detection_probs, y_true, args):
        """
        Calculate combined loss using configurable parameters
        
        Args:
            model: The model instance
            mu: Predicted mean (concentration)
            phi: Precision parameter
            detection_probs: List of detection probabilities for each threshold
            y_true: Ground truth concentration
            args: Arguments including detection thresholds and weights
            
        Returns:
            total_loss: Combined loss for optimization
            concentration_loss: Loss component for concentration estimation
            detection_loss: Loss component for binary detection
        """
        # Get concentration loss
        concentration_loss = model.compute_loss(mu, phi, y_true)
        
        # Calculate detection losses for each threshold
        detection_losses = []
        for i, threshold in enumerate(args.detection_thresholds):
            # Convert continuous concentration to binary label
            binary_y = (y_true >= threshold).float()
            
            # Ensure detection probabilities are properly bounded
            det_probs = torch.clamp(detection_probs[i], 0.0, 1.0)
            
            # Binary cross-entropy loss
            det_loss = F.binary_cross_entropy(det_probs, binary_y)
            detection_losses.append(det_loss)
        
        # Use configurable detection loss weight
        detection_loss_weight = args.detection_loss_weight if hasattr(args, 'detection_loss_weight') else 1.0
        combined_detection_loss = sum(detection_losses) / len(detection_losses)
        
        # Calculate total loss
        total_loss = concentration_loss + detection_loss_weight * combined_detection_loss
        
        return total_loss, concentration_loss, combined_detection_loss

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
            for marker_values, coverage, _ in dataloader:
                if isinstance(self.model, CancerDetectionEnsemble):
                    # For ensemble, use the first model for attention
                    _, _, _, attention_weights = self.model.models[0](marker_values, coverage)
                else:
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
        if isinstance(self.model, CancerDetectionEnsemble):
            model_thresholds = self.model.models[0].detection_thresholds
        else:
            model_thresholds = self.model.detection_thresholds
            
        if thresholds is None:
            thresholds = model_thresholds
        
        self.model.eval()
        predictions = []
        ground_truth = []
        detection_probs = []
        
        with torch.no_grad():
            for marker_values, coverage, y_true in dataloader:
                if isinstance(self.model, CancerDetectionEnsemble):
                    mu, _, det_probs = self.model.forward(marker_values, coverage)
                else:
                    mu, _, det_probs, _ = self.model(marker_values, coverage)
                
                predictions.append(mu.cpu().numpy())
                ground_truth.append(y_true.cpu().numpy())
                detection_probs.append([dp.cpu().numpy() for dp in det_probs])
        
        # Concatenate results
        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)
        detection_probs = [np.concatenate([dp[i] for dp in detection_probs]) for i in range(len(model_thresholds))]
        
        # Calculate detection metrics for each threshold
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        
        results = {}
        for i, threshold in enumerate(thresholds):
            if i >= len(detection_probs):
                continue  # Skip if threshold not in model_thresholds
                
            # Binary ground truth
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