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
    Set Transformer model cell type concentration estimation from cfDNA methylation data
    
    Provides permutation-invariant processing of marker data with specialized
    components for accurate concentration estimation and detection.
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
            y_true: Ground truth concentration (optional) [batch_size, 1]
                
        Returns:
            mu: Estimated concentration [batch_size, 1]
            phi: Concentration parameter for Beta distribution [batch_size, 1]
            detection_probs: List of detection probabilities for each threshold [batch_size, 1]
            attention_weights: Attention weights for markers [batch_size, num_markers]
                
            If y_true is provided, also returns y_true for loss calculation
        """
        B, M = marker_values.shape
        
        # Create mask for missing values (where coverage = 0)
        mask = (coverage == 0)  # [B, M]
        
        # Replace NaN values in marker_values only where coverage > 0
        # For positions with coverage=0, set to 0 as they'll be masked anyway
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
        
        # Check if any sample has all markers masked
        all_masked = mask.all(dim=1)  # [B]
        
        # If any sample has all markers masked, create a special token
        if all_masked.any():
            # Create a learned representation for samples with no valid markers
            default_rep = torch.zeros_like(features[0:1])
            # Apply it to samples where all markers are masked
            for i in range(B):
                if all_masked[i]:
                    features[i] = default_rep
        
        # Process through main encoder blocks
        x = features
        for encoder_block in self.encoder_blocks:
            x_out, _ = encoder_block(x, mask)
            # Ensure no NaN propagated through the encoder
            if torch.isnan(x_out).any():
                # Keep previous non-NaN values if NaNs appear
                x_out = torch.where(torch.isnan(x_out), x, x_out)
            x = x_out
        
        # Process through low concentration encoder blocks
        low_x = features  
        for encoder_block in self.low_conc_encoder_blocks:
            low_x_out, _ = encoder_block(low_x, mask)
            # Ensure no NaN propagated through the encoder
            if torch.isnan(low_x_out).any():
                # Keep previous non-NaN values if NaNs appear
                low_x_out = torch.where(torch.isnan(low_x_out), low_x, low_x_out)
            low_x = low_x_out
        
        # For samples with all markers masked, use a learned representation
        if all_masked.any():
            for i in range(B):
                if all_masked[i]:
                    # Create identity mapping attention since there's nothing to attend to
                    main_attention_weights_i = torch.zeros(self.num_inds, M, device=x.device)
                    # Just use the default representation directly
                    pooled_i = torch.zeros(self.num_inds, x.size(-1), device=x.device)
                    
                    # Handle these samples separately
                    if 'pooled' not in locals():
                        # First initialization
                        pooled = torch.zeros(B, self.num_inds, x.size(-1), device=x.device)
                        main_attention_weights = torch.zeros(B, self.num_inds, M, device=x.device)
                    
                    pooled[i] = pooled_i
                    main_attention_weights[i] = main_attention_weights_i
        
        # Apply pooling for samples with at least one valid marker
        if 'pooled' not in locals():
            # No samples had all markers masked, do normal pooling
            pooled, main_attention_weights = self.main_pooling(x, mask)
            low_pooled, _ = self.low_conc_pooling(low_x, mask)
        else:
            # Some samples had all markers masked, handle remaining samples
            valid_indices = ~all_masked
            if valid_indices.any():
                # Pool only for samples with at least one valid marker
                valid_x = x[valid_indices]
                valid_low_x = low_x[valid_indices]
                valid_mask = mask[valid_indices]
                
                valid_pooled, valid_main_weights = self.main_pooling(valid_x, valid_mask)
                valid_low_pooled, _ = self.low_conc_pooling(valid_low_x, valid_mask)
                
                # Update the tensor for valid samples
                pooled[valid_indices] = valid_pooled
                main_attention_weights[valid_indices] = valid_main_weights
                
                # Initialize low_pooled if not done
                if 'low_pooled' not in locals():
                    low_pooled = torch.zeros(B, self.num_inds, x.size(-1), device=x.device)
                
                low_pooled[valid_indices] = valid_low_pooled
            else:
                # All samples had all markers masked
                low_pooled = torch.zeros_like(pooled)
        
        # Ensure no NaN in pooled representations
        pooled = torch.nan_to_num(pooled, nan=0.0)
        low_pooled = torch.nan_to_num(low_pooled, nan=0.0)
        
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
        
        # Final check for NaN values
        blended_mu = torch.nan_to_num(blended_mu, nan=0.5)
        blended_phi = torch.nan_to_num(blended_phi, nan=1.0)
        
        # Calculate uncertainty for detection heads
        _, _, uncertainty = self.get_estimate_and_ci(blended_mu, blended_phi)
        
        # Extract marker-level attention weights
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
        # Simple MSE loss for debugging
        mse_loss = F.mse_loss(mu, y_true, reduction='none')
        
        # Minimal focal weighting
        focal_weight = torch.exp(-y_true * self.focal_weight_factor) + 1.0
        focal_weight = torch.clamp(focal_weight, 1.0, 5.0)
        
        # Apply weighting
        weighted_loss = mse_loss * focal_weight
        return weighted_loss.mean()
    
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
                        lower[i] = max(0.0, mu[i].item() - 0.1)
                        upper[i] = min(1.0, mu[i].item() + 0.1)
                except:
                    # Fallback if scipy calculation fails
                    lower[i] = max(0.0, mu[i].item() - 0.1)
                    upper[i] = min(1.0, mu[i].item() + 0.1)
        
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
            for marker_values, coverage, y_true in dataloader:
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