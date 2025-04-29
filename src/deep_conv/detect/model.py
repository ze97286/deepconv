import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.stats as stats
import numpy as np

class DynamicBackgroundCorrection(nn.Module):
    """
    Dynamic background correction module with more conservative settings
    """
    def __init__(self, feature_dim, min_bg=0.003, max_bg=0.04):
        super().__init__()
        self.min_bg = min_bg
        self.max_bg = max_bg
        
        # Network to predict sample-specific background level
        self.bg_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
        # Initialize to predict conservative background level
        with torch.no_grad():
            self.bg_network[-2].bias.data.fill_(-3.0)
    
    def forward(self, features):
        """
        Predict background level for each sample based on its features
        
        Args:
            features: Sample features [batch_size, feature_dim]
            
        Returns:
            bg_level: Predicted background level [batch_size, 1]
        """
        # Scale sigmoid output to desired background range
        bg_scale = self.bg_network(features)
        bg_level = self.min_bg + bg_scale * (self.max_bg - self.min_bg)
        
        return bg_level
    
class EnhancedCancerDetectionModel(nn.Module):
    """
    Enhanced deep learning model for cancer detection from cfDNA methylation markers.
    Uses transformer architecture with attention mechanisms and dynamic background
    correction to estimate cell type concentration accurately across all ranges.
    """
    def __init__(self, num_markers, feature_dim=96, num_heads=6, num_layers=2, 
                 dropout_rate=0.2, detection_thresholds=(0.001, 0.005, 0.01, 0.05),
                 min_reliable_coverage=5.0):
        """
        Initialize the EnhancedCancerDetectionModel.
        
        Args:
            num_markers: Number of methylation markers in input
            feature_dim: Dimension of feature representations
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer encoder layers
            dropout_rate: Dropout probability for regularization
            detection_thresholds: Concentration thresholds for binary detection
            min_reliable_coverage: Minimum coverage to consider a marker reliable
        """
        super().__init__()
        
        # Store configuration parameters
        self.detection_thresholds = detection_thresholds
        self.num_markers = num_markers
        self.min_reliable_coverage = min_reliable_coverage
            
        # Separate embeddings for marker values and coverage
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
        
        # Transformer encoder for learning marker interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention mechanism for marker importance weighting
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Coverage reliability weighting mechanism
        self.reliability_weight = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Concentration estimation head (mu)
        self.mu_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Dynamic background correction module
        self.bg_correction = DynamicBackgroundCorrection(feature_dim)
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()
        )
        
        # Binary detection heads for different concentration thresholds
        self.detection_heads = nn.ModuleList()
        for _ in detection_thresholds:
            head = nn.Sequential(
                nn.Linear(feature_dim + 1, feature_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            )
            self.detection_heads.append(head)
        
        # Confidence interval calibration parameter
        self.register_buffer('calibration', torch.ones(1))
        
        # Initialize mu_head bias to predict low values initially
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
        mask_missing = (coverage == 0)
        mask_low_cov = (coverage < self.min_reliable_coverage)
        mask = mask_missing | mask_low_cov
        
        # Calculate reliability weights
        log_coverage = torch.log1p(coverage).unsqueeze(-1)
        reliability_weight = self.reliability_weight(log_coverage)
        
        # Apply quadratic scaling for sharper dropoff below threshold
        coverage_reliability = torch.pow(
            torch.clamp(coverage.unsqueeze(-1) / self.min_reliable_coverage, 0.0, 1.0), 
            2
        )
        reliability_weight = reliability_weight * coverage_reliability
        
        # Handle NaN values
        marker_values = torch.nan_to_num(marker_values, nan=0.0)
        
        # Embed marker values and coverage
        value_features = self.value_embedding(marker_values.unsqueeze(-1))
        coverage_features = self.coverage_embedding(log_coverage)
        
        # Combine features
        features = torch.cat([value_features, coverage_features], dim=-1)
        features = self.feature_projection(features)
        
        # Apply transformer with masking
        transformer_output = self.transformer_encoder(
            features, 
            src_key_padding_mask=mask
        )
        
        # Apply attention with reliability weighting
        attention_scores = self.attention(transformer_output).squeeze(-1)
        attention_scores = attention_scores * reliability_weight.squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute weighted sum of features
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)
        
        # Predict raw concentration
        mu = self.mu_head(aggregated)
        
        # Apply dynamic background correction
        bg_level = self.bg_correction(aggregated)
        mu_corrected = torch.clamp(mu - bg_level, min=0.0)
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(aggregated)
        
        # Get detection probabilities
        detection_features = torch.cat([aggregated, uncertainty], dim=1)
        detection_probs = [head(detection_features) for head in self.detection_heads]
        
        return mu_corrected, uncertainty, detection_probs, attention_weights
    
    def compute_loss(self, mu, uncertainty, y_true, control_mask=None):
        """
        Compute concentration-focused loss with modest range-specific weighting
        
        Args:
            mu: Predicted concentration values [batch_size, 1]
            uncertainty: Predicted uncertainty values [batch_size, 1]
            y_true: Ground truth concentration values [batch_size, 1]
            control_mask: Optional boolean mask identifying control samples
            
        Returns:
            total_loss: Combined loss value for optimization
        """
        # Basic MSE loss
        mse_loss = F.mse_loss(mu, y_true, reduction='none')
        
        # Calculate relative error for non-zero targets
        epsilon = 1e-6
        non_zero_mask = (y_true > epsilon)
        
        # Initialize relative error tensor
        rel_error = torch.zeros_like(mse_loss)
        
        # Compute relative error only for non-zero targets
        if non_zero_mask.sum() > 0:
            rel_error[non_zero_mask] = torch.abs(mu[non_zero_mask] - y_true[non_zero_mask]) / (y_true[non_zero_mask] + epsilon)
        
        # Create range-specific weights - modest 1.5x for critical range
        critical_range_mask = (y_true >= 0.001) & (y_true < 0.01)
        
        # Apply log-scale weighting - but adjust it for critical range
        log_weights = 1.0 / torch.log10(y_true * 1000 + 10.0)
        log_weights = torch.clamp(log_weights, 0.5, 2.0)
        
        # Apply modest 1.5x for critical range
        log_weights = torch.where(critical_range_mask, log_weights * 1.5, log_weights)
        
        # Zero-concentration specific penalty
        zero_mask = (y_true < epsilon)
        zero_penalty = 10.0 * mu[zero_mask].mean() if zero_mask.sum() > 0 else 0.0
        
        # Control sample penalty
        control_loss = 0.0
        if control_mask is not None and control_mask.sum() > 0:
            control_loss = 15.0 * mu[control_mask].mean()
        
        # Apply weights to MSE and relative error
        weighted_mse = (mse_loss * log_weights).mean()
        weighted_rel = (rel_error * log_weights).mean() if non_zero_mask.sum() > 0 else 0.0
        
        # Add calibration component
        calibration_loss = 0.0
        if uncertainty is not None:
            z_scores = torch.abs(mu - y_true) / (uncertainty + 1e-6)
            calibration_loss = F.smooth_l1_loss(z_scores, torch.ones_like(z_scores) * 1.96)
        
        # Combine all components - slight increase in relative error weight
        total_loss = weighted_mse + 0.6 * weighted_rel + zero_penalty + control_loss + 0.1 * calibration_loss
        
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
        scaled_uncertainty = uncertainty * self.calibration
        
        # Calculate z-score for desired confidence level
        z_score = stats.norm.ppf((1 + ci_level) / 2)
        
        # Calculate CI bounds
        lower = torch.clamp(mu - z_score * scaled_uncertainty, min=0.0)
        upper = torch.clamp(mu + z_score * scaled_uncertainty, max=1.0)
        
        # Combine into tensor
        ci = torch.cat([lower, upper], dim=1)
        
        return mu, ci, scaled_uncertainty
    
    def calibrate_model(model, val_loader, control_loader=None, device='cpu'):
        """
        Calibrate model confidence intervals and background level
        
        Args:
            model: The cancer detection model to calibrate
            val_loader: DataLoader for validation data
            control_loader: Optional DataLoader for control samples
            device: Device to run calibration on
            
        Returns:
            dict: Dictionary containing calibration parameters
        """
        from scipy import stats
        import numpy as np
        
        model.eval()
        
        # 1. Calibrate confidence intervals - use fewer test factors for speed
        best_factor = 1.0
        best_error = float('inf')
        
        with torch.no_grad():
            # Test different calibration factors - simplified options
            for factor in [0.5, 1.0, 2.0]:
                coverage_error = 0
                n_batches = 0
                
                for batch_data in val_loader:
                    # Handle both 3-element and 4-element returns
                    if len(batch_data) == 4:
                        marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask
                    else:
                        marker_values, coverage, y_true = batch_data
                    
                    marker_values = marker_values.to(device)
                    coverage = coverage.to(device)
                    y_true = y_true.to(device)
                    
                    mu, uncertainty, _, _ = model(marker_values, coverage)
                    
                    # Apply test calibration factor
                    uncertainty_calibrated = uncertainty * factor
                    
                    # Calculate CI
                    z_score = 1.96  # for 95% CI
                    lower = torch.clamp(mu - z_score * uncertainty_calibrated, min=0.0)
                    upper = torch.clamp(mu + z_score * uncertainty_calibrated, max=1.0)
                    
                    # Calculate CI coverage
                    in_ci = (y_true >= lower) & (y_true <= upper)
                    ci_coverage = in_ci.float().mean().item()
                    
                    # Error relative to target 95%
                    error = abs(ci_coverage - 0.95)
                    coverage_error += error
                    n_batches += 1
                
                avg_error = coverage_error / n_batches
                if avg_error < best_error:
                    best_error = avg_error
                    best_factor = factor
        
        # 2. Calibrate background level using controls - more conservative approach
        bg_params = {}
        if control_loader is not None:
            all_raw_preds = []
            
            with torch.no_grad():
                for batch_data in control_loader:
                    if len(batch_data) == 4:
                        marker_values, coverage, _, _ = batch_data
                    else:
                        marker_values, coverage, _ = batch_data
                    
                    marker_values = marker_values.to(device)
                    coverage = coverage.to(device)
                    
                    # Get features for raw predictions
                    value_features = model.value_embedding(marker_values.unsqueeze(-1))
                    log_coverage = torch.log1p(coverage).unsqueeze(-1)
                    coverage_features = model.coverage_embedding(log_coverage)
                    features = torch.cat([value_features, coverage_features], dim=-1)
                    features = model.feature_projection(features)
                    
                    # Apply transformer with masking
                    mask_missing = (coverage == 0)
                    mask_low_cov = (coverage < model.min_reliable_coverage)
                    mask = mask_missing | mask_low_cov
                    transformer_output = model.transformer_encoder(features, src_key_padding_mask=mask)
                    
                    # Apply attention
                    attention_scores = model.attention(transformer_output).squeeze(-1)
                    reliability_weight = model.reliability_weight(log_coverage)
                    coverage_reliability = torch.pow(
                        torch.clamp(coverage.unsqueeze(-1) / model.min_reliable_coverage, 0.0, 1.0), 2
                    )
                    reliability_weight = reliability_weight * coverage_reliability
                    attention_scores = attention_scores * reliability_weight.squeeze(-1)
                    attention_scores = attention_scores.masked_fill(mask, -1e9)
                    attention_weights = F.softmax(attention_scores, dim=1)
                    
                    # Get aggregated features
                    aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)
                    
                    # Get raw predictions before background correction
                    raw_mu = model.mu_head(aggregated)
                    all_raw_preds.append(raw_mu.cpu())
            
            # Use 90th percentile instead of 95th for more conservative background
            all_raw_preds = torch.cat(all_raw_preds)
            global_bg_level = float(torch.quantile(all_raw_preds, 0.9))
            
            # Ensure minimum is at least 0.003 and maximum is 0.04
            global_bg_level = max(0.003, min(global_bg_level, 0.04))
            
            # Update background correction module parameters
            with torch.no_grad():
                model.bg_correction.min_bg = global_bg_level * 0.75  # Set min to 75% of detected level
                model.bg_correction.max_bg = global_bg_level * 1.25  # Set max to 125% of detected level
                
            bg_params = {
                'global_bg_level': global_bg_level,
                'min_bg': model.bg_correction.min_bg,
                'max_bg': model.bg_correction.max_bg
            }
        
        # Apply calibration factors to model
        with torch.no_grad():
            model.calibration.copy_(torch.tensor([best_factor]))
        
        # Return calibration parameters
        calibration_results = {
            'calibration_factor': best_factor,
            'coverage_error': float(best_error),
            **bg_params
        }
    
        return calibration_results
    
    def get_background_levels(self, data_loader, device='cpu'):
        """
        Get dynamic background levels for all samples in a dataset.
        Useful for analyzing background distribution.
        
        Args:
            data_loader: DataLoader with samples
            device: Device to run inference on
            
        Returns:
            bg_levels: Numpy array of background levels for each sample
        """
        self.eval()
        all_bg_levels = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                # Handle different data formats
                if len(batch_data) == 4:
                    marker_values, coverage, _, _ = batch_data
                else:
                    marker_values, coverage, _ = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                
                # Get features for background prediction
                value_features = self.value_embedding(marker_values.unsqueeze(-1))
                log_coverage = torch.log1p(coverage).unsqueeze(-1)
                coverage_features = self.coverage_embedding(log_coverage)
                features = torch.cat([value_features, coverage_features], dim=-1)
                features = self.feature_projection(features)
                
                # Apply transformer with masking
                mask_missing = (coverage == 0)
                mask_low_cov = (coverage < self.min_reliable_coverage)
                mask = mask_missing | mask_low_cov
                transformer_output = self.transformer_encoder(features, src_key_padding_mask=mask)
                
                # Apply attention
                attention_scores = self.attention(transformer_output).squeeze(-1)
                reliability_weight = self.reliability_weight(log_coverage)
                coverage_reliability = torch.pow(
                    torch.clamp(coverage.unsqueeze(-1) / self.min_reliable_coverage, 0.0, 1.0), 2
                )
                reliability_weight = reliability_weight * coverage_reliability
                attention_scores = attention_scores * reliability_weight.squeeze(-1)
                attention_scores = attention_scores.masked_fill(mask, -1e9)
                attention_weights = F.softmax(attention_scores, dim=1)
                
                # Get aggregated features
                aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)
                
                # Get background level
                bg_levels = self.bg_correction(aggregated)
                all_bg_levels.append(bg_levels.cpu().numpy())
        
        return np.concatenate(all_bg_levels)
   

def compute_concentration_loss(mu, y_true):
    """
    Compute a concentration-aware loss function that emphasizes accurate estimation
    across all concentration ranges.
    
    Args:
        mu: Predicted concentration values [batch_size, 1]
        y_true: Ground truth concentration values [batch_size, 1]
        
    Returns:
        Loss value for optimization
    """
    # Basic MSE loss
    mse_loss = F.mse_loss(mu, y_true, reduction='none')
    
    # Calculate relative error for non-zero targets
    # This helps focus on percentage accuracy for higher concentrations
    epsilon = 1e-6
    non_zero_mask = (y_true > epsilon)
    
    # Initialize relative error tensor
    rel_error = torch.zeros_like(mse_loss)
    
    # Compute relative error only for non-zero targets
    if non_zero_mask.sum() > 0:
        rel_error[non_zero_mask] = torch.abs(mu[non_zero_mask] - y_true[non_zero_mask]) / (y_true[non_zero_mask] + epsilon)
    
    # Create concentration-based weights using log-scale approach
    # Higher weights for lower concentrations, gradually decreasing for higher concentrations
    # This addresses the natural imbalance in percentage error impact
    log_weights = 1.0 / torch.log10(y_true * 1000 + 10.0)  # +10 to avoid log(0)
    log_weights = torch.clamp(log_weights, 0.5, 2.0)  # Limit weight range
    
    # Zero-concentration specific penalty
    # Critical for avoiding false positives in control samples
    zero_mask = (y_true < epsilon)
    zero_penalty = 10.0 * mu[zero_mask].sum() if zero_mask.sum() > 0 else 0.0
    
    # Combine MSE and relative error components with log weighting
    weighted_mse = (mse_loss * log_weights).mean()
    weighted_rel = (rel_error * log_weights).mean() if non_zero_mask.sum() > 0 else 0.0
    
    # Combine components with appropriate scaling
    # MSE is good for overall accuracy, relative error helps with percentage accuracy
    combined_loss = weighted_mse + 0.5 * weighted_rel + zero_penalty
    
    return combined_loss


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