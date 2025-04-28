import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.stats as stats
import numpy as np

class DynamicBackgroundCorrection(nn.Module):
    """
    Dynamic background correction module with conservative settings for low concentrations
    """
    def __init__(self, feature_dim, min_bg=0.0005, max_bg=0.03):
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
        
        # Initialize to predict lower background initially
        with torch.no_grad():
            self.bg_network[-2].bias.data.fill_(-3.5)  # Start with more conservative bg
    
    def forward(self, features, concentration_hint=None):
        """
        Predict background level for each sample based on its features
        
        Args:
            features: Sample features [batch_size, feature_dim]
            concentration_hint: Optional hint about concentration range [batch_size, 1]
            
        Returns:
            bg_level: Predicted background level [batch_size, 1]
        """
        # Scale sigmoid output to desired background range
        bg_scale = self.bg_network(features)
        
        # Apply concentration-dependent scaling if hint is provided
        if concentration_hint is not None:
            # Reduce background for samples predicted to be in 0.1-1% range
            low_conc_mask = (concentration_hint >= 0.001) & (concentration_hint < 0.01)
            bg_scale = torch.where(low_conc_mask, bg_scale * 0.7, bg_scale)
        
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
        Forward pass with improved background correction
        
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
        
        # Get initial concentration estimate to guide background correction
        initial_mu = mu.detach()
        
        # Apply dynamic background correction with concentration hint
        bg_level = self.bg_correction(aggregated, initial_mu)
        
        # Apply conservative adjustment for 0.1-1% range
        critical_range_mask = (initial_mu >= 0.001) & (initial_mu < 0.01)
        bg_adjusted = torch.where(critical_range_mask, bg_level * 0.8, bg_level)
        
        # Apply background correction with the adjusted level
        mu_corrected = torch.clamp(mu - bg_adjusted, min=0.0)
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(aggregated)
        
        # Get detection probabilities
        detection_features = torch.cat([aggregated, uncertainty], dim=1)
        detection_probs = [head(detection_features) for head in self.detection_heads]
        
        return mu_corrected, uncertainty, detection_probs, attention_weights

    def compute_loss(self, mu, uncertainty, y_true, control_mask=None):
        """
        Compute concentration-focused loss function with improved weighting for critical ranges
        
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
        
        # Create concentration-specific weights
        # 1. Higher weights for 0.1-1% range (critical range)
        # 2. Moderate weights for 1-5% range
        # 3. Lower weights for other ranges
        range_weights = torch.ones_like(y_true)
        
        # 0.1-1% range - critical range with higher weight
        critical_range_mask = (y_true >= 0.001) & (y_true < 0.01)
        range_weights[critical_range_mask] = 3.0  # Increased from original
        
        # 1-5% range
        mid_range_mask = (y_true >= 0.01) & (y_true < 0.05)
        range_weights[mid_range_mask] = 2.0
        
        # Apply log-scale weighting on top of range weights
        log_weights = 1.0 / torch.log10(y_true * 1000 + 10.0)
        log_weights = torch.clamp(log_weights, 0.5, 2.0)
        
        # Combined weights
        combined_weights = range_weights * log_weights
        
        # Zero-concentration specific penalty
        zero_mask = (y_true < epsilon)
        zero_penalty = 10.0 * mu[zero_mask].mean() if zero_mask.sum() > 0 else 0.0
        
        # Control sample penalty
        control_loss = 0.0
        if control_mask is not None and control_mask.sum() > 0:
            control_loss = 15.0 * mu[control_mask].mean()
        
        # Combine MSE and relative error with weights
        weighted_mse = (mse_loss * combined_weights).mean()
        
        # Increase weight on relative error for accuracy at low concentrations
        weighted_rel = (rel_error * combined_weights).mean() if non_zero_mask.sum() > 0 else 0.0
        
        # Add calibration component
        calibration_loss = 0.0
        if uncertainty is not None:
            z_scores = torch.abs(mu - y_true) / (uncertainty + 1e-6)
            calibration_loss = F.smooth_l1_loss(z_scores, torch.ones_like(z_scores) * 1.96)
        
        # Combine all loss components with adjusted weights
        # Increased weight on relative error component from 0.5 to 0.8
        total_loss = weighted_mse + 0.8 * weighted_rel + zero_penalty + control_loss + 0.1 * calibration_loss
        
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
    
    def calibrate(self, val_loader, control_loader=None, device='cpu'):
        """
        Calibrate model with concentration-specific settings
        
        Args:
            val_loader: DataLoader with validation data
            control_loader: Optional DataLoader with control samples
            device: Device to run calibration on
            
        Returns:
            dict: Calibration results with calibration_factor and background parameters
        """
        self.eval()
        
        # 1. Calibrate confidence intervals
        all_errors = []
        all_uncertainties = []
        all_concentrations = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data
                else:
                    marker_values, coverage, y_true = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                mu, uncertainty, _, _ = self(marker_values, coverage)
                
                errors = torch.abs(mu - y_true)
                all_errors.append(errors.cpu())
                all_uncertainties.append(uncertainty.cpu())
                all_concentrations.append(y_true.cpu())
        
        # Calculate calibration factor
        all_errors = torch.cat(all_errors)
        all_uncertainties = torch.cat(all_uncertainties)
        all_concentrations = torch.cat(all_concentrations)
        
        # Stratify errors and calculate calibration factors by concentration range
        critical_range_mask = (all_concentrations >= 0.001) & (all_concentrations < 0.01)
        
        # For the critical range (0.1-1%), use a higher percentile for better coverage
        if critical_range_mask.sum() > 0:
            critical_errors = all_errors[critical_range_mask]
            critical_uncertainties = all_uncertainties[critical_range_mask]
            critical_error_97_percentile = torch.quantile(critical_errors, 0.97)  # Higher percentile
            critical_avg_uncertainty = critical_uncertainties.mean()
            critical_calibration = critical_error_97_percentile / (1.96 * critical_avg_uncertainty)
            
            # Ensure calibration factor is at least 1.5 for critical range
            critical_calibration = max(float(critical_calibration), 1.5)
        else:
            critical_calibration = 2.0  # Default if no samples in range
        
        # For other ranges, use the standard 95% percentile
        error_95_percentile = torch.quantile(all_errors, 0.95)
        avg_uncertainty = all_uncertainties.mean()
        standard_calibration = error_95_percentile / (1.96 * avg_uncertainty)
        
        # Use the larger of the two calibration factors to ensure good coverage
        calibration_factor = max(float(standard_calibration), float(critical_calibration))
        
        # Update calibration parameter
        with torch.no_grad():
            self.calibration.copy_(torch.tensor([calibration_factor]))
        
        # 2. Calibrate background correction parameters using control samples
        bg_params = {}
        if control_loader is not None:
            # Extract raw predictions from control samples
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
                    
                    # Get raw predictions before background correction
                    raw_mu = self.mu_head(aggregated)
                    all_raw_preds.append(raw_mu.cpu())
            
            # Calculate optimal background parameters
            all_raw_preds = torch.cat(all_raw_preds)
            
            # Use more conservative settings for background
            # 40th percentile instead of median (50th) for min_bg
            # 90th percentile instead of 95th for max_bg
            min_bg = float(torch.quantile(all_raw_preds, 0.4))
            max_bg = float(torch.quantile(all_raw_preds, 0.9))
            
            # Ensure min_bg is at most 0.0008 and max_bg is at most 0.03
            min_bg = min(0.0008, max(0.0003, min_bg))
            max_bg = min(0.03, max(min_bg + 0.005, max_bg))
            
            # Update background correction module parameters
            with torch.no_grad():
                self.bg_correction.min_bg = min_bg
                self.bg_correction.max_bg = max_bg
            
            bg_params = {
                'min_bg': min_bg,
                'max_bg': max_bg
            }
        
        # Return all calibration results
        return {
            'calibration_factor': float(calibration_factor),
            'critical_calibration': float(critical_calibration),
            'standard_calibration': float(standard_calibration),
            **bg_params
        }
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