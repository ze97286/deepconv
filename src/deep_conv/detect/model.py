import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.stats as stats
import numpy as np

class DynamicBackgroundCorrection(nn.Module):
    """
    Dynamic background correction module with customisable settings
    """
    def __init__(self, feature_dim, min_bg=0.001, max_bg=0.04, marker_specific=False, num_markers=None):
        super().__init__()
        self.min_bg = min_bg
        self.max_bg = max_bg
        self.marker_specific = marker_specific
        
        # Network to predict sample-specific background level
        self.bg_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, 1 if not marker_specific else num_markers),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
        # Initialise to predict conservative background level
        with torch.no_grad():
            self.bg_network[-2].bias.data.fill_(-3.0)
    
    def forward(self, features, marker_features=None):
        """
        Predict background level for each sample based on its features
        
        Args:
            features: Sample features [batch_size, feature_dim]
            marker_features: Optional marker features for marker-specific correction
                Shape: [batch_size, num_markers, feature_dim]
                
        Returns:
            bg_level: Predicted background level [batch_size, 1] or [batch_size, num_markers]
        """
        if self.marker_specific and marker_features is not None:
            # For marker-specific background correction
            # Process each marker separately
            B, M, D = marker_features.shape
            
            # Reshape to process all markers at once
            flat_marker_features = marker_features.reshape(-1, D)  # [B*M, D]
            flat_bg_scale = self.bg_network(flat_marker_features)  # [B*M, 1]
            
            # Reshape back to [B, M, 1]
            bg_scale = flat_bg_scale.reshape(B, M, 1)
            
            # Scale sigmoid output to desired background range
            bg_level = self.min_bg + bg_scale * (self.max_bg - self.min_bg)
        else:
            # Sample-level background
            bg_scale = self.bg_network(features)
            bg_level = self.min_bg + bg_scale * (self.max_bg - self.min_bg)
        
        return bg_level
      
class AdaptiveDetectionThresholds(nn.Module):
    """
    Adaptive detection thresholds module that adjusts based on SNR profile
    """
    def __init__(self, feature_dim, base_thresholds):
        super().__init__()
        self.base_thresholds = base_thresholds
        self.num_thresholds = len(base_thresholds)
        
        # Network to adjust thresholds based on signal-to-noise characteristics
        self.threshold_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, self.num_thresholds),
            nn.Sigmoid()
        )
        
        # Initialise to produce base thresholds
        with torch.no_grad():
            # Set bias to produce approximately base thresholds from sigmoid
            for i, threshold in enumerate(base_thresholds):
                # Inverse sigmoid: log(p/(1-p))
                bias = -math.log(1.0/threshold - 1.0)
                self.threshold_network[-2].bias.data[i] = bias
    
    def forward(self, features):
        """
        Compute adaptive thresholds for each sample
        
        Args:
            features: Sample features [batch_size, feature_dim]
            
        Returns:
            thresholds: Adjusted thresholds [batch_size, num_thresholds]
        """
        # Get adjustment factors (0.5 to 1.5 range)
        threshold_factors = 0.5 + self.threshold_network(features)
        
        # Apply to base thresholds
        base = torch.tensor(self.base_thresholds, device=features.device).unsqueeze(0)
        
        # Limit adjustment based on SNR
        adjusted_thresholds = base * threshold_factors
        
        return adjusted_thresholds
        
class ConcentrationFocusedLoss(nn.Module):
    """
    Enhanced concentration-focused loss function with range-specific weighting
    """
    def __init__(self, critical_ranges=None, zero_penalty=10.0, control_penalty=20.0):
        super().__init__()
        self.critical_ranges = critical_ranges or [(0.0001, 0.001, 1.8), (0.001, 0.01, 1.5), (0.01, 0.05, 1.2)]
        self.zero_penalty = zero_penalty
        self.control_penalty = control_penalty
    
    def forward(self, mu, uncertainty, y_true, control_mask=None):
        """
        Compute concentration-focused loss with enhanced range-specific weighting
        
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
        
        # Initialise relative error tensor
        rel_error = torch.zeros_like(mse_loss)
        
        # Compute relative error only for non-zero targets
        if non_zero_mask.sum() > 0:
            rel_error[non_zero_mask] = torch.abs(mu[non_zero_mask] - y_true[non_zero_mask]) / (y_true[non_zero_mask] + epsilon)
        
        # Apply log-scale weighting - but adjust it for critical ranges
        log_weights = 1.0 / torch.log10(y_true * 1000 + 10.0)
        log_weights = torch.clamp(log_weights, 0.5, 2.0)
        
        # Apply additional weights for critical ranges
        for low, high, weight in self.critical_ranges:
            range_mask = (y_true >= low) & (y_true < high)
            log_weights = torch.where(range_mask, log_weights * weight, log_weights)
        
        # Zero-concentration specific penalty - stronger for false positives
        zero_mask = (y_true < epsilon)
        zero_penalty = self.zero_penalty * mu[zero_mask].mean() if zero_mask.sum() > 0 else 0.0
        
        # Control sample penalty - even stronger for known negatives
        control_loss = 0.0
        if control_mask is not None and control_mask.sum() > 0:
            control_loss = self.control_penalty * mu[control_mask].mean()
        
        # Apply weights to MSE and relative error
        weighted_mse = (mse_loss * log_weights).mean()
        weighted_rel = (rel_error * log_weights).mean() if non_zero_mask.sum() > 0 else 0.0
        
        # Add calibration component for uncertainty estimates
        calibration_loss = 0.0
        if uncertainty is not None:
            z_scores = torch.abs(mu - y_true) / (uncertainty + 1e-6)
            calibration_loss = F.smooth_l1_loss(z_scores, torch.ones_like(z_scores) * 1.96)
        
        # Combine all components with adjusted weights
        total_loss = weighted_mse + 0.7 * weighted_rel + zero_penalty + control_loss + 0.2 * calibration_loss
        
        return total_loss
    
class EnhancedCancerDetectionModel(nn.Module):
    """
    Enhanced deep learning model for cancer detection from cfDNA methylation markers.
    Enhancements focused on lower concentration ranges and adaptive behavior.
    """
    def __init__(self, num_markers, feature_dim=128, num_heads=8, num_layers=3, 
                 dropout_rate=0.2, detection_thresholds=(0.0005, 0.001, 0.005, 0.01, 0.05),
                 min_reliable_coverage=5.0, marker_specific_bg=False, 
                 critical_ranges=None, enable_adaptive_thresholds=True,
                 snr_profile="high"):
        """
        Initialise the ImprovedCancerDetectionModel.
        
        Args:
            num_markers: Number of methylation markers in input
            feature_dim: Dimension of feature representations
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer encoder layers
            dropout_rate: Dropout probability for regularization
            detection_thresholds: Concentration thresholds for binary detection
            min_reliable_coverage: Minimum coverage to consider a marker reliable
            marker_specific_bg: Whether to use marker-specific background correction
            critical_ranges: Ranges to emphasize in loss function
            enable_adaptive_thresholds: Whether to use adaptive detection thresholds
            snr_profile: Profile indicating signal-to-noise characteristics
        """
        super().__init__()
        
        # Store configuration parameters
        self.detection_thresholds = detection_thresholds
        self.num_markers = num_markers
        self.min_reliable_coverage = min_reliable_coverage
        self.marker_specific_bg = marker_specific_bg
        self.snr_profile = snr_profile
        
        # Adjust settings based on SNR profile
        if snr_profile == "high":
            # For OAC and other high SNR cases
            min_bg, max_bg = 0.001, 0.02
            init_mu_bias = -2.5
            critical_ranges = critical_ranges or [(0.0005, 0.001, 2.0), (0.001, 0.01, 1.5), (0.01, 0.05, 1.2)]
        elif snr_profile == "medium":
            # Default for most cell types
            min_bg, max_bg = 0.002, 0.03
            init_mu_bias = -2.0
            critical_ranges = critical_ranges or [(0.001, 0.005, 2.0), (0.005, 0.02, 1.5), (0.02, 0.1, 1.2)]
        else:  # "low"
            # For T-cells and other challenging cell types
            min_bg, max_bg = 0.003, 0.05
            init_mu_bias = -1.5
            critical_ranges = critical_ranges or [(0.001, 0.01, 2.0), (0.01, 0.05, 1.8), (0.05, 0.2, 1.5)]
            
        # Separate embeddings for marker values and coverage
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
        
        # Add positional embeddings for markers to help model understand marker positions
        self.marker_pos_embedding = nn.Parameter(torch.randn(1, num_markers, feature_dim) * 0.02)
        
        # Transformer encoder for learning marker interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 3,
            dropout=dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Two-level attention mechanism for marker importance weighting
        # First level - base importance
        self.attention_base = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Second level - concentration-dependent importance
        self.attention_conc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Coverage reliability weighting mechanism - enhanced version
        self.reliability_weight = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, feature_dim // 8),
            nn.GELU(),
            nn.Linear(feature_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # Concentration estimation head (mu) with residual connection for fine tuning
        self.mu_pre = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU()
        )
        
        self.mu_head = nn.Sequential(
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Fine-tuning residual connection for mu
        self.mu_residual = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1),
            nn.Tanh()  # Outputs small adjustments centered around 0
        )
        
        # Dynamic background correction module
        self.bg_correction = DynamicBackgroundCorrection(
            feature_dim, 
            min_bg=min_bg, 
            max_bg=max_bg,
            marker_specific=marker_specific_bg,
            num_markers=num_markers if marker_specific_bg else None
        )
        
        # Uncertainty estimation head with enhanced architecture
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Softplus()
        )
        
        # Adaptive detection thresholds (optional)
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        if enable_adaptive_thresholds:
            self.adaptive_thresholds = AdaptiveDetectionThresholds(feature_dim, detection_thresholds)
        
        # Binary detection heads for different concentration thresholds
        self.detection_heads = nn.ModuleList()
        for _ in detection_thresholds:
            head = nn.Sequential(
                nn.Linear(feature_dim + 1, feature_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim // 2, feature_dim // 4),
                nn.GELU(),
                nn.Linear(feature_dim // 4, 1),
                nn.Sigmoid()
            )
            self.detection_heads.append(head)
        
        # Concentration-focused loss module
        self.concentration_loss = ConcentrationFocusedLoss(critical_ranges=critical_ranges)
        
        # Confidence interval calibration parameter
        self.register_buffer('calibration', torch.ones(1))
        
        # Initialise mu_head bias to predict low values initially
        with torch.no_grad():
            self.mu_head[0].bias.data.fill_(init_mu_bias)

    def forward(self, marker_values, coverage):
        """
        Forward pass through the model with enhanced attention and background correction
        
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
        
        # Calculate reliability weights with smoother cutoff function
        log_coverage = torch.log1p(coverage).unsqueeze(-1)
        reliability_weight = self.reliability_weight(log_coverage)
        
        # Apply enhanced coverage reliability using sigmoid function
        # This creates a smoother transition between unreliable and reliable
        coverage_reliability = torch.sigmoid((coverage.unsqueeze(-1) - self.min_reliable_coverage) / 2)
        reliability_weight = reliability_weight * coverage_reliability
        
        # Handle NaN values
        marker_values = torch.nan_to_num(marker_values, nan=0.0)
        
        # Embed marker values and coverage
        value_features = self.value_embedding(marker_values.unsqueeze(-1))
        coverage_features = self.coverage_embedding(log_coverage)
        
        # Combine features and add positional embeddings
        features = torch.cat([value_features, coverage_features], dim=-1)
        features = self.feature_projection(features)
        features = features + self.marker_pos_embedding
        
        # Apply transformer with masking
        transformer_output = self.transformer_encoder(
            features, 
            src_key_padding_mask=mask
        )
        
        # Apply two-level attention with reliability weighting
        attention_base = self.attention_base(transformer_output).squeeze(-1)
        attention_conc = self.attention_conc(transformer_output).squeeze(-1)
        
        # Combine attentions and apply reliability weighting
        attention_scores = attention_base + attention_conc
        attention_scores = attention_scores * reliability_weight.squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute weighted sum of features
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)
        
        # Predict raw concentration with residual fine-tuning
        mu_features = self.mu_pre(aggregated)
        mu_base = self.mu_head(mu_features)
        mu_adj = 0.05 * self.mu_residual(aggregated)  # Small residual adjustment
        mu = torch.clamp(mu_base + mu_adj, 0.0, 1.0)
        
        # Apply dynamic background correction
        if self.marker_specific_bg:
            # For marker-specific background (more complex)
            marker_bg = self.bg_correction(aggregated, marker_features=transformer_output)
            # Apply attention-weighted background correction
            bg_level = torch.sum(attention_weights.unsqueeze(-1) * marker_bg, dim=1)
        else:
            # Sample-level background
            bg_level = self.bg_correction(aggregated)
        
        mu_corrected = torch.clamp(mu - bg_level, min=0.0)
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(aggregated)
        
        # Get adaptive detection thresholds if enabled
        if self.enable_adaptive_thresholds:
            thresholds = self.adaptive_thresholds(aggregated)
        
        # Get detection probabilities
        detection_features = torch.cat([aggregated, uncertainty], dim=1)
        detection_probs = []
        
        for i, head in enumerate(self.detection_heads):
            prob = head(detection_features)
            # If using adaptive thresholds, adjust probability
            if self.enable_adaptive_thresholds:
                # Simple interpolation between base probabilities
                base_prob = prob
                threshold_ratio = thresholds[:, i:i+1] / self.detection_thresholds[i]
                # Adjust probability based on threshold changes
                adj_factor = torch.pow(threshold_ratio, 0.5)  # Square root to dampen effect
                prob = torch.clamp(base_prob * adj_factor, 0.0, 1.0)
            
            detection_probs.append(prob)
        
        return mu_corrected, uncertainty, detection_probs, attention_weights
    
    def compute_loss(self, mu, uncertainty, y_true, control_mask=None):
        """
        Compute concentration-focused loss using the dedicated loss module
        
        Args:
            mu: Predicted concentration values [batch_size, 1]
            uncertainty: Predicted uncertainty values [batch_size, 1]
            y_true: Ground truth concentration values [batch_size, 1]
            control_mask: Optional boolean mask identifying control samples
            
        Returns:
            total_loss: Combined loss value for optimization
        """
        return self.concentration_loss(mu, uncertainty, y_true, control_mask)
    
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
        Calibrate model confidence intervals and background level
        
        Args:
            val_loader: DataLoader for validation data
            control_loader: Optional DataLoader for control samples
            device: Device to run calibration on
            
        Returns:
            dict: Dictionary containing calibration parameters
        """
        from scipy import stats
        import numpy as np
        
        self.eval()
        
        # 1. Calibrate confidence intervals with more options
        best_factor = 1.0
        best_error = float('inf')
        
        with torch.no_grad():
            # Test different calibration factors with finer granularity
            for factor in [0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 2.5]:
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
                    
                    mu, uncertainty, _, _ = self(marker_values, coverage)
                    
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
        
        # 2. Calibrate dynamic background correction using controls
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
                    
                    # Extract features for background estimation
                    mask_missing = (coverage == 0)
                    mask_low_cov = (coverage < self.min_reliable_coverage)
                    mask = mask_missing | mask_low_cov
                    
                    # Embed marker values and coverage
                    value_features = self.value_embedding(marker_values.unsqueeze(-1))
                    log_coverage = torch.log1p(coverage).unsqueeze(-1)
                    coverage_features = self.coverage_embedding(log_coverage)
                    
                    # Combine features
                    features = torch.cat([value_features, coverage_features], dim=-1)
                    features = self.feature_projection(features)
                    features = features + self.marker_pos_embedding
                    
                    # Process through transformer
                    transformer_output = self.transformer_encoder(
                        features, 
                        src_key_padding_mask=mask
                    )
                    
                    # Calculate attention weights
                    attention_base = self.attention_base(transformer_output).squeeze(-1)
                    attention_conc = self.attention_conc(transformer_output).squeeze(-1)
                    
                    # Calculate reliability
                    reliability_weight = self.reliability_weight(log_coverage)
                    coverage_reliability = torch.sigmoid((coverage.unsqueeze(-1) - self.min_reliable_coverage) / 2)
                    reliability_weight = reliability_weight * coverage_reliability
                    
                    # Combine attentions and apply reliability
                    attention_scores = attention_base + attention_conc
                    attention_scores = attention_scores * reliability_weight.squeeze(-1)
                    attention_scores = attention_scores.masked_fill(mask, -1e9)
                    attention_weights = F.softmax(attention_scores, dim=1)
                    
                    # Get aggregated features
                    aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)
                    
                    # Get raw predictions before background correction
                    mu_features = self.mu_pre(aggregated)
                    mu_base = self.mu_head(mu_features)
                    mu_adj = 0.05 * self.mu_residual(aggregated)
                    mu = torch.clamp(mu_base + mu_adj, 0.0, 1.0)
                    
                    all_raw_preds.append(mu.cpu())
            
            # Calculate background level from controls
            all_raw_preds = torch.cat(all_raw_preds)
            
            # Use 90th percentile for more conservative background
            global_bg_level = float(torch.quantile(all_raw_preds, 0.9))
            
            # Ensure minimum and maximum are reasonable for the SNR profile
            if self.snr_profile == "high":
                global_bg_level = max(0.001, min(global_bg_level, 0.02))
            elif self.snr_profile == "medium":
                global_bg_level = max(0.002, min(global_bg_level, 0.03))
            else:  # "low"
                global_bg_level = max(0.003, min(global_bg_level, 0.05))
            
            # Update background correction module parameters
            with torch.no_grad():
                self.bg_correction.min_bg = global_bg_level * 0.75
                self.bg_correction.max_bg = global_bg_level * 1.25
                
            bg_params = {
                'global_bg_level': global_bg_level,
                'min_bg': self.bg_correction.min_bg,
                'max_bg': self.bg_correction.max_bg
            }
        
        # Apply calibration factors to model
        with torch.no_grad():
            self.calibration.copy_(torch.tensor([best_factor]))
        
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
                
                # Process through first part of model to get aggregated features
                mask_missing = (coverage == 0)
                mask_low_cov = (coverage < self.min_reliable_coverage)
                mask = mask_missing | mask_low_cov
                
                # Embed marker values and coverage
                value_features = self.value_embedding(marker_values.unsqueeze(-1))
                log_coverage = torch.log1p(coverage).unsqueeze(-1)
                coverage_features = self.coverage_embedding(log_coverage)
                
                # Combine features
                features = torch.cat([value_features, coverage_features], dim=-1)
                features = self.feature_projection(features)
                features = features + self.marker_pos_embedding
                
                # Process through transformer
                transformer_output = self.transformer_encoder(
                    features, 
                    src_key_padding_mask=mask
                )
                
                # Calculate attention weights
                attention_base = self.attention_base(transformer_output).squeeze(-1)
                attention_conc = self.attention_conc(transformer_output).squeeze(-1)
                
                # Combine attentions and apply reliability
                reliability_weight = self.reliability_weight(log_coverage)
                coverage_reliability = torch.sigmoid((coverage.unsqueeze(-1) - self.min_reliable_coverage) / 2)
                reliability_weight = reliability_weight * coverage_reliability
                
                attention_scores = attention_base + attention_conc
                attention_scores = attention_scores * reliability_weight.squeeze(-1)
                attention_scores = attention_scores.masked_fill(mask, -1e9)
                attention_weights = F.softmax(attention_scores, dim=1)
                
                # Get aggregated features
                aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)
                
                # Get background level
                bg_levels = self.bg_correction(aggregated)
                all_bg_levels.append(bg_levels.cpu().numpy())
        
        return np.concatenate(all_bg_levels)


class MarkerImportanceAnalyser:
    """
    Enhanced utility class to analyse marker importance with SNR considerations
    """
    def __init__(self, model):
        self.model = model
    
    def get_marker_importance(self, dataloader, top_k=20, stratify_by_concentration=True):
        """
        Analyse marker importance across the dataset with optional stratification
        
        Args:
            dataloader: DataLoader with samples to analyse
            top_k: Number of top markers to return
            stratify_by_concentration: Whether to stratify importance by concentration
            
        Returns:
            top_indices: Indices of top markers
            top_weights: Weights of top markers
            conc_stratified: Optional dictionary of stratified results
        """
        self.model.eval()
        all_attentions = []
        all_concentrations = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Handle both 3-element and 4-element returns
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data
                else:
                    marker_values, coverage, y_true = batch_data
                
                # Get predictions and attention weights
                mu, _, _, attention_weights = self.model(marker_values, coverage)
                
                # Store results
                all_attentions.append(attention_weights)
                all_concentrations.append(y_true)
        
        # Concatenate results
        attention_weights = torch.cat(all_attentions, dim=0)  # [N, M]
        concentrations = torch.cat(all_concentrations, dim=0)  # [N, 1]
        
        # Average attention weights across all samples
        avg_attention = attention_weights.mean(dim=0)  # [M]
        
        # Get top-k markers by attention weight
        top_k_indices = torch.topk(avg_attention, k=min(top_k, len(avg_attention))).indices
        top_k_weights = avg_attention[top_k_indices]
        
        # Optionally stratify by concentration
        conc_stratified = None
        if stratify_by_concentration:
            # Define concentration ranges
            ranges = [
                ("very_low", 0.0, 0.001),
                ("low", 0.001, 0.01),
                ("medium", 0.01, 0.05),
                ("high", 0.05, 1.0)
            ]
            
            conc_stratified = {}
            for name, low, high in ranges:
                # Create mask for this range
                mask = (concentrations >= low) & (concentrations < high)
                
                # Skip if no samples in this range
                if not mask.any():
                    continue
                    
                # Calculate stratified attention
                range_attention = attention_weights[mask.squeeze()].mean(dim=0)
                
                # Get top-k markers for this range
                range_top_indices = torch.topk(range_attention, k=min(top_k, len(range_attention))).indices
                range_top_weights = range_attention[range_top_indices]
                
                # Store results
                conc_stratified[name] = {
                    "count": int(mask.sum().item()),
                    "indices": range_top_indices.cpu().numpy(),
                    "weights": range_top_weights.cpu().numpy()
                }
        
        return top_k_indices.cpu().numpy(), top_k_weights.cpu().numpy(), conc_stratified
    
    def analyse_detection_performance(self, dataloader, thresholds=None, min_samples_per_bin=20):
        """
        Analyse detection performance with enhanced concentration-specific metrics
        
        Args:
            dataloader: DataLoader with samples to analyse
            thresholds: Optional concentration thresholds to analyse
            min_samples_per_bin: Minimum samples required for concentration bin analysis
            
        Returns:
            Dictionary of metrics for each threshold
        """
        model_thresholds = self.model.detection_thresholds
                
        if thresholds is None:
            thresholds = model_thresholds
        
        self.model.eval()
        predictions = []
        ground_truth = []
        detection_probs = []
        uncertainties = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Handle both 3-element and 4-element returns
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data
                else:
                    marker_values, coverage, y_true = batch_data
                
                # Get model outputs
                mu, uncertainty, det_probs, _ = self.model(marker_values, coverage)
                
                # Store results
                predictions.append(mu.cpu().numpy())
                ground_truth.append(y_true.cpu().numpy())
                detection_probs.append([dp.cpu().numpy() for dp in det_probs])
                uncertainties.append(uncertainty.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth).flatten()
        detection_probs = [np.concatenate([dp[i] for dp in detection_probs]) for i in range(len(model_thresholds))]
        uncertainties = np.concatenate(uncertainties).flatten()
        
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
            
            # Find sensitivity at various specificity levels
            spec_levels = [0.95, 0.98, 0.99]
            sens_at_spec = {}
            
            for spec_level in spec_levels:
                fpr_target = 1.0 - spec_level
                idx = np.argmin(np.abs(fpr - fpr_target))
                sens_at_spec[f"{spec_level:.2f}"] = {
                    "sensitivity": float(tpr[idx]),
                    "threshold": float(roc_thresholds[idx]) if idx < len(roc_thresholds) else None
                }
            
            # Precision-recall curve and average precision
            precision, recall, pr_thresholds = precision_recall_curve(y_binary, detection_probs[i])
            ap = average_precision_score(y_binary, detection_probs[i])
            
            # Add concentration-specific analysis
            conc_bins = [
                (0.0, 0.0005, "0-0.05%"),
                (0.0005, 0.001, "0.05-0.1%"),
                (0.001, 0.005, "0.1-0.5%"),
                (0.005, 0.01, "0.5-1%"),
                (0.01, 0.05, "1-5%"),
                (0.05, 0.1, "5-10%"),
                (0.1, 1.0, ">10%")
            ]
            
            bin_performance = {}
            for low, high, name in conc_bins:
                # Create mask for this bin
                bin_mask = (ground_truth >= low) & (ground_truth < high)
                bin_count = np.sum(bin_mask)
                
                # Skip if not enough samples
                if bin_count < min_samples_per_bin:
                    continue
                
                # Calculate bin-specific metrics
                bin_pred = predictions[bin_mask]
                bin_gt = ground_truth[bin_mask]
                bin_uncertainty = uncertainties[bin_mask]
                
                # Basic metrics
                bin_mae = np.mean(np.abs(bin_pred - bin_gt))
                bin_rmse = np.sqrt(np.mean((bin_pred - bin_gt)**2))
                
                # Relative error (for non-zero targets)
                bin_rel_errors = []
                non_zero_mask = bin_gt > 0
                if np.sum(non_zero_mask) > 0:
                    bin_rel_errors = np.abs(bin_pred[non_zero_mask] - bin_gt[non_zero_mask]) / bin_gt[non_zero_mask]
                    bin_mape = np.mean(bin_rel_errors) * 100
                    bin_within_25pct = np.mean(bin_rel_errors <= 0.25) * 100
                else:
                    bin_mape = None
                    bin_within_25pct = None
                
                # Detection accuracy if threshold within this bin
                if low <= threshold < high:
                    bin_detection_probs = detection_probs[i][bin_mask]
                    bin_y_binary = (bin_gt >= threshold).astype(int)
                    
                    # Only calculate if we have both positive and negative examples
                    if np.sum(bin_y_binary) > 0 and np.sum(bin_y_binary) < len(bin_y_binary):
                        try:
                            bin_fpr, bin_tpr, _ = roc_curve(bin_y_binary, bin_detection_probs)
                            bin_auc = auc(bin_fpr, bin_tpr)
                        except:
                            bin_auc = None
                    else:
                        bin_auc = None
                else:
                    bin_auc = None
                
                # Store bin results
                bin_performance[name] = {
                    "count": int(bin_count),
                    "mae": float(bin_mae),
                    "rmse": float(bin_rmse),
                    "mean_uncertainty": float(np.mean(bin_uncertainty)),
                    "mape": float(bin_mape) if bin_mape is not None else None,
                    "within_25pct": float(bin_within_25pct) if bin_within_25pct is not None else None,
                    "auc": float(bin_auc) if bin_auc is not None else None
                }
            
            # Store all results for this threshold
            results[threshold] = {
                'auc': float(roc_auc),
                'sensitivity_at_specificity': sens_at_spec,
                'average_precision': float(ap),
                'precision_recall_data': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
                },
                'concentration_bins': bin_performance
            }
        
        return results