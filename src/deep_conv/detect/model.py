import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.stats as stats
import numpy as np

class ZeroAnchoringLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.zero_detector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(feature_dim // 2),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.BatchNorm1d(feature_dim // 4),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        self.sharpness = nn.Parameter(torch.tensor(12.0))
        
    def forward(self, features, concentration):
        # Detect if sample should be zero
        features_flat = features.reshape(-1, features.size(-1))
        zero_prob_flat = self.zero_detector(features_flat)
        zero_prob = zero_prob_flat.reshape(concentration.shape)
        
        # Apply exponential dampening based on zero probability
        # When zero_prob is high, output approaches zero
        zero_factor = torch.exp(-self.sharpness * zero_prob)
        
        # Apply dampening (multiplication reduces the value toward zero)
        anchored_concentration = concentration * zero_factor
        
        return anchored_concentration, zero_prob
    
class ConcentrationFocusedLoss(nn.Module):
    def __init__(self, log_space_weight=6.0, critical_ranges=None, zero_penalty=10.0, control_penalty=1000.0):
        super().__init__()
        self.log_space_weight = log_space_weight
        self.critical_ranges = critical_ranges or [(0.001, 0.005, 2.0), (0.005, 0.01, 1.5), (0.01, 0.05, 1.2)]
        self.zero_penalty = zero_penalty
        self.control_penalty = control_penalty
    
    def forward(self, mu, uncertainty, y_true, control_mask=None, zero_prob=None):
        # Basic MSE loss
        mse_loss = F.mse_loss(mu, y_true, reduction='none')
        
        # Calculate relative error for non-zero targets
        epsilon = 1e-6
        non_zero_mask = (y_true > epsilon)
        
        # Initialize relative error tensor
        rel_error = torch.zeros_like(mse_loss, device=mse_loss.device)
        
        # Compute relative error only for non-zero targets
        if non_zero_mask.sum() > 0:
            rel_error = torch.where(
                non_zero_mask,
                torch.abs(mu - y_true) / (y_true + epsilon),
                rel_error
            )
        
        # Apply log-scale weighting
        log_weights = 1.0 / torch.log10(y_true * 1000 + 10.0)
        log_weights = torch.clamp(log_weights, 0.5, 3.0)
        
        # Apply range-specific weights
        range_weights = torch.ones_like(log_weights, device=log_weights.device)
        for low, high, weight in self.critical_ranges:
            range_mask = (y_true >= low) & (y_true < high)
            range_weights = torch.where(range_mask, torch.ones_like(range_weights, device=range_weights.device) * weight, range_weights)
        
        # Special focus on 0.1-0.5% range
        ultra_focused_range = (y_true >= 0.001) & (y_true < 0.005)
        if ultra_focused_range.sum() > 0:
            ultra_range_weight = 3.0
            rel_error = torch.where(
                ultra_focused_range,
                rel_error * ultra_range_weight,
                rel_error
            )
            
            # Add special "within 25%" loss component for this range
            within_25pct = torch.where(
                ultra_focused_range & (rel_error <= 0.25),
                torch.zeros_like(rel_error, device=rel_error.device),
                torch.where(
                    ultra_focused_range,
                    torch.pow(rel_error - 0.25, 2),
                    torch.zeros_like(rel_error, device=rel_error.device)
                )
            )
            rel_error = rel_error + within_25pct
        
        # Apply weights
        combined_weights = log_weights * range_weights
        weighted_mse = (mse_loss * combined_weights).mean()
        weighted_rel = (rel_error * combined_weights).mean() if non_zero_mask.sum() > 0 else torch.tensor(0.0, device=mse_loss.device)
        
        # Add log-space loss component
        log_mse = torch.tensor(0.0, device=mu.device)
        slope_penalty = torch.tensor(0.0, device=mu.device)
        intercept_penalty = torch.tensor(0.0, device=mu.device)
        
        non_zero_mask_both = (y_true > epsilon) & (mu > epsilon)
        if non_zero_mask_both.sum() > 0:
            log_pred = torch.log10(mu[non_zero_mask_both])
            log_true = torch.log10(y_true[non_zero_mask_both])
            
            # MSE in log space
            log_mse = F.mse_loss(log_pred, log_true)
            
            # Penalize deviation from slope=1 and intercept=0 in log space
            if non_zero_mask_both.sum() > 1:
                # Simple linear regression coefficients
                x_mean = log_true.mean()
                y_mean = log_pred.mean()
                
                numerator = ((log_true - x_mean) * (log_pred - y_mean)).sum()
                denominator = ((log_true - x_mean) ** 2).sum() + epsilon
                
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean  # Calculate the intercept
                
                # Penalties for slope and intercept
                slope_penalty = (slope - 1.0) ** 2 * 4.0
                intercept_penalty = intercept ** 2 * 5.0  # Penalize deviation from zero
                
                # Calculate residuals in log space
                residuals = log_pred - log_true
                
                # Group by concentration range for bias correction
                low_mask = log_true < -3.0
                mid_mask = (log_true >= -3.0) & (log_true < -1.5)
                high_mask = log_true >= -1.5
                
                # Calculate mean residual by group
                low_mean = residuals[low_mask].mean() if low_mask.sum() > 0 else torch.tensor(0.0, device=mu.device)
                mid_mean = residuals[mid_mask].mean() if mid_mask.sum() > 0 else torch.tensor(0.0, device=mu.device)
                high_mean = residuals[high_mask].mean() if high_mask.sum() > 0 else torch.tensor(0.0, device=mu.device)
                
                # Add bias and pattern penalty
                bias_penalty = low_mean**2 + mid_mean**2 + high_mean**2
                pattern_penalty = torch.abs(low_mean - mid_mean) + torch.abs(mid_mean - high_mean)
                
                # Add to slope penalty
                slope_penalty = slope_penalty + 2.0 * bias_penalty + 1.5 * pattern_penalty
        
        # Zero-concentration penalty
        zero_mask = (y_true < epsilon)
        zero_penalty = self.zero_penalty * mu[zero_mask].mean() if zero_mask.sum() > 0 else torch.tensor(0.0, device=mse_loss.device)
        
        # Control sample penalty
        control_loss = torch.tensor(0.0, device=mse_loss.device)
        if control_mask is not None and control_mask.sum() > 0:
            control_loss = self.control_penalty * mu[control_mask].mean()
        
        # Zero probability supervision (if available)
        zero_prob_loss = torch.tensor(0.0, device=mse_loss.device)
        if zero_prob is not None:
            zero_target = (y_true < epsilon).float()
            zero_prob_loss = F.binary_cross_entropy(zero_prob.squeeze(), zero_target.squeeze()) * 5.0
            
            # Additional loss for control samples
            if control_mask is not None and control_mask.sum() > 0:
                control_zero_loss = F.binary_cross_entropy(
                    zero_prob[control_mask].squeeze(),
                    torch.ones(control_mask.sum(), device=zero_prob.device)
                ) * 10.0
                zero_prob_loss = zero_prob_loss + control_zero_loss
        
        # Calibration loss
        calibration_loss = torch.tensor(0.0, device=mse_loss.device)
        if uncertainty is not None:
            z_scores = torch.abs(mu - y_true) / (uncertainty + epsilon)
            calibration_loss = F.smooth_l1_loss(z_scores, torch.ones_like(z_scores, device=z_scores.device) * 1.96)
        
        # Monotonicity regularization
        batch_size = y_true.shape[0]
        monotonicity_penalty = torch.tensor(0.0, device=mse_loss.device)
        if batch_size > 1:
            sorted_targets, indices = torch.sort(y_true.squeeze(), dim=0)
            sorted_preds = mu.squeeze()[indices]
            
            # Only penalize when predictions decrease as targets increase
            monotonicity_penalty = torch.clamp(sorted_preds[:-1] - sorted_preds[1:], min=0).mean() * 2.0
        
        # Total loss with strong weight on log-space accuracy and explicit intercept penalty
        total_loss = (weighted_mse + 
            1.5 * weighted_rel +  
            self.log_space_weight * log_mse +       
            slope_penalty + 
            intercept_penalty +  # Added explicit intercept penalty
            zero_penalty + 
            control_loss + 
            zero_prob_loss +
            0.2 * calibration_loss + 
            0.8 * monotonicity_penalty)
        
        return total_loss

class ResidualBiasCorrectionLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        
        self.correction_network = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Tanh()
        )
        
        self.correction_scale = nn.Parameter(torch.tensor(0.05))
    
    def forward(self, features, initial_pred):
        log_pred = torch.log10(torch.clamp(initial_pred, 1e-6, 1.0))
        input_features = torch.cat([features, log_pred], dim=1)
        
        correction = self.correction_network(input_features) * self.correction_scale
        
        corrected_pred = initial_pred * torch.exp(correction)
        
        return torch.clamp(corrected_pred, 0.0, 1.0)
    
class DynamicMarkerPruning(nn.Module):
    def __init__(self, low_coverage_threshold=3.0):
        super().__init__()
        self.low_coverage_threshold = low_coverage_threshold
    
    def forward(self, marker_values, coverage):
        # Create dynamic pruning mask
        low_coverage_mask = coverage < self.low_coverage_threshold
        
        # Apply the mask by zeroing out low coverage marker values
        marker_values_pruned = marker_values.clone()
        marker_values_pruned[low_coverage_mask] = 0.0
        
        return marker_values_pruned

class EnhancedCancerDetectionModel(nn.Module):
    def __init__(self, num_markers, feature_dim=128, num_heads=8, num_layers=3, 
                 dropout_rate=0.2, min_reliable_coverage=5.0):
        super().__init__()
        
        self.num_markers = num_markers
        self.feature_dim = feature_dim
        self.min_reliable_coverage = min_reliable_coverage
        
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.log_value_embedding = nn.Linear(1, feature_dim // 2)
        
        self.value_bn = nn.BatchNorm1d(feature_dim // 2)
        self.coverage_bn = nn.BatchNorm1d(feature_dim // 2)
        self.log_value_bn = nn.BatchNorm1d(feature_dim // 2)
        
        self.feature_projection = nn.Linear(feature_dim * 3 // 2, feature_dim)
        
        self.marker_pos_embedding = nn.Parameter(torch.randn(1, num_markers, feature_dim) * 0.02)
        
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
        
        self.reliability_weight = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.attention = nn.Linear(feature_dim, 1)
        
        self.concentration_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.low_concentration_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.ultra_low_concentration_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.concentration_gate = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()
        )
        
        self.bias_correction = ResidualBiasCorrectionLayer(feature_dim)
        
        # Add zero anchoring layer
        self.zero_anchoring = ZeroAnchoringLayer(feature_dim)
        
        self.register_buffer('calibration', torch.ones(1))
        self.register_buffer('clinical_threshold', torch.tensor(0.001))
        
        # Add dynamic marker pruning module
        self.marker_pruning = DynamicMarkerPruning(
            low_coverage_threshold=min_reliable_coverage
        )
    
    def forward(self, marker_values, coverage):
        # Apply dynamic marker pruning
        marker_values_pruned = self.marker_pruning(marker_values, coverage)
        
        # Apply more aggressive coverage-based reliability weighting
        # Exponential penalty for low coverage
        coverage_reliability = 1.0 - torch.exp(-coverage / self.min_reliable_coverage)
        coverage_reliability = torch.clamp(coverage_reliability, 0.01, 1.0)
        
        # Apply stronger dampening to marker values based on coverage
        marker_values_weighted = marker_values_pruned * coverage_reliability
        
        # Missing mask (after pruning)
        missing_mask = (coverage == 0)
        unreliable_mask = (coverage < self.min_reliable_coverage) & ~missing_mask
        combined_mask = missing_mask | unreliable_mask
        
        # Ensure NaN values are handled
        marker_values_weighted = torch.nan_to_num(marker_values_weighted, nan=0.0)
        
        batch_size, num_markers = marker_values_weighted.shape
        
        value_features = self.value_embedding(marker_values_weighted.unsqueeze(-1))
        value_features = value_features.reshape(batch_size * num_markers, -1)
        value_features = self.value_bn(value_features)
        value_features = value_features.reshape(batch_size, num_markers, -1)
        
        log_values = torch.log1p(marker_values_weighted * 100)
        log_features = self.log_value_embedding(log_values.unsqueeze(-1))
        log_features = log_features.reshape(batch_size * num_markers, -1)
        log_features = self.log_value_bn(log_features)
        log_features = log_features.reshape(batch_size, num_markers, -1)
        
        log_coverage = torch.log1p(coverage).unsqueeze(-1)
        coverage_features = self.coverage_embedding(log_coverage)
        coverage_features = coverage_features.reshape(batch_size * num_markers, -1)
        coverage_features = self.coverage_bn(coverage_features)
        coverage_features = coverage_features.reshape(batch_size, num_markers, -1)
        
        features = torch.cat([value_features, coverage_features, log_features], dim=-1)
        features = self.feature_projection(features)
        features = features + self.marker_pos_embedding
        
        transformer_output = self.transformer_encoder(
            features, 
            src_key_padding_mask=combined_mask
        )
        
        # Enhanced reliability weighting with stronger coverage dependence
        reliability = self.reliability_weight(log_coverage)
        reliability = reliability * coverage_reliability.unsqueeze(-1)  # Enhanced reliability weighting
        
        attention_scores = self.attention(transformer_output).squeeze(-1)
        attention_scores = attention_scores * reliability.squeeze(-1)
        attention_scores = attention_scores.masked_fill(missing_mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)
        
        standard_pred = self.concentration_head(aggregated)
        low_conc_pred = self.low_concentration_head(aggregated)
        ultra_low_pred = self.ultra_low_concentration_head(aggregated)
        
        standard_pred = F.softplus(standard_pred) * 0.1
        log_low_pred = F.softplus(low_conc_pred) * 0.01
        log_ultra_low_pred = F.softplus(ultra_low_pred) * 0.001
        
        gates = self.concentration_gate(aggregated)
        ultra_low_gate = 1.0 - gates.sum(dim=1, keepdim=True)
        
        concentration = (
            gates[:, 0:1] * standard_pred + 
            gates[:, 1:2] * log_low_pred + 
            ultra_low_gate * log_ultra_low_pred
        )
        
        concentration = self.bias_correction(aggregated, concentration)
        
        # Apply zero anchoring with enhanced dampening
        concentration, zero_prob = self.zero_anchoring(aggregated, concentration)
        
        # Apply additional coverage-based dampening at the final prediction stage
        mean_coverage = coverage.mean(dim=1, keepdim=True)
        coverage_factor = torch.clamp(mean_coverage / (self.min_reliable_coverage * 2.0), 0.2, 1.0)
        
        # Apply coverage-based dampening
        concentration = concentration * coverage_factor
        
        concentration = torch.clamp(concentration, 0.0, 1.0)
        
        uncertainty = self.uncertainty_head(aggregated)
        
        return concentration, uncertainty, attention_weights, zero_prob
    
    def get_estimate_and_ci(self, mu, uncertainty, ci_level=0.95):
        # Apply clinical threshold to force low values to zero
        mu_thresholded = torch.where(mu >= self.clinical_threshold, mu, torch.zeros_like(mu))
        
        scaled_uncertainty = uncertainty * self.calibration
        
        z_score = torch.tensor(1.96) if ci_level == 0.95 else torch.tensor(
            torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + ci_level) / 2))
        )
        
        lower = torch.clamp(mu_thresholded - z_score * scaled_uncertainty, min=0.0)
        upper = torch.clamp(mu_thresholded + z_score * scaled_uncertainty, max=1.0)
        
        ci = torch.cat([lower, upper], dim=1)
        
        return mu_thresholded, ci, scaled_uncertainty
  
    def calibrate(self, val_loader, device='cpu'):
        self.eval()
        
        best_factor = 1.0
        best_error = float('inf')
        
        with torch.no_grad():
            for factor in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.5]:
                coverage_error = 0
                n_batches = 0
                
                for batch_data in val_loader:
                    if len(batch_data) == 4:
                        marker_values, coverage, y_true, _ = batch_data
                    else:
                        marker_values, coverage, y_true = batch_data
                    
                    marker_values = marker_values.to(device)
                    coverage = coverage.to(device)
                    y_true = y_true.to(device)
                    
                    mu, uncertainty, _, _ = self(marker_values, coverage)
                    
                    uncertainty_calibrated = uncertainty * factor
                    
                    z_score = 1.96
                    lower = torch.clamp(mu - z_score * uncertainty_calibrated, min=0.0)
                    upper = torch.clamp(mu + z_score * uncertainty_calibrated, max=1.0)
                    
                    in_ci = (y_true >= lower) & (y_true <= upper)
                    ci_coverage = in_ci.float().mean().item()
                    
                    error = abs(ci_coverage - 0.95)
                    coverage_error += error
                    n_batches += 1
                
                avg_error = coverage_error / n_batches
                if avg_error < best_error:
                    best_error = avg_error
                    best_factor = factor
        
        with torch.no_grad():
            self.calibration.copy_(torch.tensor([best_factor]))
        
        return {
            'calibration_factor': best_factor,
            'coverage_error': float(best_error)
        }
    
    def calibrate_clinical_threshold(self, val_loader, device='cpu', target_metric='concentration_aware', target_value=0.95):
        self.eval()
        all_preds = []
        all_targets = []
        all_zero_probs = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data
                else:
                    marker_values, coverage, y_true = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                mu, _, _, zero_prob = self(marker_values, coverage)
                
                all_preds.extend(mu.cpu().numpy().flatten())
                all_targets.extend(y_true.cpu().numpy().flatten())
                all_zero_probs.extend(zero_prob.cpu().numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_zero_probs = np.array(all_zero_probs)
        
        if target_metric == 'specificity':
            negatives = all_targets < 0.001
            if negatives.sum() > 0:
                negative_preds = all_preds[negatives]
                threshold = np.percentile(negative_preds, target_value * 100)
        
        elif target_metric == 'concentration_aware':
            thresholds = np.linspace(0.0001, 0.01, 100)
            best_score = -np.inf
            best_threshold = 0.001
            
            for threshold in thresholds:
                detected = all_preds >= threshold
                true_positives = detected & (all_targets >= 0.001)
                true_negatives = ~detected & (all_targets < 0.001)
                
                if (all_targets >= 0.001).sum() == 0 or (all_targets < 0.001).sum() == 0:
                    continue
                
                sensitivity = true_positives.sum() / (all_targets >= 0.001).sum()
                specificity = true_negatives.sum() / (all_targets < 0.001).sum()
                
                # Higher weight to specificity to ensure controls are handled better
                # Increased from 0.7 to 0.85 for stronger emphasis on controls
                weighted_accuracy = 0.85 * specificity + 0.15 * sensitivity
                
                concentration_score = 0
                if true_positives.sum() > 0:
                    tp_preds = all_preds[true_positives]
                    tp_targets = all_targets[true_positives]
                    
                    rel_errors = np.abs(tp_preds - tp_targets) / tp_targets
                    within_25_pct = (rel_errors <= 0.25).mean()
                    within_50_pct = (rel_errors <= 0.50).mean()
                    
                    if tp_preds.size > 3:
                        log_corr = np.corrcoef(np.log10(tp_preds + 1e-6), 
                                            np.log10(tp_targets + 1e-6))[0, 1]
                        log_x = np.log10(tp_targets + 1e-6)
                        log_y = np.log10(tp_preds + 1e-6)
                        coeffs = np.polyfit(log_x, log_y, 1)
                        slope = coeffs[0]
                        
                        slope_penalty = 1 - abs(slope - 1)
                    else:
                        log_corr = 0
                        slope_penalty = 0
                    
                    concentration_score = (
                        0.4 * within_25_pct +
                        0.2 * within_50_pct +
                        0.2 * log_corr +
                        0.2 * slope_penalty
                    )
                
                # Weighted score with emphasis on specificity and accuracy
                score = (
                    0.5 * specificity +  # Higher weight on specificity
                    0.2 * sensitivity +  # Lower weight on sensitivity
                    0.3 * concentration_score
                )
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            threshold = best_threshold
        
        detected = all_preds >= threshold
        true_positives = detected & (all_targets >= 0.001)
        false_positives = detected & (all_targets < 0.001)
        true_negatives = ~detected & (all_targets < 0.001)
        false_negatives = ~detected & (all_targets >= 0.001)
        
        sensitivity = true_positives.sum() / (all_targets >= 0.001).sum() if (all_targets >= 0.001).sum() > 0 else 0
        specificity = true_negatives.sum() / (all_targets < 0.001).sum() if (all_targets < 0.001).sum() > 0 else 0
        
        concentration_metrics = {}
        if detected.sum() > 0:
            detected_preds = all_preds[detected]
            detected_targets = all_targets[detected]
            
            concentration_metrics['mae'] = np.mean(np.abs(detected_preds - detected_targets))
            
            tp_mask = detected_targets >= 0.001
            if tp_mask.sum() > 0:
                tp_preds = detected_preds[tp_mask]
                tp_targets = detected_targets[tp_mask]
                rel_errors = np.abs(tp_preds - tp_targets) / tp_targets
                
                concentration_metrics['tp_within_25_pct'] = float((rel_errors <= 0.25).mean())
                concentration_metrics['tp_within_50_pct'] = float((rel_errors <= 0.50).mean())
                concentration_metrics['tp_median_rel_error'] = float(np.median(rel_errors))
        
        self.clinical_threshold.copy_(torch.tensor(threshold))
        
        return {
            'threshold': float(threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(true_positives.sum() / detected.sum()) if detected.sum() > 0 else 0,
            'npv': float(true_negatives.sum() / (~detected).sum()) if (~detected).sum() > 0 else 0,
            'concentration_metrics': concentration_metrics
        }
    
    def predict_with_clinical_threshold(self, marker_values, coverage):
        concentration, uncertainty, attention_weights, zero_prob = self.forward(marker_values, coverage)
        
        # Dynamic thresholding based on coverage
        mean_coverage = coverage.mean(dim=1, keepdim=True)
        low_coverage_mask = mean_coverage < (self.min_reliable_coverage * 1.5)
        
        # Increase threshold for low coverage samples
        dynamic_threshold = torch.where(
            low_coverage_mask,
            self.clinical_threshold * 2.0,  # Double threshold for low coverage
            self.clinical_threshold
        )
        
        # Apply zero probability-based adjustment
        is_likely_zero = zero_prob > 0.7
        concentration_adjusted = torch.where(
            is_likely_zero,
            torch.zeros_like(concentration),
            concentration
        )
        
        # Apply the clinical threshold with the dynamic adjustment
        is_detected = concentration_adjusted >= dynamic_threshold
        
        # Zero out anything below threshold
        concentration_thresholded = torch.where(is_detected, concentration_adjusted, torch.zeros_like(concentration_adjusted))
        
        return concentration_thresholded, uncertainty, attention_weights, is_detected

class MarkerImportanceAnalyser:
    def __init__(self, model):
        self.model = model
    
    def get_marker_importance(self, dataloader, top_k=20, stratify_by_concentration=True):
        self.model.eval()
        all_attentions = []
        all_concentrations = []
        
        with torch.no_grad():
            for marker_values, coverage, y_true in dataloader:
                _, _, attention_weights = self.model(marker_values, coverage)
                
                all_attentions.append(attention_weights)
                all_concentrations.append(y_true)
        
        attention_weights = torch.cat(all_attentions, dim=0)
        concentrations = torch.cat(all_concentrations, dim=0)
        
        avg_attention = attention_weights.mean(dim=0)
        
        top_k_indices = torch.topk(avg_attention, k=min(top_k, len(avg_attention))).indices
        top_k_weights = avg_attention[top_k_indices]
        
        conc_stratified = None
        if stratify_by_concentration:
            ranges = [
                ("ultra_low", 0.0, 0.001),
                ("very_low", 0.001, 0.005),
                ("low", 0.005, 0.01),
                ("medium_low", 0.01, 0.05),
                ("medium", 0.05, 0.1),
                ("high", 0.1, 1.0)
            ]
            
            conc_stratified = {}
            for name, low, high in ranges:
                mask = (concentrations >= low) & (concentrations < high)
                
                if not mask.any():
                    continue
                    
                range_attention = attention_weights[mask.squeeze()].mean(dim=0)
                
                range_top_indices = torch.topk(range_attention, k=min(top_k, len(range_attention))).indices
                range_top_weights = range_attention[range_top_indices]
                
                conc_stratified[name] = {
                    "count": int(mask.sum().item()),
                    "indices": range_top_indices.cpu().numpy(),
                    "weights": range_top_weights.cpu().numpy()
                }
        
        return top_k_indices.cpu().numpy(), top_k_weights.cpu().numpy(), conc_stratified