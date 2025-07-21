import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ZeroAnchoringLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.zero_detector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),  
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        # Learnable sharpness with more conservative init
        self.sharpness = nn.Parameter(torch.tensor(10.0))
        
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
    
class ResidualBiasCorrectionLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        
        self.correction_network = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Tanh()
        )
        
        self.correction_scale = nn.Parameter(torch.tensor(0.05))
    
    def forward(self, features, initial_pred):
        # More stable log transform with larger epsilon
        log_pred = torch.log10(torch.clamp(initial_pred, 1e-4, 1.0))
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

class DeepSetsMarkerProcessor(nn.Module):
    """
    Deep Sets architecture for cancer detection from methylation markers
    f(markers) = ρ(Σ φ(marker_i))
    """
    def __init__(self, feature_dim=16, hidden_dim=64, dropout_rate=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # φ: processes each marker individually
        self.marker_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # For compatibility with existing code, return same feature_dim
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, features, key_padding_mask=None):
        """
        Args:
            features: [batch_size, num_markers, feature_dim]
            key_padding_mask: [batch_size, num_markers] - True for masked positions
        Returns:
            processed_features: [batch_size, num_markers, feature_dim] 
        """
        batch_size, num_markers, feature_dim = features.shape
        
        # φ: Process each marker individually
        # Flatten for batch processing
        features_flat = features.reshape(-1, feature_dim)
        encoded_flat = self.marker_encoder(features_flat)
        
        # Reshape back to [batch, markers, hidden_dim]
        encoded = encoded_flat.reshape(batch_size, num_markers, self.hidden_dim)
        
        # Project back to original feature_dim for compatibility
        output = self.output_projection(encoded)
        
        # Apply mask if provided
        if key_padding_mask is not None:
            mask_expanded = key_padding_mask.unsqueeze(-1)
            output = output.masked_fill(mask_expanded, 0.0)
        
        return output

class EnhancedCancerDetectionModel(nn.Module):
    def __init__(self, num_markers, feature_dim=16,
                 dropout_rate=0.2, min_reliable_coverage=5.0):
        super().__init__()
        
        # Add dynamic marker pruning module
        self.min_reliable_coverage = min_reliable_coverage
        
        self.marker_pruning = DynamicMarkerPruning(
            low_coverage_threshold=min_reliable_coverage
        )
        
        # Minimal stable embedding system - no BatchNorm
        self.simple_projection = nn.Linear(3, feature_dim)  # [value, log_value, coverage] -> feature_dim

        
        # MINIMAL marker processor - just one linear layer for φ function
        self.marker_processor = nn.Linear(feature_dim, feature_dim)

        # the critical bridge between marker-level processing and sample-level representation.
        self.attention = nn.Linear(feature_dim, 1)
        
        # Simplified single concentration head with much smaller scale
        self.concentration_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),  # Smaller hidden layer
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim // 4, 1),
            # Remove sigmoid - let the model learn the scale naturally
        )
        
        # estimation of uncertainty
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()
        )
        
        # Simplified model - removing complex bias correction and zero anchoring for stability
        # self.bias_correction = ResidualBiasCorrectionLayer(feature_dim)
        # self.zero_anchoring = ZeroAnchoringLayer(feature_dim)
        
        self.register_buffer('calibration', torch.ones(1))
        self.register_buffer('clinical_threshold', torch.tensor(0.001))
        
        # Conservative weight initialization for stability
        self._init_weights()
    
    def _init_weights(self):
        """Ultra-conservative weight initialization for training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Ultra-small initialization variance to prevent explosion
                nn.init.xavier_normal_(module.weight, gain=0.01)  # Extremely small gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, marker_values, coverage):
        # Input validation and scaling to prevent explosions
        marker_values = torch.clamp(marker_values, 0.0, 1.0)
        coverage = torch.clamp(coverage, 0.0, 1000.0)
        
        # Apply dynamic marker pruning
        marker_values_pruned = self.marker_pruning(marker_values, coverage)
        
        # Gentler coverage-based reliability weighting for Deep Sets
        # Linear scaling instead of exponential to preserve more signal
        coverage_reliability = torch.clamp(coverage / self.min_reliable_coverage, 0.1, 1.0)
        
        # Apply stronger dampening to marker values based on coverage
        marker_values_weighted = marker_values_pruned * coverage_reliability

        # Missing mask (after pruning)
        missing_mask = (coverage == 0)
        unreliable_mask = (coverage < self.min_reliable_coverage) & ~missing_mask
        combined_mask = missing_mask | unreliable_mask
        
        # Ensure NaN values are handled
        marker_values_weighted = torch.nan_to_num(marker_values_weighted, nan=0.0)
        
        batch_size, num_markers = marker_values_weighted.shape
        
        # MINIMAL stable feature creation - no complex embeddings
        # Simple concatenated features: [value, log_value, coverage]
        features = torch.stack([
            marker_values_weighted,
            torch.log1p(marker_values_weighted),  
            torch.log1p(coverage) / 10.0  # Scale coverage down
        ], dim=-1)  # Shape: [batch, markers, 3]
        
        # Single linear projection 
        features = self.simple_projection(features)  # [batch, markers, feature_dim]
        
        # φ: Process each marker individually (Deep Sets φ function) - MINIMAL
        processed_features = self.marker_processor(features)  # Just linear transformation
        
        # Apply mask manually
        if combined_mask is not None:
            mask_expanded = combined_mask.unsqueeze(-1)
            processed_features = processed_features.masked_fill(mask_expanded, 0.0)
        
        # Enhanced reliability weighting with stronger coverage dependence
        reliability = coverage_reliability.unsqueeze(-1)  
        
        # ρ: Aggregate the set (Deep Sets ρ function via coverage-weighted averaging)
        # Use coverage reliability as weights - more biologically meaningful than learned attention
        weights = reliability.squeeze(-1)
        weights = weights.masked_fill(missing_mask, 0.0)  # Zero out missing markers
        
        # Normalize weights to sum to 1 (avoid division by zero)
        weight_sum = weights.sum(dim=1, keepdim=True)
        weight_sum = torch.clamp(weight_sum, min=1e-8)  # Prevent division by zero
        normalized_weights = weights / weight_sum
        
        # This is the Deep Sets aggregation: Σ φ(marker_i) weighted by coverage reliability
        aggregated = torch.sum(normalized_weights.unsqueeze(-1) * processed_features, dim=1)
        
        # Simplified single concentration prediction with scaling
        concentration_raw = self.concentration_head(aggregated)
        
        # Remove artificial cap - let model learn full range
        concentration = torch.sigmoid(concentration_raw) * 0.2  # Max 20% concentration
        
        # Simplified model - skip bias correction and zero anchoring for stability
        # concentration = self.bias_correction(aggregated, concentration)
        # concentration, zero_prob = self.zero_anchoring(aggregated, concentration)
        
        # Simplified coverage-based dampening
        mean_coverage = coverage.mean(dim=1, keepdim=True)
        coverage_factor = torch.clamp(mean_coverage / (self.min_reliable_coverage * 2.0), 0.3, 1.0)
        
        # Apply dampening and clamp
        concentration = concentration * coverage_factor
        concentration = torch.clamp(concentration, 0.0, 1.0)
        
        # Calculate uncertainty
        uncertainty = self.uncertainty_head(aggregated)
        
        # Create dummy zero_prob to maintain compatibility
        zero_prob = torch.zeros_like(concentration)
        
        # Return normalized_weights instead of attention_weights for compatibility
        return concentration, uncertainty, normalized_weights, zero_prob
    
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

