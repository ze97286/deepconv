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

class ScalableMarkerAggregator(nn.Module):
    """
    Simplified O(n) replacement for transformer encoder
    Much simpler than original - just marker-wise processing + single attention
    """
    def __init__(self, feature_dim=16, dropout_rate=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple marker-wise processing (no grouping complexity)
        self.marker_processor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Single self-attention layer for inter-marker communication
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=2,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, features, key_padding_mask=None):
        """
        Simplified forward pass - much closer to original transformer
        Args:
            features: [batch_size, num_markers, feature_dim]
            key_padding_mask: [batch_size, num_markers] - True for masked positions
        """
        # Step 1: Process each marker independently (like feedforward in transformer)
        processed = self.marker_processor(features)
        
        # Step 2: Single self-attention for inter-marker communication
        # This is O(n²) but still much faster than full transformer stack
        attended_features, _ = self.self_attention(
            processed, processed, processed, 
            key_padding_mask=key_padding_mask
        )
        
        # Step 3: Residual connection
        output = processed + attended_features
        
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
        
        # Multi-modal feature embedding + batch normalisation
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.log_value_embedding = nn.Linear(1, feature_dim // 2)
        self.value_bn = nn.BatchNorm1d(feature_dim // 2)
        self.coverage_bn = nn.BatchNorm1d(feature_dim // 2)
        self.log_value_bn = nn.BatchNorm1d(feature_dim // 2)
        
        # Feature projection and marker identity embedding
        self.feature_projection = nn.Linear(feature_dim * 3 // 2, feature_dim)

        # Marker Identity Embedding allows the model to learn the relative importance weights - markers can have different SNR profiles, 
        # and some may be more reliable or consistent across samples - the embedding can help adjust for these technical differences.
        # Provides positional/identity information for marker differentiation in O(n) aggregation.
        self.marker_identity_embedding = nn.Parameter(torch.randn(1, num_markers, feature_dim) * 0.02)
        
        # Scalable marker aggregator (O(n) replacement for transformer)
        # Input [batch × 10k × 16]
        #   │
        #   ↓
        # Step 1: Assign markers to groups (64 groups) - O(n)
        #   │
        #   ↓
        # Step 2: Process within groups - O(n)
        #   │
        #   ↓
        # Step 3: Cross-group attention - O(groups²) = O(64²) = O(4096) << O(10k²)
        #   │
        #   ↓
        # Step 4: Broadcast back to markers - O(n)
        #   │
        #   ↓
        # Output [batch × 10k × 16]
        self.scalable_aggregator = ScalableMarkerAggregator(
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )

        # the critical bridge between marker-level processing and sample-level representation.
        self.attention = nn.Linear(feature_dim, 1)
        
        # mixture of experts - 3 heads each specialising in a different concentration range
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
        
        # concentration gate is merging the outputs of the three heads and deciding which one to use for a given sample
        self.concentration_gate = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )
        
        # estimation of uncertainty
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()
        )
        
        # Small factor for systematic bias correction
        self.bias_correction = ResidualBiasCorrectionLayer(feature_dim)
        
        # Zero anchoring layer to distinguish between true zero and trace concentration
        self.zero_anchoring = ZeroAnchoringLayer(feature_dim)
        
        self.register_buffer('calibration', torch.ones(1))
        self.register_buffer('clinical_threshold', torch.tensor(0.001))
        
    
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
        features = features + self.marker_identity_embedding
        
        aggregated_output = self.scalable_aggregator(
            features, 
            key_padding_mask=combined_mask
        )
        
        # Enhanced reliability weighting with stronger coverage dependence
        reliability = coverage_reliability.unsqueeze(-1)  
        
        attention_scores = self.attention(aggregated_output).squeeze(-1)
        attention_scores = attention_scores * reliability.squeeze(-1)
        attention_scores = attention_scores.masked_fill(missing_mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * aggregated_output, dim=1)
        
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
        # Softer dampening - changed from 0.2-1.0 to 0.5-1.0 range
        coverage_factor = torch.clamp(mean_coverage / (self.min_reliable_coverage * 1.5), 0.5, 1.0)
        
        # Apply coverage-based dampening
        concentration = concentration * coverage_factor
        concentration = torch.clamp(concentration, 0.0, 1.0)
        
        # Calculate uncertainty
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

