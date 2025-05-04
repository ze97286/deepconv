import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.stats as stats
import numpy as np

class ConcentrationFocusedLoss(nn.Module):
    def __init__(self, critical_ranges=None, zero_penalty=10.0, control_penalty=20.0):
        super().__init__()
        self.critical_ranges = critical_ranges or [(0.0001, 0.001, 1.8), (0.001, 0.01, 1.5), (0.01, 0.05, 1.2)]
        self.zero_penalty = zero_penalty
        self.control_penalty = control_penalty
    
    def forward(self, mu, uncertainty, y_true, control_mask=None):
        """
        Compute concentration-focused loss with enhanced range-specific weighting
        and monotonicity regularization
        """
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
        log_weights = torch.clamp(log_weights, 0.5, 2.0)
        
        # Apply additional weights for critical ranges
        range_weights = torch.ones_like(log_weights, device=log_weights.device)
        for low, high, weight in self.critical_ranges:
            range_mask = (y_true >= low) & (y_true < high)
            range_weights = torch.where(range_mask, torch.ones_like(range_weights, device=range_weights.device) * weight, range_weights)
        
        # Special focus on 0.1-0.5% range
        ultra_focused_range = (y_true >= 0.001) & (y_true < 0.005)
        if ultra_focused_range.sum() > 0:
            # Extra weight for relative error in this range
            ultra_range_weight = 2.0
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
                    torch.pow(rel_error - 0.25, 2),  # Quadratic penalty for > 25% error
                    torch.zeros_like(rel_error, device=rel_error.device)
                )
            )
            rel_error = rel_error + within_25pct
            
            # Additional penalty for errors in this critical range
            critical_error = torch.abs(mu[ultra_focused_range] - y_true[ultra_focused_range])
            critical_penalty = torch.mean(critical_error * torch.log10(1.0 / (y_true[ultra_focused_range] + epsilon)))
        else:
            critical_penalty = torch.tensor(0.0, device=mse_loss.device)
        
        # Apply weights to MSE and relative error
        log_weights = log_weights * range_weights
        weighted_mse = (mse_loss * log_weights).mean()
        weighted_rel = (rel_error * log_weights).mean() if non_zero_mask.sum() > 0 else torch.tensor(0.0, device=mse_loss.device)
        
        # Zero-concentration specific penalty
        zero_mask = (y_true < epsilon)
        zero_penalty = self.zero_penalty * mu[zero_mask].mean() if zero_mask.sum() > 0 else torch.tensor(0.0, device=mse_loss.device)
        
        # Control sample penalty
        control_loss = torch.tensor(0.0, device=mse_loss.device)
        if control_mask is not None and control_mask.sum() > 0:
            control_loss = self.control_penalty * mu[control_mask].mean()
        
        # Add calibration component for uncertainty estimates
        calibration_loss = torch.tensor(0.0, device=mse_loss.device)
        if uncertainty is not None:
            z_scores = torch.abs(mu - y_true) / (uncertainty + epsilon)
            calibration_loss = F.smooth_l1_loss(z_scores, torch.ones_like(z_scores, device=z_scores.device) * 1.96)
        
        # Add monotonicity regularization
        batch_size = y_true.shape[0]
        if batch_size > 1:
            sorted_targets, indices = torch.sort(y_true.squeeze(), dim=0)
            sorted_preds = mu.squeeze()[indices]
            
            # Only penalize when predictions decrease as targets increase
            monotonicity_penalty = torch.clamp(sorted_preds[:-1] - sorted_preds[1:], min=0).mean()
        else:
            monotonicity_penalty = torch.tensor(0.0, device=mse_loss.device)
        
        # Combine all components
        total_loss = (weighted_mse + 0.7 * weighted_rel + zero_penalty + control_loss + 
                     0.2 * calibration_loss + 2.0 * critical_penalty + 0.5 * monotonicity_penalty)
        
        return total_loss
     
class EnhancedCancerDetectionModel(nn.Module):
    def __init__(self, num_markers, feature_dim=128, num_heads=8, num_layers=3, 
                 dropout_rate=0.2, min_reliable_coverage=3.0):
        super().__init__()
        
        # Keep the original architecture
        self.num_markers = num_markers
        self.feature_dim = feature_dim
        self.min_reliable_coverage = min_reliable_coverage
        
        # Input embeddings
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
        
        # Position embeddings for markers
        self.marker_pos_embedding = nn.Parameter(torch.randn(1, num_markers, feature_dim) * 0.02)
        
        # Transformer encoder
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
        
        # Coverage reliability weighting
        self.reliability_weight = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Simplified attention mechanism
        self.attention = nn.Linear(feature_dim, 1)
        
        # Single concentration prediction head (remove dual-head)
        self.concentration_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()
        )
        
        # Calibration parameter
        self.register_buffer('calibration', torch.ones(1))
        self.register_buffer('clinical_threshold', torch.tensor(0.001))
    
    def forward(self, marker_values, coverage):
        # Use original forward pass
        missing_mask = (coverage == 0)
        unreliable_mask = (coverage < self.min_reliable_coverage)
        combined_mask = missing_mask | unreliable_mask
        
        marker_values = torch.nan_to_num(marker_values, nan=0.0)
        
        value_features = self.value_embedding(marker_values.unsqueeze(-1))
        log_coverage = torch.log1p(coverage).unsqueeze(-1)
        coverage_features = self.coverage_embedding(log_coverage)
        
        features = torch.cat([value_features, coverage_features], dim=-1)
        features = self.feature_projection(features)
        features = features + self.marker_pos_embedding
        
        transformer_output = self.transformer_encoder(
            features, 
            src_key_padding_mask=combined_mask
        )
        
        reliability = self.reliability_weight(log_coverage)
        
        attention_scores = self.attention(transformer_output).squeeze(-1)
        attention_scores = attention_scores * reliability.squeeze(-1)
        attention_scores = attention_scores.masked_fill(combined_mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)
        
        concentration = self.concentration_head(aggregated)
        uncertainty = self.uncertainty_head(aggregated)
        
        return concentration, uncertainty, attention_weights
    
    def get_estimate_and_ci(self, mu, uncertainty, ci_level=0.95):
        """Get point estimate and confidence interval"""
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
    
    def calibrate(self, val_loader, device='cpu'):
        """
        Calibrate the model's uncertainty estimates
        
        Args:
            val_loader: DataLoader for validation data
            device: Device to run calibration on
            
        Returns:
            dict: Dictionary containing calibration parameters
        """
        self.eval()
        
        # Test different calibration factors
        best_factor = 1.0
        best_error = float('inf')
        
        with torch.no_grad():
            # Try different calibration factors
            for factor in [0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 2.5]:
                coverage_error = 0
                n_batches = 0
                
                for batch_data in val_loader:
                    # Handle both dataset types (with or without control_mask)
                    if len(batch_data) == 4:
                        marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask
                    else:
                        marker_values, coverage, y_true = batch_data
                    
                    marker_values = marker_values.to(device)
                    coverage = coverage.to(device)
                    y_true = y_true.to(device)
                    
                    # Forward pass
                    mu, uncertainty, _ = self(marker_values, coverage)
                    
                    # Apply test calibration factor
                    uncertainty_calibrated = uncertainty * factor
                    
                    # Calculate CI
                    z_score = 1.96  # For 95% CI
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
        
        # Apply calibration factor to model
        with torch.no_grad():
            self.calibration.copy_(torch.tensor([best_factor]))
        
        return {
            'calibration_factor': best_factor,
            'coverage_error': float(best_error)
        }
    
    def calibrate_clinical_threshold(self, val_loader, device='cpu', target_metric='concentration_aware', target_value=0.95):
        """
        Calibrate decision threshold for clinical use with multiple metric options
        
        Args:
            val_loader: DataLoader containing validation data
            device: Device to run calibration on
            target_metric: Which metric to optimize ('specificity', 'ppv', 'balanced', 'concentration_aware')
            target_value: Target value for the chosen metric
        """
        self.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data
                else:
                    marker_values, coverage, y_true = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                mu, _, _ = self(marker_values, coverage)
                
                all_preds.extend(mu.cpu().numpy().flatten())
                all_targets.extend(y_true.cpu().numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        if target_metric == 'specificity':
            # Original approach - optimize for specificity
            negatives = all_targets < 0.001
            if negatives.sum() > 0:
                negative_preds = all_preds[negatives]
                threshold = np.percentile(negative_preds, target_value * 100)
        
        elif target_metric == 'concentration_aware':
            # New approach - consider both detection AND concentration accuracy
            thresholds = np.linspace(0.0001, 0.01, 100)
            best_score = -np.inf
            best_threshold = 0.001
            
            for threshold in thresholds:
                # Classification metrics
                detected = all_preds >= threshold
                true_positives = detected & (all_targets >= 0.001)
                true_negatives = ~detected & (all_targets < 0.001)
                
                sensitivity = true_positives.sum() / (all_targets >= 0.001).sum()
                specificity = true_negatives.sum() / (all_targets < 0.001).sum()
                
                # Concentration accuracy for true positives
                if true_positives.sum() > 0:
                    tp_preds = all_preds[true_positives]
                    tp_targets = all_targets[true_positives]
                    
                    # Calculate concentration accuracy metrics
                    rel_errors = np.abs(tp_preds - tp_targets) / tp_targets
                    within_25_pct = (rel_errors <= 0.25).mean()
                    within_50_pct = (rel_errors <= 0.50).mean()
                    
                    # Log-space correlation
                    log_corr = np.corrcoef(np.log10(tp_preds + 1e-6), 
                                        np.log10(tp_targets + 1e-6))[0, 1]
                else:
                    within_25_pct = 0
                    within_50_pct = 0
                    log_corr = 0
                
                # Combined score
                score = (
                    0.3 * specificity +  # Avoid false positives
                    0.2 * sensitivity +  # Detect true cases
                    0.3 * within_25_pct +  # Accurate concentration estimates
                    0.2 * log_corr  # Good correlation in log space
                )
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            threshold = best_threshold
        
        # Evaluate the chosen threshold
        detected = all_preds >= threshold
        true_positives = detected & (all_targets >= 0.001)
        false_positives = detected & (all_targets < 0.001)
        true_negatives = ~detected & (all_targets < 0.001)
        false_negatives = ~detected & (all_targets >= 0.001)
        
        sensitivity = true_positives.sum() / (all_targets >= 0.001).sum()
        specificity = true_negatives.sum() / (all_targets < 0.001).sum()
        
        # Calculate concentration accuracy for detected samples
        concentration_metrics = {}
        if detected.sum() > 0:
            detected_preds = all_preds[detected]
            detected_targets = all_targets[detected]
            
            # Overall MAE for detected samples
            concentration_metrics['mae'] = np.mean(np.abs(detected_preds - detected_targets))
            
            # Relative error metrics for true positives
            tp_mask = detected_targets >= 0.001
            if tp_mask.sum() > 0:
                tp_preds = detected_preds[tp_mask]
                tp_targets = detected_targets[tp_mask]
                rel_errors = np.abs(tp_preds - tp_targets) / tp_targets
                
                concentration_metrics['tp_within_25_pct'] = float((rel_errors <= 0.25).mean())
                concentration_metrics['tp_within_50_pct'] = float((rel_errors <= 0.50).mean())
                concentration_metrics['tp_median_rel_error'] = float(np.median(rel_errors))
        
        # Update model's clinical threshold
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
        """
        Make predictions with clinical threshold applied
        
        Returns:
            concentration: Predicted concentration
            uncertainty: Prediction uncertainty
            attention_weights: Attention weights
            is_detected: Boolean indicating if concentration exceeds clinical threshold
        """
        concentration, uncertainty, attention_weights = self.forward(marker_values, coverage)
        is_detected = concentration >= self.clinical_threshold
        
        return concentration, uncertainty, attention_weights, is_detected
    
class MarkerImportanceAnalyser:
    """Utility class to analyse marker importance"""
    def __init__(self, model):
        self.model = model
    
    def get_marker_importance(self, dataloader, top_k=20, stratify_by_concentration=True):
        """Analyse marker importance across the dataset with optional stratification"""
        self.model.eval()
        all_attentions = []
        all_concentrations = []
        
        with torch.no_grad():
            for marker_values, coverage, y_true in dataloader:
                # Get predictions and attention weights
                _, _, attention_weights = self.model(marker_values, coverage)
                
                # Store results
                all_attentions.append(attention_weights)
                all_concentrations.append(y_true)
        
        # Concatenate results
        attention_weights = torch.cat(all_attentions, dim=0)
        concentrations = torch.cat(all_concentrations, dim=0)
        
        # Average attention weights across all samples
        avg_attention = attention_weights.mean(dim=0)
        
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
