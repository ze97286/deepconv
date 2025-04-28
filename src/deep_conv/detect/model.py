import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import math
import scipy.stats as stats
import numpy as np


class EnhancedCancerDetectionModel(nn.Module):
    """
    Improved DL model for cancer detection with enhanced concentration handling
    and better calibration capabilities.
    """
    def __init__(self, num_markers, feature_dim=128, num_heads=8, num_layers=3, 
             dropout_rate=0.3, detection_thresholds=(0.001, 0.01, 0.05),
             focal_weight_factor=50, low_concentration_threshold=0.01,
             l2_weight=0.05, marker_specific_bg=False, 
             min_reliable_coverage=5.0):
        super().__init__()
        
        # Store configuration
        self.detection_thresholds = detection_thresholds
        self.num_markers = num_markers
        self.focal_weight_factor = focal_weight_factor
        self.low_concentration_threshold = low_concentration_threshold
        self.l2_weight = l2_weight
        self.marker_specific_bg = marker_specific_bg
        self.min_reliable_coverage = min_reliable_coverage
            
        # Embedding components
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
        
        # Transformer components
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout_rate,
            activation=F.gelu,  # Using GELU for smoother gradients
            batch_first=True,
            norm_first=True  # Pre-norm helps training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Create a secondary encoder for low concentration focus
        low_conc_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.low_conc_encoder = nn.TransformerEncoder(low_conc_encoder_layer, num_layers=1)  # Single layer instead of 2
        
        # Readout components with enhanced signal handling
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Main prediction components for concentration estimation
        self.mu_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.phi_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()  # Ensures positive concentration parameter
        )
        
        # Low concentration specialist head
        self.low_mu_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.low_phi_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()
        )
        
        # Binary detection heads
        self.detection_heads = nn.ModuleList()
        for _ in detection_thresholds:
            head = nn.Sequential(
                nn.Linear(feature_dim + 2, feature_dim),  # +2 for uncertainty feature & coverage
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 1),
                nn.Sigmoid()
            )
            # Initialize with negative bias for lower initial predictions
            with torch.no_grad():
                head[-2].bias.data.fill_(-1.0)
            self.detection_heads.append(head)
        
        # Improved reliability weighting component - more aggressive for low coverage
        self.reliability_weight = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Calibration components
        self.calibration = nn.Parameter(torch.ones(1) * 0.5)  # Start with lower calibration
        self.low_calibration = nn.Parameter(torch.ones(1) * 0.5)  # Start with lower calibration
        
        # Dropout for regularisation
        self.dropout = nn.Dropout(dropout_rate)

        # Coverage attention for reliability weighting
        self.coverage_attention = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Background correction parameters with initializations
        if marker_specific_bg:
            # One background parameter per marker
            self.background_level = nn.Parameter(torch.ones(1, num_markers) * 0.05)
        else:
            # Global background parameter
            self.background_level = nn.Parameter(torch.tensor([0.05]))
        
        # Bias prediction components toward low values initially
        with torch.no_grad():
            self.mu_head[-2].bias.data.fill_(-2.0)
            self.low_mu_head[-2].bias.data.fill_(-2.0)
        
    def forward(self, marker_values, coverage, y_true=None, control_mask=None):
        B, M = marker_values.shape
        
        # Create mask for missing values (where coverage = 0) AND low coverage regions
        mask_missing = (coverage == 0)  # [B, M]
        mask_low_cov = (coverage < self.min_reliable_coverage)  # Low coverage mask
        mask = mask_missing | mask_low_cov  # Combined mask for unreliable regions
        
        # Enhanced reliability weighting based on coverage
        # Log-transform for better scaling of coverage values
        log_coverage = torch.log1p(coverage).unsqueeze(-1)
        
        # Calculate reliability weights - more aggressive for low coverage regions
        reliability_weight = self.reliability_weight(log_coverage)
        
        # Quadratic scaling for sharper dropoff below threshold
        coverage_reliability = torch.pow(
            torch.clamp(coverage.unsqueeze(-1) / self.min_reliable_coverage, 0.0, 1.0), 
            2  # Quadratic power for sharper falloff
        )
        reliability_weight = reliability_weight * coverage_reliability
        
        # Handle NaN values in marker_values
        marker_values = torch.nan_to_num(marker_values, nan=0.5)  # Replace NaN with 0.5 (neutral)
        
        # Embed marker values and coverage separately
        value_features = self.value_embedding(marker_values.unsqueeze(-1))  # [B, M, feature_dim//2]
        coverage_features = self.coverage_embedding(log_coverage)  # [B, M, feature_dim//2]
        
        # Combine features
        features = torch.cat([value_features, coverage_features], dim=-1)  # [B, M, feature_dim]
        features = self.feature_projection(features)  # [B, M, feature_dim]
        
        # Create padding mask for transformer (True indicates positions to mask)
        padding_mask = mask  # [B, M]
        
        # Apply main transformer with masking
        transformer_output = self.transformer_encoder(
            features, 
            src_key_padding_mask=padding_mask
        )  # [B, M, feature_dim]
        
        # Apply simplified low concentration encoder
        low_conc_output = self.low_conc_encoder(
            features,
            src_key_padding_mask=padding_mask
        )  # [B, M, feature_dim]
        
        # Apply attention mechanism with more aggressive reliability weighting
        attention_scores = self.attention(transformer_output).squeeze(-1)  # [B, M]
        
        # Apply reliability weights to attention scores - makes low coverage markers much less influential
        attention_scores = attention_scores * reliability_weight.squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask, -1e9)  # Set masked positions to large negative
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, M]
        
        # Aggregate features with attention weights
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)  # [B, feature_dim]
        aggregated = self.dropout(aggregated)
        
        # Direct prediction without Beta distribution complexities
        mu = self.mu_head(aggregated)  # [B, 1]
        
        # Force background subtraction
        if self.marker_specific_bg:
            # Apply marker-specific background correction using attention weights
            marker_bg = torch.matmul(attention_weights, self.background_level.squeeze(0))
            marker_bg = marker_bg.unsqueeze(1)
            mu = torch.clamp(mu - marker_bg, min=0.0)
        else:
            # Apply global background correction
            mu = torch.clamp(mu - self.background_level, min=0.0)
        
        # Add additional forced background subtraction
        mu = torch.clamp(mu - 0.01, min=0.0)
        
        # Create a placeholder for phi (concentration parameter)
        phi = torch.ones_like(mu) * 10.0
        
        # Calculate uncertainty for detection heads (placeholder since we're not using Beta distribution)
        uncertainty = torch.ones_like(mu) * 0.1
        
        # Enhanced features for detection heads (including uncertainty and average coverage)
        # IMPORTANT: This must match the expected input dimension of detection_heads
        detection_features = torch.cat([aggregated, uncertainty, torch.mean(coverage, dim=1, keepdim=True) / 100.0], dim=1)
        
        # Get detection probabilities for each threshold
        detection_probs = [head(detection_features) for head in self.detection_heads]
        
        # Return with control_mask if provided for contrastive learning
        if y_true is not None and control_mask is not None:
            return mu, phi, detection_probs, attention_weights, y_true, control_mask
        elif y_true is not None:
            return mu, phi, detection_probs, attention_weights, y_true
            
        return mu, phi, detection_probs, attention_weights
    
    # In compute_loss, add weighted BCE for detection thresholds:
    def compute_loss(self, mu, phi, y_true, control_mask=None, epsilon=1e-6):
        # Keep existing MSE loss
        mse_loss = F.mse_loss(mu, y_true, reduction='none')
        
        # Focal weighting - but more balanced
        weight = 1.0 + 2.0 * (1.0 - y_true)  # Reduced from 5.0 to 2.0
        weighted_loss = (mse_loss * weight).mean()
        
        # Reduce zero-concentration penalty
        zero_penalty = 2.0 * (mu * (y_true < epsilon).float()).mean() if (y_true < epsilon).sum() > 0 else 0.0
        
        # Reduce control sample penalty
        control_loss = 0.0
        if control_mask is not None and control_mask.sum() > 0:
            control_loss = 10.0 * torch.mean(mu[control_mask])  # Reduced from 20.0
        
        # NEW: Add balance penalty to encourage some positive predictions
        # This penalizes if average predictions are too far from average targets
        avg_pred = mu.mean()
        avg_target = y_true.mean()
        balance_penalty = 5.0 * torch.abs(avg_pred - avg_target)
        
        # Total loss
        total_loss = weighted_loss + zero_penalty + control_loss + balance_penalty
        
        return total_loss

    def get_estimate_and_ci(self, mu, phi, ci_level=0.95):
        """
        Get point estimate and confidence interval for concentration
        """
        phi_calibrated = phi * self.calibration
        alpha = mu * phi_calibrated
        beta = (1 - mu) * phi_calibrated
        
        # Move tensors to CPU and convert to numpy for scipy
        alpha_np = alpha.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        
        # Initialise tensors for results
        lower = torch.zeros_like(mu)
        upper = torch.zeros_like(mu)
        
        # Calculate CI bounds for each sample
        for i in range(len(alpha_np)):
            a_val = float(alpha_np[i])
            b_val = float(beta_np[i])
            
            # Handle potential numerical issues
            if a_val <= 0 or b_val <= 0:
                lower[i] = 0.0
                upper[i] = 1.0
            else:
                try:
                    lower[i] = torch.tensor(stats.beta.ppf((1 - ci_level) / 2, a_val, b_val))
                    upper[i] = torch.tensor(stats.beta.ppf(1 - (1 - ci_level) / 2, a_val, b_val))
                except:
                    # Fallback method using approximation
                    mean = a_val / (a_val + b_val)
                    variance = (a_val * b_val) / ((a_val + b_val)**2 * (a_val + b_val + 1))
                    std_dev = np.sqrt(variance)
                    z_score = 1.96  # Approx. for 95% CI
                    
                    lower[i] = max(0.0, mean - z_score * std_dev)
                    upper[i] = min(1.0, mean + z_score * std_dev)
        
        estimate = mu
        ci = torch.cat([lower, upper], dim=1)
        
        # Calculate uncertainty (width of CI)
        uncertainty = upper - lower
        
        return estimate, ci, uncertainty
        
    def calibrate_background(self, control_loader, device):
        """
        Enhanced calibration of background level using control samples
        with more robust estimation methods
        
        Args:
            control_loader: DataLoader with control samples
            device: Device to run inference on
            
        Returns:
            Dictionary with calibration parameters
        """
        self.eval()
        all_preds = []
        all_attentions = []
        
        with torch.no_grad():
            for batch_data in control_loader:
                # Handle both 3-element and 4-element returns from DataLoader
                if len(batch_data) == 4:
                    marker_values, coverage, _, _ = batch_data
                else:
                    marker_values, coverage, _ = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                
                # Forward pass to get predictions and attention weights
                mu, _, _, attention_weights = self(marker_values, coverage)
                
                all_preds.append(mu.cpu())
                all_attentions.append(attention_weights.cpu())
                
        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0)
        all_attentions = torch.cat(all_attentions, dim=0)
        
        # Use 95th percentile instead of median for more conservative background correction
        global_bg_level = float(torch.quantile(all_preds, 0.95))
        
        # Update background level parameter
        if self.marker_specific_bg:
            # Calculate marker-specific background levels
            # This is based on marker contribution to the predictions via attention
            marker_importance = all_attentions.mean(dim=0)  # Average attention per marker
            
            # Calculate weighted background level for each marker
            weighted_bg = torch.zeros(self.num_markers)
            for i, pred in enumerate(all_preds):
                weighted_bg += all_attentions[i] * pred.item()
            
            # Normalize by attention sum
            attention_sum = all_attentions.sum(dim=0)
            mask = attention_sum > 0
            weighted_bg[mask] = weighted_bg[mask] / attention_sum[mask]
            weighted_bg[~mask] = global_bg_level
            
            # Apply bootstrap to get more robust estimates
            bootstrap_samples = 100
            bootstrap_bgs = []
            
            for _ in range(bootstrap_samples):
                # Sample with replacement
                indices = torch.randint(0, len(all_preds), (len(all_preds),))
                bootstrap_preds = all_preds[indices]
                bootstrap_attentions = all_attentions[indices]
                
                # Calculate background
                bootstrap_bg = torch.zeros(self.num_markers)
                for i, pred in enumerate(bootstrap_preds):
                    bootstrap_bg += bootstrap_attentions[i] * pred.item()
                
                # Normalize
                bootstrap_bg[mask] = bootstrap_bg[mask] / attention_sum[mask]
                bootstrap_bg[~mask] = global_bg_level
                
                bootstrap_bgs.append(bootstrap_bg)
            
            # Get 95th percentile for each marker
            bootstrap_bgs = torch.stack(bootstrap_bgs)
            percentile_bg = torch.quantile(bootstrap_bgs, 0.95, dim=0)
            
            # Set background level parameter
            with torch.no_grad():
                self.background_level.copy_(percentile_bg.unsqueeze(0))
                
            marker_bg_stats = {
                'mean': float(percentile_bg.mean()),
                'median': float(torch.median(percentile_bg)),
                'min': float(percentile_bg.min()),
                'max': float(percentile_bg.max()),
                'std': float(percentile_bg.std())
            }
            
            return {
                'global_bg_level': global_bg_level,
                'marker_specific_bg': True,
                'marker_bg_stats': marker_bg_stats
            }
        else:
            # Set global background level using 95th percentile
            with torch.no_grad():
                self.background_level.fill_(global_bg_level)
                
            return {
                'global_bg_level': global_bg_level,
                'marker_specific_bg': False
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
            for batch_data in dataloader:
                # Handle both 3-element and 4-element returns
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data  # Unpack 4 elements, ignore control_mask
                else:
                    marker_values, coverage, y_true = batch_data
                    
                if isinstance(self.model, CancerDetectionEnsemble):
                    mu, _, det_probs = self.model.forward(marker_values, coverage)
                else:
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