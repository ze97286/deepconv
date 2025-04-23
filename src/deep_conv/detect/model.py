import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import math
import scipy.stats as stats
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional encoding to provide marker position information
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class EnhancedCancerDetectionModel(nn.Module):
    """
    Enhanced model for cancer detection with specialized concentration handling
    and improved detection capabilities.
    """
    def __init__(self, num_markers, feature_dim=128, num_heads=8, num_layers=3, 
                 dropout_rate=0.2, use_pos_encoding=True, detection_thresholds=(0.001, 0.01, 0.05)):
        super().__init__()
        
        # Store configuration
        self.detection_thresholds = detection_thresholds
        self.num_markers = num_markers
        
        # Embedding components
        self.value_embedding = nn.Linear(1, feature_dim // 2)
        self.coverage_embedding = nn.Linear(1, feature_dim // 2)
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
        self.pos_encoding = PositionalEncoding(feature_dim) if use_pos_encoding else None
        
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
            dim_feedforward=feature_dim * 4,
            dropout=dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.low_conc_encoder = nn.TransformerEncoder(low_conc_encoder_layer, num_layers=2)
        
        # Readout components
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1)
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
        B, M = marker_values.shape
        
        # Create mask for missing values (where coverage = 0)
        mask = (coverage == 0)  # [B, M]
        
        # Handle NaN values in marker_values
        marker_values = torch.nan_to_num(marker_values, nan=0.5)  # Replace NaN with 0.5 (neutral)
        
        # Embed marker values and coverage separately
        value_features = self.value_embedding(marker_values.unsqueeze(-1))  # [B, M, feature_dim//2]
        coverage_features = self.coverage_embedding(
            torch.log1p(coverage).unsqueeze(-1)  # Log transform for better numerical stability
        )  # [B, M, feature_dim//2]
        
        # Combine features
        features = torch.cat([value_features, coverage_features], dim=-1)  # [B, M, feature_dim]
        features = self.feature_projection(features)  # [B, M, feature_dim]
        
        # Apply positional encoding if used
        if self.pos_encoding is not None:
            features = self.pos_encoding(features)
        
        # Create padding mask for transformer (True indicates positions to mask)
        padding_mask = mask  # [B, M]
        
        # Apply main transformer with masking
        transformer_output = self.transformer_encoder(
            features, 
            src_key_padding_mask=padding_mask
        )  # [B, M, feature_dim]
        
        # Apply low concentration encoder
        low_conc_output = self.low_conc_encoder(
            features,
            src_key_padding_mask=padding_mask
        )  # [B, M, feature_dim]
        
        # Apply attention mechanism (ignoring masked positions)
        attention_scores = self.attention(transformer_output).squeeze(-1)  # [B, M]
        attention_scores = attention_scores.masked_fill(mask, -1e9)  # Set masked positions to large negative
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, M]
        
        # Aggregate features with attention weights
        aggregated = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)  # [B, feature_dim]
        aggregated = self.dropout(aggregated)
        
        # Aggregate low concentration features with the same attention weights
        low_conc_aggregated = torch.sum(attention_weights.unsqueeze(-1) * low_conc_output, dim=1)
        low_conc_aggregated = self.dropout(low_conc_aggregated)
        
        # Predict parameters for Beta distribution from main encoder
        mu = self.mu_head(aggregated)  # [B, 1]
        phi = self.phi_head(aggregated) * self.calibration  # [B, 1], calibrated concentration
        
        # Predict parameters from low concentration encoder
        low_mu = self.low_mu_head(low_conc_aggregated)  # [B, 1]
        low_phi = self.low_phi_head(low_conc_aggregated) * self.low_calibration  # [B, 1]
        
        # Blend predictions based on predicted concentration
        # More weight to low_mu for low concentrations
        with torch.no_grad():
            blend_weight = torch.exp(-mu * 200)  # Weight decreases as concentration increases
        
        blended_mu = blend_weight * low_mu + (1 - blend_weight) * mu
        blended_phi = blend_weight * low_phi + (1 - blend_weight) * phi
        
        # Calculate uncertainty for detection heads
        _, _, uncertainty = self.get_estimate_and_ci(blended_mu, blended_phi)
        
        # Enhanced features for detection heads (including uncertainty)
        detection_features = torch.cat([aggregated, uncertainty], dim=1)
        
        # Get detection probabilities for each threshold
        detection_probs = [head(detection_features) for head in self.detection_heads]
        
        if y_true is not None:
            # Compute concentration loss
            loss = self.compute_loss(blended_mu, blended_phi, y_true)
            
            # Add detection losses for each threshold
            detection_losses = []
            for i, threshold in enumerate(self.detection_thresholds):
                # Convert continuous concentration to binary label
                binary_y = (y_true >= threshold).float()
                # Binary cross-entropy loss
                det_loss = F.binary_cross_entropy(detection_probs[i], binary_y)
                detection_losses.append(det_loss)
            
            # Combine losses - concentration loss + weighted sum of detection losses
            detection_loss_weight = 0.2  # How much to weight detection vs concentration
            combined_detection_loss = sum(detection_losses) / len(detection_losses)
            total_loss = loss + detection_loss_weight * combined_detection_loss
            
            return blended_mu, blended_phi, detection_probs, total_loss, attention_weights
            
        return blended_mu, blended_phi, detection_probs, attention_weights
    
    def compute_loss(self, mu, phi, y_true, epsilon=1e-6, gamma=2.0):
        """
        Compute enhanced focal Beta negative log likelihood loss with threshold emphasis
        """
        y_clipped = torch.clamp(y_true, epsilon, 1 - epsilon)
        
        # Calculate Beta distribution parameters
        alpha = mu * phi  # [B, 1]
        beta = (1 - mu) * phi  # [B, 1]
        
        # Create Beta distribution
        dist = Beta(alpha, beta)
        
        # Negative log likelihood
        nll_loss = -dist.log_prob(y_clipped)
        
        # Enhanced focal weighting that emphasizes both low concentrations
        # and samples near critical thresholds
        thresholds = torch.tensor(self.detection_thresholds).to(y_true.device)
        
        # Basic weight for low concentrations
        base_weight = torch.exp(-y_true * 100) + 1.0
        
        # Additional weight for samples near thresholds
        threshold_weight = torch.zeros_like(y_true)
        for threshold in thresholds:
            # Samples within 20% of threshold get extra weight
            # Handle threshold=0 case
            if threshold > 0:
                relative_distance = torch.abs(y_true - threshold) / threshold
                threshold_weight += torch.exp(-relative_distance * 5) * 2.0
        
        # Combine weights (cap at 5x to prevent extreme values)
        focal_weight = torch.clamp(base_weight + threshold_weight, 1.0, 5.0)
        
        # Apply focal weighting
        focal_loss = nll_loss * focal_weight
        
        # Add regularization to prevent extremely confident predictions
        reg_loss = 0.01 * torch.abs(torch.log(phi)).mean()
        
        return focal_loss.mean() + reg_loss
    
    def get_estimate_and_ci(self, mu, phi, ci_level=0.95):
        """
        Get point estimate and confidence interval using scipy
        """
        alpha = mu * phi
        beta = (1 - mu) * phi
        
        # Move tensors to CPU and convert to numpy for scipy
        alpha_np = alpha.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        
        # Initialize tensors for results
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
                lower[i] = torch.tensor(stats.beta.ppf((1 - ci_level) / 2, a_val, b_val))
                upper[i] = torch.tensor(stats.beta.ppf(1 - (1 - ci_level) / 2, a_val, b_val))
        
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


class CancerDetectionEnsemble:
    """
    Ensemble of cancer detection models for improved robustness
    """
    def __init__(self, models):
        """
        Initialize ensemble with multiple model instances
        
        Args:
            models: List of EnhancedCancerDetectionModel instances
        """
        self.models = models
        
    def forward(self, marker_values, coverage, y_true=None):
        """
        Forward pass that combines predictions from all models
        """
        all_mus = []
        all_phis = []
        all_detection_probs = []
        
        # Get predictions from all models
        for model in self.models:
            model.eval()  # Ensure evaluation mode
            if y_true is not None:
                mu, phi, det_probs, _, _ = model(marker_values, coverage, y_true)
            else:
                mu, phi, det_probs, _ = model(marker_values, coverage)
                
            all_mus.append(mu)
            all_phis.append(phi)
            all_detection_probs.append(det_probs)
        
        # Average concentration estimates
        ensemble_mu = torch.mean(torch.stack(all_mus), dim=0)
        
        # Weighted average of phi (inverse weighting by uncertainty)
        all_phi_stack = torch.stack(all_phis)
        weights = 1.0 / all_phi_stack
        ensemble_phi = torch.sum(all_phi_stack * weights, dim=0) / torch.sum(weights, dim=0)
        
        # Take max detection probability (conservative approach favoring sensitivity)
        ensemble_detection_probs = []
        for i in range(len(all_detection_probs[0])):
            probs_for_threshold = torch.stack([model_probs[i] for model_probs in all_detection_probs])
            ensemble_detection_probs.append(torch.max(probs_for_threshold, dim=0)[0])
        
        return ensemble_mu, ensemble_phi, ensemble_detection_probs
    
    def get_estimate_and_ci(self, marker_values, coverage, ci_level=0.95):
        """
        Get ensemble estimate and confidence interval
        """
        mu, phi, _ = self.forward(marker_values, coverage)
        
        # Use the first model's get_estimate_and_ci method with our ensemble params
        return self.models[0].get_estimate_and_ci(mu, phi, ci_level)
    
    def get_binary_prediction(self, marker_values, coverage, threshold_idx=1):
        """
        Get binary prediction from ensemble
        """
        mu, _, detection_probs = self.forward(marker_values, coverage)
        return (detection_probs[threshold_idx] >= 0.5).float()


class MarkerImportanceAnalyzer:
    """
    Utility class to analyze the importance of different markers
    """
    def __init__(self, model):
        self.model = model
    
    def get_marker_importance(self, dataloader, top_k=20):
        """
        Analyze marker importance across the dataset
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
    
    def analyze_detection_performance(self, dataloader, thresholds=None):
        """
        Analyze detection performance at different concentration thresholds
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