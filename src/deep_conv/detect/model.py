import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import math


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
    def __init__(self, num_markers, feature_dim=128, num_heads=8, num_layers=3, 
                 dropout_rate=0.2, use_pos_encoding=True):
        super().__init__()
        
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
        
        # Readout components
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Prediction components
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
        
        # Calibration component for uncertainty estimation
        self.calibration = nn.Parameter(torch.ones(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        self.num_markers = num_markers
        
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
        
        # Apply transformer with masking
        transformer_output = self.transformer_encoder(
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
        
        # Predict parameters for Beta distribution
        mu = self.mu_head(aggregated)  # [B, 1]
        phi = self.phi_head(aggregated) * self.calibration  # [B, 1], calibrated concentration
        
        if y_true is not None:
            loss = self.compute_loss(mu, phi, y_true)
            return mu, phi, loss, attention_weights
            
        return mu, phi, attention_weights
    
    def compute_loss(self, mu, phi, y_true, epsilon=1e-6):
        """
        Compute negative log likelihood of Beta distribution
        with additional calibration loss
        """
        y_clipped = torch.clamp(y_true, epsilon, 1 - epsilon)
        
        # Calculate Beta distribution parameters
        alpha = mu * phi  # [B, 1]
        beta = (1 - mu) * phi  # [B, 1]
        
        # Create Beta distribution
        dist = Beta(alpha, beta)
        
        # Negative log likelihood
        nll_loss = -dist.log_prob(y_clipped).mean()
        
        # Optional: Add calibration regularization to prevent extremely confident predictions
        # This helps ensure the uncertainty estimates are meaningful
        reg_loss = 0.01 * torch.abs(torch.log(phi)).mean()
        
        return nll_loss + reg_loss
    
    def get_estimate_and_ci(self, mu, phi, ci_level=0.95):
        """
        Get point estimate and confidence interval
        """
        alpha = mu * phi
        beta = (1 - mu) * phi
        
        dist = Beta(alpha, beta)
        lower = dist.icdf(torch.tensor((1 - ci_level) / 2))
        upper = dist.icdf(torch.tensor(1 - (1 - ci_level) / 2))
        
        estimate = mu
        ci = torch.cat([lower, upper], dim=1)
        
        # Calculate uncertainty (width of CI)
        uncertainty = upper - lower
        
        return estimate, ci, uncertainty


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
                _, _, attention_weights = self.model(marker_values, coverage)
                all_attentions.append(attention_weights)
        
        # Average attention weights across batches
        avg_attention = torch.cat(all_attentions, dim=0).mean(dim=0)  # [M]
        
        # Get top-k markers by attention weight
        top_k_indices = torch.topk(avg_attention, k=min(top_k, len(avg_attention))).indices
        top_k_weights = avg_attention[top_k_indices]
        
        return top_k_indices.cpu().numpy(), top_k_weights.cpu().numpy()
    
    def analyze_marker_coverage_impact(self, dataloader):
        """
        Analyze how marker coverage impacts prediction uncertainty
        """
        self.model.eval()
        coverage_values = []
        uncertainty_values = []
        
        with torch.no_grad():
            for marker_values, coverage, _ in dataloader:
                mu, phi, _ = self.model(marker_values, coverage)
                _, _, uncertainty = self.model.get_estimate_and_ci(mu, phi)
                
                # Calculate average coverage per sample
                avg_coverage = torch.mean(coverage, dim=1, keepdim=True)
                
                coverage_values.append(avg_coverage.cpu().numpy())
                uncertainty_values.append(uncertainty.cpu().numpy())
        
        # Concatenate results
        coverage_values = np.concatenate(coverage_values)
        uncertainty_values = np.concatenate(uncertainty_values)
        
        return coverage_values, uncertainty_values


# Extension: Ensemble model for improved robustness
class EnsembleCancerDetectionModel(nn.Module):
    """
    Ensemble of multiple cancer detection models for improved robustness
    """
    def __init__(self, num_markers, num_models=5, **model_kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            EnhancedCancerDetectionModel(num_markers, **model_kwargs)
            for _ in range(num_models)
        ])
    
    def forward(self, marker_values, coverage, y_true=None):
        all_mu = []
        all_phi = []
        all_losses = []
        all_attentions = []
        
        for model in self.models:
            if y_true is not None:
                mu, phi, loss, attention_weights = model(marker_values, coverage, y_true)
                all_losses.append(loss)
            else:
                mu, phi, attention_weights = model(marker_values, coverage)
                
            all_mu.append(mu)
            all_phi.append(phi)
            all_attentions.append(attention_weights)
        
        # Average predictions
        mu = torch.stack(all_mu).mean(dim=0)
        
        # For uncertainty, we want to account for both aleatoric (data) and epistemic (model) uncertainty
        # We use a combination of the average phi and the variance of mu across models
        phi_avg = torch.stack(all_phi).mean(dim=0)
        mu_var = torch.stack(all_mu).var(dim=0)
        
        # Adjusted phi to account for model disagreement
        phi = phi_avg * (1 + mu_var * 10)  # Scale factor to make epistemic uncertainty meaningful
        
        # Average attention weights
        attention_weights = torch.stack(all_attentions).mean(dim=0)
        
        if y_true is not None:
            # Use average loss
            loss = torch.stack(all_losses).mean()
            return mu, phi, loss, attention_weights
            
        return mu, phi, attention_weights
    
    def get_estimate_and_ci(self, mu, phi, ci_level=0.95):
        """
        Get point estimate and confidence interval using scipy instead of torch icdf
        """
        import scipy.stats as stats
        
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