import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class BinaryCellTypeDataset(Dataset):
    """
    A PyTorch Dataset that treats cell type detection as a binary classification problem.
    
    Args:
        fraction: Methylation fraction values [num_samples, num_markers]
        coverage: Coverage values [num_samples, num_markers]
        y: Ground truth proportions [num_samples, num_cell_types]
        target_cell_type: Index of the target cell type
        target_ids: Marker to cell type mapping
        presence_threshold: Threshold for considering a cell type present
    """
    def __init__(self, fraction, coverage, y, target_cell_type, target_ids=None, 
                 presence_threshold=0.0005):
        # Convert inputs to tensors
        self.fraction = torch.tensor(fraction, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        
        # Extract the binary label for the target cell type
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32)
            self.target_prop = y_tensor[:, target_cell_type]
            self.label = (self.target_prop > presence_threshold).float()
        else:
            self.target_prop = None
            self.label = None
        
        self.target_cell_type = target_cell_type
        
        # Create marker mask for the target cell type
        if target_ids is not None:
            target_ids_t = torch.tensor(target_ids, dtype=torch.long)
            self.target_markers_mask = (target_ids_t == target_cell_type)
        else:
            self.target_markers_mask = None

    def __len__(self):
        return self.fraction.size(0)

    def __getitem__(self, idx):
        """
        Return a dictionary containing:
            'X': The methylation fraction row for this sample
            'coverage': The coverage row for this sample
            'label': Binary label indicating presence/absence of target cell type
            'concentration': Concentration of the target cell type (if available)
        """
        item = {
            'X': self.fraction[idx],
            'coverage': self.coverage[idx],
        }
        
        if self.label is not None:
            item['label'] = self.label[idx]
            
        if self.target_prop is not None:
            item['concentration'] = self.target_prop[idx]
            
        if self.target_markers_mask is not None:
            item['target_markers_mask'] = self.target_markers_mask
            
        return item


class Lambda(nn.Module):
    """Custom Lambda layer for arbitrary transformations"""
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)


class SingleCellTypePresenceModel(nn.Module):
    """
    Enhanced cell type detector with specialized pathways for low-concentration detection.
    
    Key features:
    1. Multi-pathway architecture with specialized processing for different concentration ranges
    2. Concentration estimation for adaptive processing and thresholding
    3. Cross-attention between pathways to leverage information from different concentration ranges
    4. Enhanced normalization and transformations for low-concentration signals
    """
    def __init__(self, num_markers, target_markers_mask=None, feature_dim=64, dropout_rate=0.3):
        super().__init__()
        self.num_markers = num_markers
        self.feature_dim = feature_dim
        
        # Register target markers mask buffer if provided
        if target_markers_mask is not None:
            self.register_buffer("target_markers_mask", target_markers_mask)
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(1)
        
        # Standard pathway for higher concentrations
        self.standard_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Low-concentration enhancement pathway
        self.low_conc_enhancer = nn.Sequential(
            # Log-transform input to amplify small differences
            Lambda(lambda x: torch.log1p(x * 100)),
            nn.Linear(1, feature_dim),
            nn.LeakyReLU(0.1),  # Better for small signals than ReLU
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout_rate * 0.7)  # Less dropout to preserve signal
        )
        
        # Medium-concentration pathway
        self.medium_conc_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),  # Smoother activation function
            nn.Dropout(dropout_rate)
        )
        
        # Feature transformation layers (with residual connections)
        self.standard_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.low_conc_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.7)
        )
        
        # Initial concentration estimation
        self.initial_concentration_estimator = nn.Sequential(
            nn.Linear(feature_dim * 3, 32),  # Takes concatenated features from all pathways
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
        # Cross-attention mechanism
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)
        
        # Attention mechanisms for each pathway
        self.standard_attention = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
        self.low_conc_attention = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid() 
        )
        
        # Final feature integration
        self.feature_integration = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),  # Layer norm instead of batch norm for final layers
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1)
        )
        
        # Final concentration estimator (more refined)
        self.concentration_estimator = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate strategies for each component"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize the final classification layer with slightly negative bias
        # for better precision at the expense of some recall
        if hasattr(self.classifier[-1], 'bias'):
            self.classifier[-1].bias.data.fill_(-0.2)
    
    def cross_attention(self, query_features, key_features, value_features):
        """
        Compute cross-attention between different feature sets.
        
        Args:
            query_features: Features to use as queries [B, M, feature_dim]
            key_features: Features to use as keys [B, M, feature_dim]
            value_features: Features to use as values [B, M, feature_dim]
            
        Returns:
            Attention-weighted features [B, M, feature_dim]
        """
        # Project features to query, key, value spaces
        queries = self.query_projection(query_features)
        keys = self.key_projection(key_features)
        values = self.value_projection(value_features)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        
        # Apply attention mask (optional, for valid markers only)
        # attention_scores = attention_scores.masked_fill(~attention_mask, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        attended_features = torch.matmul(attention_weights, values)
        
        return attended_features
    
    def forward(self, marker_values, coverage, target_markers_mask=None):
        """
        Forward pass of the enhanced cell type detector.
        
        Args:
            marker_values: [B, M] Methylation values
            coverage: [B, M] Coverage values
            target_markers_mask: [M] Mask indicating markers for target cell type
                                 (if not provided, uses the one stored in the model)
                               
        Returns:
            logits: [B, 1] Logits for binary classification
            concentration: [B, 1] Estimated concentration
            attention_weights: Dictionary of attention weights for visualization/analysis
        """
        B, M = marker_values.shape
        
        # Get target markers mask (either from input or model)
        if target_markers_mask is None:
            if hasattr(self, 'target_markers_mask'):
                target_markers_mask = self.target_markers_mask
            else:
                # If no mask provided, use all markers
                target_markers_mask = torch.ones(M, dtype=torch.bool, device=marker_values.device)
        
        # Create valid markers mask (coverage > 0 AND is target marker)
        valid_mask = (coverage > 0)
        target_valid_mask = valid_mask & target_markers_mask.expand(B, -1)
        
        # Replace NaNs with zeros (these will be masked out later)
        marker_values_safe = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))
        
        # Different normalizations for different pathways
        coverage_safe = coverage.clone() + 1e-10  # Add epsilon to avoid division by zero
        
        # Standard normalization
        standard_norm = marker_values_safe / torch.sqrt(coverage_safe)
        
        # Low-concentration-enhanced normalization
        # Logarithmic transformation to amplify small signals
        low_conc_norm = torch.log1p(marker_values_safe * 20) / torch.log1p(torch.sqrt(coverage_safe))
        low_conc_norm = torch.where(valid_mask, low_conc_norm, torch.zeros_like(low_conc_norm))
        
        # Medium-concentration normalization (between standard and low)
        medium_conc_norm = marker_values_safe / torch.pow(coverage_safe, 0.3)
        medium_conc_norm = torch.where(valid_mask, medium_conc_norm, torch.zeros_like(medium_conc_norm))
        
        # Process each normalization through its respective pathway
        # Reshape for batch normalization
        standard_flat = standard_norm.reshape(-1, 1)
        low_conc_flat = low_conc_norm.reshape(-1, 1)
        medium_flat = medium_conc_norm.reshape(-1, 1)
        
        # Apply input normalization
        standard_flat_norm = self.input_norm(standard_flat)
        
        # Extract features through each pathway
        standard_features = self.standard_extractor(standard_flat_norm)
        low_conc_features = self.low_conc_enhancer(low_conc_flat)
        medium_features = self.medium_conc_extractor(medium_flat)
        
        # Apply transformations with residual connections
        standard_transformed = self.standard_transform(standard_features)
        standard_features = standard_features + standard_transformed
        
        low_conc_transformed = self.low_conc_transform(low_conc_features)
        low_conc_features = low_conc_features + low_conc_transformed
        
        # Reshape features back to batch form
        standard_reshaped = standard_features.reshape(B, M, -1)
        low_conc_reshaped = low_conc_features.reshape(B, M, -1)
        medium_reshaped = medium_features.reshape(B, M, -1)
        
        # Calculate attention weights for standard and low-conc pathways
        standard_attention = self.standard_attention(standard_features).reshape(B, M)
        low_conc_attention = self.low_conc_attention(low_conc_features).reshape(B, M)
        
        # Apply target and valid mask to attention weights
        masked_std_attention = standard_attention * target_valid_mask.float()
        masked_low_attention = low_conc_attention * target_valid_mask.float()
        
        # Normalize attention weights to sum to 1 for each sample
        std_attention_sum = masked_std_attention.sum(dim=1, keepdim=True)
        std_attention_sum = torch.where(std_attention_sum > 0, std_attention_sum, torch.ones_like(std_attention_sum))
        std_attention_norm = masked_std_attention / std_attention_sum
        
        low_attention_sum = masked_low_attention.sum(dim=1, keepdim=True)
        low_attention_sum = torch.where(low_attention_sum > 0, low_attention_sum, torch.ones_like(low_attention_sum))
        low_attention_norm = masked_low_attention / low_attention_sum
        
        # Compute sample-level features by applying attention
        std_expanded_attn = std_attention_norm.unsqueeze(-1).expand_as(standard_reshaped)
        low_expanded_attn = low_attention_norm.unsqueeze(-1).expand_as(low_conc_reshaped)
        
        std_weighted = standard_reshaped * std_expanded_attn
        low_weighted = low_conc_reshaped * low_expanded_attn
        
        std_aggregated = std_weighted.sum(dim=1)  # [B, feature_dim]
        low_aggregated = low_weighted.sum(dim=1)  # [B, feature_dim]
        medium_aggregated = medium_reshaped.mean(dim=1)  # Using mean for medium pathway
        
        # Make initial concentration estimate
        combined_initial = torch.cat([std_aggregated, low_aggregated, medium_aggregated], dim=1)
        initial_concentration = self.initial_concentration_estimator(combined_initial)
        
        # Apply cross-attention between pathways (optional)
        # This can be computationally expensive, so we use the aggregated features
        # Reshape for cross-attention
        std_features_2d = std_aggregated.unsqueeze(1)  # [B, 1, feature_dim]
        low_features_2d = low_aggregated.unsqueeze(1)  # [B, 1, feature_dim]
        medium_features_2d = medium_aggregated.unsqueeze(1)  # [B, 1, feature_dim]
        
        # Cross-attention from low to standard (low queries standard)
        low_enhanced = self.cross_attention(
            low_features_2d,  # Queries from low
            std_features_2d,  # Keys from standard
            std_features_2d   # Values from standard
        ).squeeze(1)  # Remove the singleton dimension
        
        # Integrate features from all pathways with concentration-adaptive weighting
        # Lower concentration gives more weight to low-concentration pathway
        low_conc_weight = torch.sigmoid(1.5 - 30.0 * initial_concentration)  # Weight for low-conc pathway
        
        # Concatenate all features
        all_features = torch.cat([
            std_aggregated,
            low_enhanced,  # Cross-attended low features
            medium_aggregated
        ], dim=1)
        
        # Apply final integration
        integrated_features = self.feature_integration(all_features)
        
        # Final concentration estimation (more refined)
        estimated_concentration = self.concentration_estimator(integrated_features)
        
        # Final classification
        logits = self.classifier(integrated_features)
        
        # Store attention weights for visualization/analysis
        attention_weights = {
            'standard': std_attention_norm,
            'low_concentration': low_attention_norm,
            'concentration_weight': low_conc_weight
        }
        
        return logits, estimated_concentration, attention_weights
    
    def predict_with_adaptive_threshold(self, marker_values, coverage, target_markers_mask=None):
        """
        Make predictions with concentration-adaptive thresholding.
        
        Args:
            marker_values: [B, M] Methylation values
            coverage: [B, M] Coverage values
            target_markers_mask: [M] Mask for target markers
            
        Returns:
            predictions: [B, 1] Binary predictions
            probabilities: [B, 1] Prediction probabilities
            concentration: [B, 1] Estimated concentration
        """
        logits, concentration, _ = self.forward(marker_values, coverage, target_markers_mask)
        probabilities = torch.sigmoid(logits)
        
        # Adaptive threshold based on concentration
        # Higher concentration -> higher threshold (more confident)
        # Lower concentration -> lower threshold (more sensitive)
        base_threshold = 0.5
        threshold_adjustment = 0.2 * (1.0 - torch.clamp(concentration * 50, 0, 1))
        adaptive_threshold = base_threshold - threshold_adjustment
        
        predictions = (probabilities >= adaptive_threshold).float()
        
        return predictions, probabilities, concentration