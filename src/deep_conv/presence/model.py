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


class SingleCellTypePresenceModel(nn.Module):
    """
    An enhanced binary classifier for cell type detection with improved handling of
    low coverage and missing markers.
    
    Key features:
    1. Multi-resolution analysis with pooling at different scales
    2. Enhanced coverage-aware feature extraction
    3. Explicit missing marker handling
    4. Attention mechanism to focus on the most informative markers
    5. Coverage-adaptive prediction threshold
    6. Improved confidence factors for very low coverage
    """
    def __init__(self, feature_dim=64, dropout_rate=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(2)
        
        # Feature extraction with marker values and coverage
        self.feature_extractor = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Multi-resolution feature fusion
        self.multi_res_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Feature transformation with residual connection
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Coverage-aware attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Classification head with missing marker information
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 1, 64),  # +1 for missing rate feature
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Set a neutral bias in the final layer (no class bias)
        if hasattr(self.classifier[-1], 'bias'):
            self.classifier[-1].bias.data.fill_(0.0)
    
    def _extract_features(self, markers, coverage):
        """
        Helper method to extract features from markers and coverage values.
        
        Args:
            markers: [B*M, 1] Methylation values (flattened)
            coverage: [B*M, 1] Coverage values (flattened)
            
        Returns:
            features: [B*M, feature_dim] Extracted features
        """
        # Concatenate marker values and coverage
        features_input = torch.cat([markers, coverage], dim=1)  # [B*M, 2]
        
        # Apply batch normalization to inputs
        features_norm = self.input_norm(features_input)
        
        # Extract features
        features = self.feature_extractor(features_norm)  # [B*M, feature_dim]
        return features
    
    def forward(self, marker_values, coverage):
        """
        Forward pass of the enhanced binary classifier.
        
        Args:
            marker_values: [B, M] Methylation values
            coverage: [B, M] Coverage values
                               
        Returns:
            logits: [B, 1] Logits for binary classification
            attention_weights: [B, M] Attention weights for each marker
            missing_rate: [B, 1] Percentage of missing markers per sample
        """
        B, M = marker_values.shape
        
        # Create valid markers mask (coverage > 0)
        valid_mask = (coverage > 0)
        
        # Calculate missing marker rate for each sample
        missing_rate = (1.0 - valid_mask.float().mean(dim=1, keepdim=True))
        
        # Replace NaNs with zeros (these will be masked out later)
        marker_values_safe = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))
        
        # Coverage-aware normalization with enhanced confidence factor
        # Reduce confidence for low coverage markers
        coverage_safe = coverage.clone() + 1e-10  # Add epsilon to avoid division by zero
        
        # Enhanced confidence factor - more conservative at very low coverage
        confidence_factor = torch.clamp(
            torch.where(
                coverage_safe < 5.0,
                coverage_safe / (coverage_safe + 15.0),  # More skeptical of very low coverage
                coverage_safe / (coverage_safe + 10.0)   # Original scaling for higher coverage
            ),
            0.2, 1.0  # Lower minimum confidence for very low coverage
        )
        
        # Apply the confidence factor to marker values
        normalised_markers = marker_values_safe * confidence_factor
        
        # ===== MULTI-RESOLUTION ANALYSIS =====
        # 1. Original resolution
        marker_values_flat = normalised_markers.reshape(-1, 1)  # [B*M, 1]
        coverage_flat = torch.log1p(coverage_safe).reshape(-1, 1)  # [B*M, 1]
        
        # 2. Medium resolution (pooling with kernel size 3)
        markers_med = F.avg_pool1d(
            normalised_markers.view(B, 1, M), 
            kernel_size=3, 
            stride=1, 
            padding=1
        ).view(B, M)
        
        coverage_med = F.avg_pool1d(
            coverage_safe.view(B, 1, M), 
            kernel_size=3, 
            stride=1, 
            padding=1
        ).view(B, M)
        
        markers_med_flat = markers_med.reshape(-1, 1)  # [B*M, 1]
        coverage_med_flat = torch.log1p(coverage_med).reshape(-1, 1)  # [B*M, 1]
        
        # 3. Low resolution (pooling with kernel size 7)
        markers_low = F.avg_pool1d(
            normalised_markers.view(B, 1, M), 
            kernel_size=7, 
            stride=1, 
            padding=3
        ).view(B, M)
        
        coverage_low = F.avg_pool1d(
            coverage_safe.view(B, 1, M), 
            kernel_size=7, 
            stride=1, 
            padding=3
        ).view(B, M)
        
        markers_low_flat = markers_low.reshape(-1, 1)  # [B*M, 1]
        coverage_low_flat = torch.log1p(coverage_low).reshape(-1, 1)  # [B*M, 1]
        
        # Extract features at each resolution
        features_orig = self._extract_features(marker_values_flat, coverage_flat)
        features_med = self._extract_features(markers_med_flat, coverage_med_flat)
        features_low = self._extract_features(markers_low_flat, coverage_low_flat)
        
        # Combine multi-resolution features
        multi_res_features = torch.cat([features_orig, features_med, features_low], dim=1)
        features = self.multi_res_fusion(multi_res_features)
        # ===== END MULTI-RESOLUTION ANALYSIS =====
        
        # Apply feature transformation with residual connection
        transformed_features = self.feature_transform(features)
        features = features + transformed_features  # [B*M, feature_dim]
        
        # Calculate attention weights
        attention_flat = self.attention(features).reshape(B, M)  # [B, M]
        
        # Apply valid mask to attention and weight by confidence factor
        masked_attention = attention_flat * valid_mask.float() * confidence_factor
        
        # Normalize attention weights to sum to 1 for each sample
        attention_sum = masked_attention.sum(dim=1, keepdim=True)
        attention_sum = torch.where(attention_sum > 0, attention_sum, torch.ones_like(attention_sum))
        normalized_attention = masked_attention / attention_sum
        
        # Reshape features back to batch form
        features_reshaped = features.reshape(B, M, self.feature_dim)  # [B, M, feature_dim]
        
        # Apply attention weights to features
        expanded_attention = normalized_attention.unsqueeze(-1).expand(-1, -1, self.feature_dim)
        weighted_features = features_reshaped * expanded_attention
        
        # Aggregate features across markers
        aggregated_features = weighted_features.sum(dim=1)  # [B, feature_dim]
        
        # Include missing marker rate as an additional feature
        enhanced_features = torch.cat([aggregated_features, missing_rate], dim=1)
        
        # Final classification
        logits = self.classifier(enhanced_features)
        
        return logits, normalized_attention, missing_rate
    
    def predict(self, marker_values, coverage, threshold=0.5):
        """
        Make binary predictions.
        
        Args:
            marker_values: [B, M] Methylation values
            coverage: [B, M] Coverage values
            threshold: Classification threshold
        
        Returns:
            predictions: [B] Binary predictions (0/1)
            probabilities: [B] Prediction probabilities
        """
        logits, _, _ = self.forward(marker_values, coverage)
        probabilities = torch.sigmoid(logits).squeeze(-1)
        predictions = (probabilities >= threshold).float()
        return predictions, probabilities
    
    def adaptive_predict(self, marker_values, coverage):
        """
        Make predictions with coverage-adaptive threshold.
        
        Args:
            marker_values: [B, M] Methylation values
            coverage: [B, M] Coverage values
        
        Returns:
            predictions: [B] Binary predictions (0/1)
            probabilities: [B] Prediction probabilities
            thresholds: [B] Coverage-adaptive thresholds used for each sample
        """
        logits, _, missing_rate = self.forward(marker_values, coverage)
        probabilities = torch.sigmoid(logits).squeeze(-1)
        
        # Calculate mean coverage for each sample
        mean_coverage = coverage.mean(dim=1)
        
        # Create base threshold
        base_threshold = 0.5
        
        # Adjust threshold based on coverage and missing rate
        # Higher threshold (more conservative) for lower coverage and more missing markers
        coverage_adjustment = torch.clamp(0.15 - 0.005 * mean_coverage, 0.0, 0.15)
        missing_adjustment = torch.clamp(0.15 * missing_rate.squeeze(), 0.0, 0.15)
        
        # Combined adjustment (max 0.25 total adjustment)
        total_adjustment = torch.clamp(coverage_adjustment + missing_adjustment, 0.0, 0.25)
        adaptive_threshold = base_threshold + total_adjustment
        
        # Make predictions
        predictions = (probabilities >= adaptive_threshold).float()
        
        return predictions, probabilities, adaptive_threshold