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
    A binary classifier for detecting the presence of a specific cell type.
    
    Key features:
    1. Uses only the markers relevant to the target cell type
    2. Processes markers with varying coverage appropriately
    3. Uses attention mechanism to focus on the most informative markers
    4. Employs a deep architecture with residual connections for better feature extraction
    5. Incorporates coverage information directly into feature extraction
    6. Uses coverage-aware normalization for improved handling of low-coverage data
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
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
        
        # Set a negative bias in the final layer to counter class imbalance
        if hasattr(self.classifier[-1], 'bias'):
            self.classifier[-1].bias.data.fill_(0.0)
    
    def forward(self, marker_values, coverage):
        """
        Forward pass of the binary classifier.
        
        Args:
            marker_values: [B, M] Methylation values
            coverage: [B, M] Coverage values
                               
        Returns:
            logits: [B, 1] Logits for binary classification
            attention_weights: [B, M] Attention weights for each marker
        """
        B, M = marker_values.shape
        
        # Create valid markers mask (coverage > 0)
        valid_mask = (coverage > 0)
        
        # Replace NaNs with zeros (these will be masked out later)
        marker_values_safe = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))
        
        # Coverage-aware normalization
        # Reduce confidence for low coverage markers
        coverage_safe = coverage.clone() + 1e-10  # Add epsilon to avoid division by zero
        
        # Create a confidence factor that scales with coverage
        # For coverage=1, factor=0.2; for coverage=5, factor=0.5; for coverage=20, factor=0.8
        confidence_factor = torch.clamp(coverage_safe / (coverage_safe + 10.0), 0.3, 1.0)

        
        # Apply the confidence factor to marker values
        normalised_markers = marker_values_safe * confidence_factor
        
        # Process all markers through feature extraction
        marker_values_flat = normalised_markers.reshape(-1, 1)  # [B*M, 1]
        coverage_flat = torch.log1p(coverage_safe).reshape(-1, 1)  # Log to compress the range
        
        # Concatenate marker values and coverage
        features_input = torch.cat([marker_values_flat, coverage_flat], dim=1)  # [B*M, 2]
        
        # Apply batch normalization to inputs
        features_norm = self.input_norm(features_input)
        
        # Extract features
        features = self.feature_extractor(features_norm)  # [B*M, feature_dim]
        
        # Apply feature transformation with residual connection
        transformed_features = self.feature_transform(features)
        features = features + transformed_features  # [B*M, feature_dim]
        
        # Calculate attention weights
        attention_flat = self.attention(features).reshape(B, M)  # [B, M]
        
        # Apply valid mask to attention and weight by coverage confidence
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
        
        # Final classification
        logits = self.classifier(aggregated_features)
        
        return logits, normalized_attention
    
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
        logits, _ = self.forward(marker_values, coverage)
        probabilities = torch.sigmoid(logits).squeeze(-1)
        predictions = (probabilities >= threshold).float()
        return predictions, probabilities   
    