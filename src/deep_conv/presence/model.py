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
    def __init__(self, num_markers, target_markers_mask=None, feature_dim=64, dropout_rate=0.3):
        super().__init__()
        self.num_markers = num_markers
        self.feature_dim = feature_dim
        
        # Register target markers mask buffer if provided
        if target_markers_mask is not None:
            self.register_buffer("target_markers_mask", target_markers_mask)
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(1)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
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
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Set a negative bias in the final layer to counter class imbalance
        if hasattr(self.classifier[-1], 'bias'):
            self.classifier[-1].bias.data.fill_(-0.5)
    
    def forward(self, marker_values, coverage, target_markers_mask=None):
        B, M = marker_values.shape
        
        # Get target markers mask
        if target_markers_mask is None:
            if hasattr(self, 'target_markers_mask'):
                target_markers_mask = self.target_markers_mask
            else:
                target_markers_mask = torch.ones(M, dtype=torch.bool, device=marker_values.device)
        
        # Create masks
        valid_mask = (coverage > 0)
        target_valid_mask = valid_mask & target_markers_mask.expand(B, -1)
        
        # Replace NaNs with zeros
        marker_values_safe = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))
        
        # Normalize by coverage
        coverage_safe = coverage.clone() + 1e-10
        normalized_markers = marker_values_safe / torch.sqrt(coverage_safe)
        
        # Reshape for batch operations
        marker_values_flat = normalized_markers.reshape(-1, 1)
        
        # Apply input normalization
        marker_values_norm = self.input_norm(marker_values_flat)
        
        # Extract features
        features = self.feature_extractor(marker_values_norm)
        
        # Apply transformation with residual connection
        transformed = self.feature_transform(features)
        features = features + transformed
        
        # Calculate attention weights
        attention_flat = self.attention(features).reshape(B, M)
        
        # Apply target and valid mask
        masked_attention = attention_flat * target_valid_mask.float()
        
        # Normalize attention
        attention_sum = masked_attention.sum(dim=1, keepdim=True)
        attention_sum = torch.where(attention_sum > 0, attention_sum, torch.ones_like(attention_sum))
        normalized_attention = masked_attention / attention_sum
        
        # Reshape features and apply attention
        features_reshaped = features.reshape(B, M, -1)
        expanded_attention = normalized_attention.unsqueeze(-1).expand_as(features_reshaped)
        weighted_features = features_reshaped * expanded_attention
        
        # Aggregate features
        aggregated_features = weighted_features.sum(dim=1)
        
        # Classification
        logits = self.classifier(aggregated_features)
        
        return logits, normalized_attention