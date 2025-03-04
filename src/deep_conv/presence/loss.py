import torch 
import torch.nn.functional as F


def presence_loss(
    presence_logits: torch.Tensor,
    true_props: torch.Tensor,
    valid_mask: torch.Tensor,
    presence_threshold: float = 0.0005,  # 0.05% threshold 
    device: torch.device = None,
    low_conc_boost: float = 5.0,  # Higher weight for low concentrations
    cd4_cd8_indices: list = [3, 4]  # Indices of CD4 and CD8 cell types
):
    """
    Loss function focused specifically on low concentration detection.
    
    Args:
        presence_logits: [B, C] Raw logits for presence prediction
        true_props: [B, C] Ground truth cell type proportions
        valid_mask: [B, M] Mask of valid markers
        presence_threshold: Threshold for considering a cell type present
        device: Device for computation
        low_conc_boost: Weight multiplier for low concentration samples
        cd4_cd8_indices: Indices of the CD4/CD8 cell types for special handling
        
    Returns:
        loss: Scalar loss value
        details: Dictionary of metrics
    """
    if device is None:
        device = presence_logits.device
    
    # Convert proportions to binary presence/absence
    presence_targets = (true_props > presence_threshold).float()
    
    # Identify low concentration samples (those just above threshold)
    low_conc_mask = (true_props > presence_threshold) & (true_props <= 0.01)
    
    # Special mask for CD4/CD8 at low concentrations
    cd4_cd8_low_mask = torch.zeros_like(true_props, dtype=torch.bool)
    for idx in cd4_cd8_indices:
        cd4_cd8_low_mask[:, idx] = low_conc_mask[:, idx]
    
    # Calculate per-element BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(
        presence_logits, presence_targets, reduction='none'
    )
    
    # Create sample weights for different scenarios
    sample_weights = torch.ones_like(bce_loss)
    
    # Boost weight for all low concentration samples
    sample_weights[low_conc_mask] *= low_conc_boost
    
    # Extra boost for CD4/CD8 at low concentrations
    sample_weights[cd4_cd8_low_mask] *= 1.5
    
    # For negatives (zero concentration), use higher weight to reduce false positives
    neg_mask = (true_props <= presence_threshold)
    sample_weights[neg_mask] *= 2.0
    
    # Calculate weighted loss
    weighted_loss = (bce_loss * sample_weights).mean()
    
    # Calculate metrics for monitoring
    with torch.no_grad():
        # Convert logits to probabilities and binary predictions
        presence_probs = torch.sigmoid(presence_logits)
        presence_preds = (presence_probs > 0.5).float()
        
        # Overall accuracy
        correct = (presence_preds == presence_targets).float()
        accuracy = torch.mean(correct)
        
        # Confusion matrix elements
        true_positives = torch.sum(presence_preds * presence_targets, dim=0)
        false_positives = torch.sum(presence_preds * (1 - presence_targets), dim=0)
        false_negatives = torch.sum((1 - presence_preds) * presence_targets, dim=0)
        true_negatives = torch.sum((1 - presence_preds) * (1 - presence_targets), dim=0)
        
        # Metrics per class
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        specificity = true_negatives / (true_negatives + false_positives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Average metrics
        avg_precision = torch.mean(precision)
        avg_recall = torch.mean(recall)
        avg_specificity = torch.mean(specificity)
        avg_f1 = torch.mean(f1)
        
        # CD4/CD8 specific metrics
        cd4_cd8_precision = torch.mean(precision[cd4_cd8_indices])
        cd4_cd8_recall = torch.mean(recall[cd4_cd8_indices])
        cd4_cd8_f1 = torch.mean(f1[cd4_cd8_indices])
        
        # Low concentration performance
        low_conc_preds = presence_preds[low_conc_mask]
        low_conc_targets = presence_targets[low_conc_mask]
        if low_conc_targets.numel() > 0:
            low_conc_accuracy = torch.mean((low_conc_preds == low_conc_targets).float())
            low_conc_recall = torch.sum(low_conc_preds * low_conc_targets) / (torch.sum(low_conc_targets) + 1e-8)
        else:
            low_conc_accuracy = torch.tensor(0.0)
            low_conc_recall = torch.tensor(0.0)
    
    # Compile metrics
    details = {
        'loss': weighted_loss.item(),
        'accuracy': accuracy.item(),
        'precision': avg_precision.item(),
        'recall': avg_recall.item(),
        'specificity': avg_specificity.item(),
        'f1': avg_f1.item(),
        'cd4_cd8': {
            'precision': cd4_cd8_precision.item(),
            'recall': cd4_cd8_recall.item(),
            'f1': cd4_cd8_f1.item()
        },
        'low_conc': {
            'accuracy': low_conc_accuracy.item(),
            'recall': low_conc_recall.item()
        },
        'metrics_per_class': {
            'precision': precision.cpu().numpy(),
            'recall': recall.cpu().numpy(),
            'specificity': specificity.cpu().numpy(),
            'f1': f1.cpu().numpy()
        },
        'valid_ratio': torch.mean(valid_mask.float()).item()
    }
    
    return weighted_loss, details