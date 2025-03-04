import torch 
import torch.nn.functional as F


def presence_loss_fn(
    presence_logits: torch.Tensor,
    true_props: torch.Tensor,
    valid_mask: torch.Tensor,
    presence_threshold: float = 0.0005,  # 0.05% threshold 
    device: torch.device = None
):
    """
    Loss function for cell type presence detection with stronger penalty for false positives.
    
    Args:
        presence_logits: [B, C] Raw logits for presence prediction
        true_props: [B, C] Ground truth cell type proportions
        valid_mask: [B, M] Mask of valid markers
        presence_threshold: Threshold for considering a cell type present
        device: Device for computation
        
    Returns:
        loss: Scalar loss value
        details: Dictionary of metrics
    """
    if device is None:
        device = presence_logits.device
    
    # Convert proportions to binary presence/absence
    presence_targets = (true_props > presence_threshold).float()
    
    # Count positives and negatives
    num_positives = torch.sum(presence_targets)
    num_negatives = presence_targets.numel() - num_positives
    
    # Calculate pos_weight to balance the classes
    # Lower values give more weight to the negative class (reduces false positives)
    if num_positives > 0:
        neg_to_pos_ratio = num_negatives / num_positives
        pos_weight = 1.0 / (neg_to_pos_ratio + 1e-6)  # Inverse ratio
        pos_weight = torch.clamp(pos_weight, min=0.1, max=1.0)  # Limit range
    else:
        pos_weight = torch.tensor(0.1)  # Default if no positives
    
    pos_weight = pos_weight.to(device)
    
    # Binary cross-entropy with logits and class weighting
    loss = F.binary_cross_entropy_with_logits(
        presence_logits, presence_targets, pos_weight=pos_weight, reduction='mean'
    )
    
    # Calculate metrics for monitoring
    with torch.no_grad():
        # Convert logits to probabilities and binary predictions
        presence_probs = torch.sigmoid(presence_logits)
        presence_preds = (presence_probs > 0.5).float()
        
        # Calculate accuracy, precision, recall
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
    
    # Compile metrics
    details = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'precision': avg_precision.item(),
        'recall': avg_recall.item(),
        'specificity': avg_specificity.item(),
        'f1': avg_f1.item(),
        'metrics_per_class': {
            'precision': precision.cpu().numpy(),
            'recall': recall.cpu().numpy(),
            'specificity': specificity.cpu().numpy(),
            'f1': f1.cpu().numpy()
        },
        'valid_ratio': torch.mean(valid_mask.float()).item()
    }
    
    return loss, details