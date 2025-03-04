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
    Loss function for cell type presence detection.
    
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
    
    # Calculate class weights to handle imbalance
    num_samples = true_props.size(0)
    positives_per_class = torch.sum(presence_targets, dim=0)
    pos_ratio = positives_per_class / num_samples
    neg_ratio = 1.0 - pos_ratio
    
    # Identify cell types that are all positive or all negative in this batch
    all_positive = (positives_per_class == num_samples)
    all_negative = (positives_per_class == 0)
    
    # Create weighted BCE based on class balance, but only for cell types with both positive and negative examples
    pos_weight = torch.ones_like(pos_ratio).to(device)
    
    # For cell types with mixed presence, calculate pos_weight
    mixed_mask = ~(all_positive | all_negative)
    if mixed_mask.any():
        mixed_indices = torch.nonzero(mixed_mask, as_tuple=True)[0]
        pos_weight[mixed_indices] = (neg_ratio[mixed_indices] / (pos_ratio[mixed_indices] + 1e-6))
        pos_weight[mixed_indices] = torch.clamp(pos_weight[mixed_indices], min=0.5, max=10.0)
    
    # Binary cross-entropy with logits (calculate per-element)
    bce_loss_elements = F.binary_cross_entropy_with_logits(
        presence_logits, presence_targets, reduction='none'
    )
    
    # Apply weights to cell types with mixed presence
    weighted_bce_loss = bce_loss_elements.clone()
    for i in range(weighted_bce_loss.size(1)):
        if mixed_mask[i]:
            # Weight positives more if they're rare
            weighted_bce_loss[:, i] = bce_loss_elements[:, i] * (
                presence_targets[:, i] * pos_weight[i] + (1 - presence_targets[:, i])
            )
    
    # Calculate loss only for cell types that have both positives and negatives
    # or are always positive (important for recall)
    valid_loss_mask = mixed_mask | all_positive
    if valid_loss_mask.any():
        loss = weighted_bce_loss[:, valid_loss_mask].mean()
    else:
        # Fallback if all cell types are always negative
        loss = bce_loss_elements.mean()
    
    # Calculate metrics for monitoring (similar updates as the evaluate function)
    with torch.no_grad():
        # Convert logits to probabilities and binary predictions
        presence_probs = torch.sigmoid(presence_logits)
        presence_preds = (presence_probs > 0.5).float()
        
        # Calculate accuracy, precision, recall
        correct = (presence_preds == presence_targets).float()
        accuracy = torch.mean(correct)
        
        # Confusion matrix elements (overall)
        true_positives = torch.sum(presence_preds * presence_targets, dim=0)
        false_positives = torch.sum(presence_preds * (1 - presence_targets), dim=0)
        false_negatives = torch.sum((1 - presence_preds) * presence_targets, dim=0)
        true_negatives = torch.sum((1 - presence_preds) * (1 - presence_targets), dim=0)
        
        # Metrics per class (handle special cases)
        precision = torch.zeros_like(true_positives).float()
        recall = torch.zeros_like(true_positives).float()
        specificity = torch.zeros_like(true_positives).float()
        f1 = torch.zeros_like(true_positives).float()
        
        # Normal case - cell type has both present and absent examples
        normal_mask = ~(all_positive | all_negative)
        if normal_mask.any():
            normal_idx = torch.nonzero(normal_mask, as_tuple=True)[0]
            precision[normal_idx] = true_positives[normal_idx] / (true_positives[normal_idx] + false_positives[normal_idx] + 1e-8)
            recall[normal_idx] = true_positives[normal_idx] / (true_positives[normal_idx] + false_negatives[normal_idx] + 1e-8)
            specificity[normal_idx] = true_negatives[normal_idx] / (true_negatives[normal_idx] + false_positives[normal_idx] + 1e-8)
            f1[normal_idx] = 2 * precision[normal_idx] * recall[normal_idx] / (precision[normal_idx] + recall[normal_idx] + 1e-8)
        
        # Always present case - can only measure recall
        if all_positive.any():
            pos_idx = torch.nonzero(all_positive, as_tuple=True)[0]
            precision[pos_idx] = torch.ones_like(precision[pos_idx])  # No false positives possible
            recall[pos_idx] = true_positives[pos_idx] / (true_positives[pos_idx] + false_negatives[pos_idx] + 1e-8)
            # specificity is undefined (no true negatives)
            f1[pos_idx] = 2 * recall[pos_idx] / (1 + recall[pos_idx] + 1e-8)  # simplified with precision=1
        
        # Always absent case - can only measure specificity
        if all_negative.any():
            neg_idx = torch.nonzero(all_negative, as_tuple=True)[0]
            # precision is undefined (no true positives)
            # recall is undefined (no true positives)
            specificity[neg_idx] = true_negatives[neg_idx] / (true_negatives[neg_idx] + false_positives[neg_idx] + 1e-8)
            # f1 is zero (no true positives)
        
        # Calculate means only for applicable metrics
        valid_precision = precision[~torch.isnan(precision)]
        valid_recall = recall[~torch.isnan(recall)]
        valid_specificity = specificity[~torch.isnan(specificity)]
        valid_f1 = f1[~torch.isnan(f1)]
        
        avg_precision = torch.mean(valid_precision) if len(valid_precision) > 0 else torch.tensor(0.0)
        avg_recall = torch.mean(valid_recall) if len(valid_recall) > 0 else torch.tensor(0.0)
        avg_specificity = torch.mean(valid_specificity) if len(valid_specificity) > 0 else torch.tensor(0.0)
        avg_f1 = torch.mean(valid_f1) if len(valid_f1) > 0 else torch.tensor(0.0)
    
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