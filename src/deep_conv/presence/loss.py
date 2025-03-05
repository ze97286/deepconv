import torch 
import torch.nn.functional as F

def single_cell_presence_loss(
    presence_logit: torch.Tensor,
    true_props: torch.Tensor,
    valid_mask: torch.Tensor,
    target_cell_type: int,
    presence_threshold: float = 0.0005,
    fp_weight: float = 2.0,  # Weight for false positives
    device: torch.device = None
):
    """
    Simple loss function for single cell type presence detection.
    
    Args:
        presence_logit: [B, 1] Logit for the target cell type
        true_props: [B, C] Ground truth proportions for all cell types
        valid_mask: [B, M] Mask of valid markers
        target_cell_type: Index of the target cell type
        presence_threshold: Threshold for considering a cell type present
        fp_weight: Weight for false positive errors
        device: Device for computation
        
    Returns:
        loss: Scalar loss value
        details: Dictionary of metrics
    """
    if device is None:
        device = presence_logit.device
    
    # Extract ground truth for target cell type
    true_props_target = true_props[:, target_cell_type]
    
    # Convert to binary presence/absence
    presence_target = (true_props_target > presence_threshold).float().view(-1, 1)
    
    # Weight false positives more heavily
    weights = torch.ones_like(presence_target)
    weights[presence_target == 0] = fp_weight  # More weight to false positives
    
    # Weighted BCE
    bce_loss = F.binary_cross_entropy_with_logits(
        presence_logit, presence_target, weight=weights, reduction='mean'
    )
    
    # Metrics for monitoring
    with torch.no_grad():
        presence_prob = torch.sigmoid(presence_logit)
        presence_pred = (presence_prob > 0.5).float()
        
        # Accuracy
        accuracy = torch.mean((presence_pred == presence_target).float())
        
        # Confusion matrix
        tp = torch.sum((presence_pred == 1) & (presence_target == 1)).item()
        fp = torch.sum((presence_pred == 1) & (presence_target == 0)).item()
        tn = torch.sum((presence_pred == 0) & (presence_target == 0)).item()
        fn = torch.sum((presence_pred == 0) & (presence_target == 1)).item()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    details = {
        'loss': bce_loss.item(),
        'accuracy': accuracy.item(),
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion': {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
    }
    
    return bce_loss, details


# def presence_loss_fn(
    # presence_logits: torch.Tensor,
    # true_props: torch.Tensor,
    # valid_mask: torch.Tensor,
    # presence_threshold: float = 0.0005,  # 0.05% threshold 
    # device: torch.device = None,
    # cell_type_names=None
# ):
#     """
#     Loss function for cell type presence detection with:
#     1. Stronger penalty for false positives
#     2. Weighting by valid marker ratio to prioritize samples with more valid data
#     3. Detailed per-cell-type logging
    
#     Args:
#         presence_logits: [B, C] Raw logits for presence prediction
#         true_props: [B, C] Ground truth cell type proportions
#         valid_mask: [B, M] Mask of valid markers
#         presence_threshold: Threshold for considering a cell type present
#         device: Device for computation
#         cell_type_names: Optional list of cell type names for better reporting
        
#     Returns:
#         loss: Scalar loss value
#         details: Dictionary of metrics
#     """
#     if device is None:
#         device = presence_logits.device
    
#     # Convert proportions to binary presence/absence
#     presence_targets = (true_props > presence_threshold).float()
    
#     # Count positives and negatives
#     num_positives = torch.sum(presence_targets)
#     num_negatives = presence_targets.numel() - num_positives
    
#     # Calculate pos_weight to balance the classes
#     # Lower values give more weight to the negative class (reduces false positives)
#     if num_positives > 0:
#         neg_to_pos_ratio = num_negatives / num_positives
#         pos_weight = 1.0 / (neg_to_pos_ratio + 1e-6)  # Inverse ratio
#         pos_weight = torch.clamp(pos_weight, min=0.05, max=0.5)
#     else:
#         pos_weight = torch.tensor(0.1)  # Default if no positives
    
#     pos_weight = pos_weight.to(device)
    
#     # Calculate the proportion of valid markers per sample
#     valid_ratio_per_sample = torch.sum(valid_mask.float(), dim=1) / valid_mask.size(1)
    
#     # Get per-element BCE loss
#     bce_loss = F.binary_cross_entropy_with_logits(
#         presence_logits, presence_targets, pos_weight=pos_weight, reduction='none'
#     )
    
#     # Weight the loss by valid ratio to prioritize samples with more valid markers
#     sample_weights = valid_ratio_per_sample.unsqueeze(1).expand_as(bce_loss)
#     weighted_loss = (bce_loss * sample_weights).mean()
    
#     # Calculate metrics for monitoring
#     with torch.no_grad():
#         # Convert logits to probabilities and binary predictions
#         presence_probs = torch.sigmoid(presence_logits)
#         presence_preds = (presence_probs > 0.8).float()
        
#         # Calculate accuracy, precision, recall
#         correct = (presence_preds == presence_targets).float()
#         accuracy = torch.mean(correct)
        
#         # Confusion matrix elements
#         true_positives = torch.sum(presence_preds * presence_targets, dim=0)
#         false_positives = torch.sum(presence_preds * (1 - presence_targets), dim=0)
#         false_negatives = torch.sum((1 - presence_preds) * presence_targets, dim=0)
#         true_negatives = torch.sum((1 - presence_preds) * (1 - presence_targets), dim=0)
        
#         # Total counts across all cell types
#         total_tp = torch.sum(true_positives).item()
#         total_fp = torch.sum(false_positives).item()
#         total_tn = torch.sum(true_negatives).item()
#         total_fn = torch.sum(false_negatives).item()
        
#         # Per-cell-type metrics
#         precision = true_positives / (true_positives + false_positives + 1e-8)
#         recall = true_positives / (true_positives + false_negatives + 1e-8)
#         specificity = true_negatives / (true_negatives + false_positives + 1e-8)
#         f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
#         # Average metrics
#         avg_precision = torch.mean(precision)
#         avg_recall = torch.mean(recall)
#         avg_specificity = torch.mean(specificity)
#         avg_f1 = torch.mean(f1)
        
#         # Find worst-performing cell types
#         if precision.numel() > 0 and not torch.all(torch.isnan(precision)):
#             valid_precision = precision[~torch.isnan(precision)]
#             worst_precision_idx = torch.argmin(valid_precision).item() if valid_precision.numel() > 0 else -1
#         else:
#             worst_precision_idx = -1
            
#         if recall.numel() > 0 and not torch.all(torch.isnan(recall)):
#             valid_recall = recall[~torch.isnan(recall)]
#             worst_recall_idx = torch.argmin(valid_recall).item() if valid_recall.numel() > 0 else -1
#         else:
#             worst_recall_idx = -1
            
#         if specificity.numel() > 0 and not torch.all(torch.isnan(specificity)):
#             valid_specificity = specificity[~torch.isnan(specificity)]
#             worst_specificity_idx = torch.argmin(valid_specificity).item() if valid_specificity.numel() > 0 else -1
#         else:
#             worst_specificity_idx = -1
        
#         # Get proportions of cell types present in this batch
#         cell_type_present_percent = torch.mean(presence_targets, dim=0) * 100
    
#     # Compile metrics with confusion matrix counts and per-cell-type details
#     details = {
#         'loss': weighted_loss.item(),
#         'accuracy': accuracy.item(),
#         'precision': avg_precision.item(),
#         'recall': avg_recall.item(),
#         'specificity': avg_specificity.item(),
#         'f1': avg_f1.item(),
#         'valid_ratio_mean': torch.mean(valid_ratio_per_sample).item(),
#         'confusion_counts': {
#             'tp': total_tp,
#             'fp': total_fp,
#             'tn': total_tn,
#             'fn': total_fn
#         },
#         'metrics_per_class': {
#             'precision': precision.cpu().numpy(),
#             'recall': recall.cpu().numpy(),
#             'specificity': specificity.cpu().numpy(),
#             'f1': f1.cpu().numpy(),
#             'tp': true_positives.cpu().numpy(),
#             'fp': false_positives.cpu().numpy(),
#             'tn': true_negatives.cpu().numpy(),
#             'fn': false_negatives.cpu().numpy(),
#             'present_percent': cell_type_present_percent.cpu().numpy()
#         }
#     }
    
#     # Add worst-performing cell types if applicable
#     worst_performers = {}
    
#     if worst_precision_idx >= 0:
#         worst_performers['worst_precision'] = {
#             'index': int(worst_precision_idx),
#             'name': cell_type_names[worst_precision_idx] if cell_type_names else f"Cell type {worst_precision_idx}",
#             'value': precision[worst_precision_idx].item(),
#             'tp': true_positives[worst_precision_idx].item(),
#             'fp': false_positives[worst_precision_idx].item()
#         }
    
#     if worst_recall_idx >= 0:
#         worst_performers['worst_recall'] = {
#             'index': int(worst_recall_idx),
#             'name': cell_type_names[worst_recall_idx] if cell_type_names else f"Cell type {worst_recall_idx}",
#             'value': recall[worst_recall_idx].item(),
#             'tp': true_positives[worst_recall_idx].item(),
#             'fn': false_negatives[worst_recall_idx].item()
#         }
    
#     if worst_specificity_idx >= 0:
#         worst_performers['worst_specificity'] = {
#             'index': int(worst_specificity_idx),
#             'name': cell_type_names[worst_specificity_idx] if cell_type_names else f"Cell type {worst_specificity_idx}",
#             'value': specificity[worst_specificity_idx].item(),
#             'tn': true_negatives[worst_specificity_idx].item(),
#             'fp': false_positives[worst_specificity_idx].item()
#         }
    
#     if worst_performers:
#         details['worst_performers'] = worst_performers
    
#     return weighted_loss, details