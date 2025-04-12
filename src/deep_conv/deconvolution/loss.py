import torch 
import torch.nn.functional as F

def focal_loss(pred, target, alpha_pos=0.25, alpha_neg=0.75, gamma=2.0, fp_weight=1.5, class_weights=None):
    """
    Focal loss with class weighting to focus on rare cell types.
    
    Args:
        pred (FloatTensor): Predicted probabilities [B, C]
        target (FloatTensor): Ground truth labels [B, C]
        alpha_pos (float): Weighting factor for positive class
        alpha_neg (float): Weighting factor for negative class
        gamma (float): Focusing parameter
        fp_weight (float): Additional weight for false positives
        class_weights (FloatTensor): Weights for each class [C]
    
    Returns:
        loss (FloatTensor): Scalar focal loss with FP penalty
    """
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    
    # Apply alpha weighting
    alpha = torch.where(target > 0, alpha_pos, alpha_neg)
    focal_term = alpha * (1 - pt) ** gamma * bce
    
    # Apply class weights (if provided)
    if class_weights is not None:
        focal_term = focal_term * class_weights
    
    # Compute FP penalty
    pred_binary = (pred > 0.5).float()
    fp_mask = (pred_binary > target).float()
    fp_penalty = fp_weight * fp_mask * bce
    
    total_loss = focal_term + fp_penalty
    return total_loss.mean()

def loss_fn(
    pred_props: torch.Tensor,
    true_props: torch.Tensor,
    reconstructed: torch.Tensor,
    marker_values: torch.Tensor,
    coverage: torch.Tensor,
    valid_mask: torch.Tensor,
    presence_probs: torch.Tensor,
    presence_logits: torch.Tensor,
    alpha: float = 0.98,
    beta: float = 0.02,
    gamma: float = 0.02,
    presence_threshold: float = 0.01,
    low_snr_indices=[11],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    focal_loss_weight: float = 0.0,
    compute_diagnostics: bool = False
):
    # Sanitize coverage
    if torch.isnan(coverage).any() or torch.isinf(coverage).any():
        print("Warning: coverage contains nan or inf values")
    if (coverage < 0).any():
        print("Warning: coverage contains negative values")
    coverage = torch.nan_to_num(coverage, nan=0.0, posinf=0.0, neginf=0.0)
    coverage = torch.clamp(coverage, min=0.0)

    # Debug: Check other inputs
    if torch.isnan(pred_props).any() or torch.isinf(pred_props).any():
        print("Warning: pred_props contains nan or inf values")
    if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
        print("Warning: reconstructed contains nan or inf values")
    if torch.isnan(presence_probs).any() or torch.isinf(presence_probs).any():
        print("Warning: presence_probs contains nan or inf values")

    # Sanitize marker_values: Replace nan with 0 (should be masked by valid_mask)
    marker_values = torch.nan_to_num(marker_values, nan=0.0)

    # Ensure valid_mask is False for nan values in coverage
    valid_mask = (coverage > 0) & (~torch.isnan(coverage))

    # Proportion Error
    cell_errors = torch.abs(pred_props - true_props)
    
    importance_weights = torch.ones_like(true_props)
    low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
    med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
    high_conc_mask = true_props > 0.05
    importance_weights = torch.where(low_conc_mask, 2.0, importance_weights)
    importance_weights = torch.where(med_conc_mask, 1.6, importance_weights)
    importance_weights = torch.where(high_conc_mask, 1.2, importance_weights)
    
    capped_fraction = torch.clamp(true_props[:, low_snr_indices], max=0.10)
    importance_weights[:, low_snr_indices] *= (1.0 + 10.0 * capped_fraction)
    
    underestimation = F.relu(true_props - pred_props)
    overestimation = F.relu(pred_props - true_props)
    
    low_snr_mask = torch.zeros_like(true_props)
    low_snr_mask[:, low_snr_indices] = 1.0
    
    underestimation_penalty = 1.3 * underestimation
    low_snr_under_penalty = low_snr_mask * underestimation * 1.2
    
    weighted_errors = importance_weights * (cell_errors + underestimation_penalty + low_snr_under_penalty)
    loss_props = weighted_errors.mean()
    
    # Reconstruction Loss
    errors = torch.abs(marker_values - reconstructed)
    weighted_errors = valid_mask * coverage * errors
    denominator = torch.sum(valid_mask * coverage) + 1e-8
    if denominator < 1e-7:  # Avoid division by near-zero
        recon_loss = torch.tensor(0.0, device=device)
    else:
        recon_loss = torch.sum(weighted_errors) / denominator
    
    # Sparsity Regularisation
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1))
    
    # Focal Loss for Presence Detection (optional)
    presence_loss = 0.0
    if focal_loss_weight > 0:
        presence_probs = torch.clamp(presence_probs, 0.0, 1.0)  # Ensure probabilities are in [0, 1]
        presence_targets = (true_props > presence_threshold).float()
        presence_loss = focal_loss(
            presence_probs,
            presence_targets,
            alpha_pos=0.25,
            alpha_neg=0.75,
            gamma=2.0,
            fp_weight=1.5,
            class_weights=None
        )
    
    # Combine All Terms
    total_loss = alpha * loss_props + beta * recon_loss + gamma * sparsity_penalty + focal_loss_weight * presence_loss
    
    # Debug: Check for nan in loss components
    if torch.isnan(total_loss):
        print(f"Loss components: loss_props={loss_props.item()}, recon_loss={recon_loss.item()}, "
              f"sparsity_penalty={sparsity_penalty.item()}, presence_loss={presence_loss.item()}")

    # Diagnostics (optional)
    details = {}
    if compute_diagnostics:
        with torch.no_grad():
            presence_targets = (true_props > presence_threshold).float()
            presence_preds = (presence_probs > 0.5).float()
            
            true_positives = torch.sum(presence_preds * presence_targets, dim=0)
            false_positives = torch.sum(presence_preds * (1 - presence_targets), dim=0)
            false_negatives = torch.sum((1 - presence_preds) * presence_targets, dim=0)
            true_negatives = torch.sum((1 - presence_preds) * (1 - presence_targets), dim=0)
            
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
            
            avg_precision = torch.mean(precision)
            avg_recall = torch.mean(recall)
            avg_f1 = torch.mean(f1)
            accuracy = torch.mean((presence_preds == presence_targets).float())
            
            low_conc_error = torch.mean(torch.masked_select(cell_errors, low_conc_mask))
            med_conc_error = torch.mean(torch.masked_select(cell_errors, med_conc_mask))
            high_conc_error = torch.mean(torch.masked_select(cell_errors, high_conc_mask))

        details = {
            'total_loss': total_loss.item(),
            'loss_props': loss_props.item(),
            'recon_loss': recon_loss.item(),
            'sparsity_loss': sparsity_penalty.item(),
            'presence_loss': presence_loss.item() if focal_loss_weight > 0 else 0.0,
            'low_snr_under': underestimation[:, low_snr_indices].mean().item(),
            'low_snr_over': overestimation[:, low_snr_indices].mean().item(),
            'alpha_stats': {
                'mean': torch.mean(pred_props).item(),
                'std': torch.std(pred_props).item(),
                'max': torch.max(pred_props).item(),
                'min': torch.min(pred_props).item()
            },
            'concentration_errors': {
                'low_conc': low_conc_error.item() if not torch.isnan(low_conc_error) else 0.0,
                'med_conc': med_conc_error.item() if not torch.isnan(med_conc_error) else 0.0,
                'high_conc': high_conc_error.item() if not torch.isnan(high_conc_error) else 0.0
            },
            'presence_stats': {
                'accuracy': accuracy.item(),
                'avg_precision': avg_precision.item(),
                'avg_recall': avg_recall.item(),
                'avg_f1': avg_f1.item(),
                'true_positives': torch.sum(true_positives).item(),
                'false_positives': torch.sum(false_positives).item(),
                'false_negatives': torch.sum(false_negatives).item(),
                'true_negatives': torch.sum(true_negatives).item()
            },
            'valid_ratio': torch.mean(valid_mask.float()).item()
        }

    return total_loss, details