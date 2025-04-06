import torch 
import torch.nn.functional as F

def loss_fn(
    pred_props: torch.Tensor,
    true_props: torch.Tensor,
    reconstructed: torch.Tensor,
    marker_values: torch.Tensor,
    coverage: torch.Tensor,
    valid_mask: torch.Tensor,
    presence_probs: torch.Tensor,
    presence_logits: torch.Tensor,
    alpha: float = 0.92,
    beta: float = 0.07,
    gamma: float = 0.01,
    presence_threshold: float = 0.005,
    low_snr_indices=[3, 4, 9, 11],
    reliable_coverage_threshold: float = 5.0,  # New parameter for reliable coverage
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    Loss function for deconvolution model with integrated presence models.
    Updated to weight errors by reliable coverage (>= 5).
    """
    # (1) Proportion Error
    cell_errors = torch.abs(pred_props - true_props)
    importance_weights = torch.ones_like(true_props)
    low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
    med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
    high_conc_mask = true_props > 0.05
    importance_weights = torch.where(low_conc_mask, 1.8, importance_weights)
    importance_weights = torch.where(med_conc_mask, 1.4, importance_weights)
    importance_weights = torch.where(high_conc_mask, 1.0, importance_weights)
    for idx in low_snr_indices:
        capped_fraction = torch.clamp(true_props[:, idx], max=0.10)
        importance_weights[:, idx] *= (1.0 + 10.0 * capped_fraction)
    underestimation = F.relu(true_props - pred_props)
    overestimation = F.relu(pred_props - true_props)
    low_snr_mask = torch.zeros_like(true_props)
    low_snr_mask[:, low_snr_indices] = 1.0
    underestimation_penalty = 1.3 * underestimation
    low_snr_under_penalty = low_snr_mask * underestimation * 0.7
    weighted_errors = importance_weights * (cell_errors + underestimation_penalty + low_snr_under_penalty)

    # Weight by proportion of markers with reliable coverage (>= 5)
    reliable_mask = (coverage >= reliable_coverage_threshold)
    reliable_ratio = reliable_mask.float().mean(dim=1)  # [B]
    sample_weights = reliable_ratio / (reliable_ratio.max() + 1e-8)  # Normalise to [0, 1]
    loss_props = torch.mean(sample_weights.unsqueeze(1) * weighted_errors)

    # (2) Coverage-Weighted Reconstruction Loss
    # Give higher weight to markers with coverage >= 5
    coverage_weights = torch.where(reliable_mask, coverage * 2.0, coverage * 0.5)  # Boost reliable markers
    safe_marker_values = torch.where(valid_mask, marker_values, reconstructed)
    recon_loss = torch.sum(
        valid_mask * coverage_weights * torch.abs(safe_marker_values - reconstructed)
    ) / (torch.sum(valid_mask * coverage_weights) + 1e-8)

    # (3) Sparsity Regularisation
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1))

    # Combine All Terms
    total_loss = alpha * loss_props + beta * recon_loss + gamma * sparsity_penalty

    # (4) Detailed Monitoring / Diagnostics
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
        'valid_ratio': torch.mean(valid_mask.float()).item(),
        'reliable_ratio': reliable_ratio.mean().item()  # New metric for monitoring
    }
    
    return total_loss, details
