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
    alpha: float = 0.9999,
    beta: float = 0.0001,
    gamma: float = 0.0001,
    presence_threshold: float = 0.005,
    low_snr_indices=[3, 4, 9, 11],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    Loss function specifically designed for the deconvolution model with pre-trained presence models.
    Unlike the original loss function, this one does not calculate BCE loss for presence detection
    since the presence models are pre-trained and frozen.

    This function has three main components:
      (1) Proportion error (with concentration-dependent weighting),
      (2) Coverage-weighted marker reconstruction error,
      (3) Sparsity regularisation.

    Args:
        pred_props (FloatTensor): [B, C]
            Model's predicted proportions per sample (B) for each cell type (C), summing to ~1.
        true_props (FloatTensor): [B, C]
            Ground-truth cell-type proportions.
        reconstructed (FloatTensor): [B, M]
            Model's reconstructed marker methylation values (decoder output).
        marker_values (FloatTensor): [B, M]
            True marker methylation values. Can contain invalid entries where coverage=0.
        coverage (FloatTensor): [B, M]
            Coverage (read depth) at each marker, used for weighting the reconstruction error.
        valid_mask (BoolTensor): [B, M]
            Indicates which (sample, marker) positions have coverage>0 (valid).
        presence_probs (FloatTensor): [B, C]
            Sigmoid probabilities from the pre-trained presence models.
        presence_logits (FloatTensor): [B, C]
            Logits (before sigmoid) from the pre-trained presence models.
        alpha (float):
            Weight for the main proportion error term (often near 1.0).
        beta (float):
            Weight for the reconstruction term (marker-level error).
        gamma (float):
            Weight for the sparsity penalty (discouraging spread-out predictions).
        presence_threshold (float):
            Threshold on true_props to decide if a cell type is "present" vs "absent" 
            in the ground truth. Used only for monitoring.
        low_snr_indices (list[int]):
            Indices of cell types considered "low SNR" or more uncertain, which receive
            additional penalty if under-predicted.
        device (torch.device):
            Device for computing pos_weight in BCE (usually the same as model device).

    Returns:
        total_loss (Tensor):
            A scalar tensor representing the combined loss.
        details (dict):
            A dictionary of intermediate scalars/statistics for monitoring.
    """
    # -----------------------------
    # (1) Proportion Error with Enhanced Concentration-Dependent Weighting
    # -----------------------------
    # Base absolute error for each cell type
    cell_errors = torch.abs(pred_props - true_props)

    # Create a base importance weight of 1.0 for all cell types
    importance_weights = torch.ones_like(true_props)

    # Concentration-dependent masks
    #  - low:  0.1% to 1%
    #  - med:  1%   to 5%
    #  - high: >5%
    low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
    med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
    high_conc_mask = true_props > 0.05

    # Scale importance by concentration range
    importance_weights = torch.where(low_conc_mask, 5.0, importance_weights)
    importance_weights = torch.where(med_conc_mask, 3.0, importance_weights)
    importance_weights = torch.where(high_conc_mask, 1.0, importance_weights)

    # Additional scaling for low-SNR cell types
    #  - For each low-SNR cell type, scale by (1 + 100 * min(true_prop, 0.10))
    for idx in low_snr_indices:
        capped_fraction = torch.clamp(true_props[:, idx], max=0.10)
        importance_weights[:, idx] *= (1.0 + 100.0 * capped_fraction)

    # Distinguish under- vs over-estimation
    underestimation = F.relu(true_props - pred_props)  # only positive if true>pred
    overestimation = F.relu(pred_props - true_props)   # only positive if pred>true

    # Create a mask indicating the low-SNR cell types
    low_snr_mask = torch.zeros_like(true_props)
    low_snr_mask[:, low_snr_indices] = 1.0

    # Stronger penalties for underestimation of any cell type
    #  - Factor of 2
    underestimation_penalty = 2.0 * underestimation

    # Additional factor for underestimation in low-SNR cell types
    low_snr_under_penalty = low_snr_mask * underestimation * 1.5
    
    # Combine all these error components
    weighted_errors = importance_weights * (cell_errors + underestimation_penalty + low_snr_under_penalty)

    # Mean across all cells in the batch
    loss_props = weighted_errors.mean()

    # -----------------------------
    # (2) Coverage-Weighted Reconstruction Loss
    # -----------------------------
    # If coverage=0 at a position => valid_mask=0 => that marker is not included in the error
    # Use safe_marker_values to avoid NaN issues: fallback to 'reconstructed' where coverage=0
    safe_marker_values = torch.where(valid_mask > 0, marker_values, reconstructed)
    recon_loss = torch.sum(
        valid_mask * coverage * torch.abs(safe_marker_values - reconstructed)
    ) / torch.sum(valid_mask * coverage)

    # -----------------------------
    # (3) Sparsity Regularisation
    # -----------------------------
    # Encourages the sum of proportions not to blow up (though they are typically normalised to 1).
    # This can help avoid the model distributing small amounts across too many cell types.
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1))

    # -----------------------------
    # Combine All Terms
    # -----------------------------
    # Weighted sum of the three main components (no presence loss)
    total_loss = alpha * loss_props + beta * recon_loss + gamma * sparsity_penalty

    # -----------------------------
    # (4) Detailed Monitoring / Diagnostics
    # -----------------------------
    with torch.no_grad():
        # Use pre-computed presence probabilities for monitoring only
        # Measure performance at threshold 0.5
        presence_preds = (presence_probs > 0.5).float()
        presence_targets = (true_props > presence_threshold).float()

        # Confusion counts
        true_positives = torch.sum(presence_preds * presence_targets, dim=0)
        false_positives = torch.sum(presence_preds * (1 - presence_targets), dim=0)
        false_negatives = torch.sum((1 - presence_preds) * presence_targets, dim=0)
        true_negatives = torch.sum((1 - presence_preds) * (1 - presence_targets), dim=0)

        # Precision / Recall / F1 per cell type
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

        avg_precision = torch.mean(precision)
        avg_recall = torch.mean(recall)
        avg_f1 = torch.mean(f1)
        accuracy = torch.mean((presence_preds == presence_targets).float())

        # Error in different concentration ranges
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
        'valid_ratio': torch.mean(valid_mask.float()).item()
    }

    return total_loss, details