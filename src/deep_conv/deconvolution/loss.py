import torch
import torch.nn.functional as F

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """Compute focal loss for binary classification to enhance presence detection.

    Args:
        pred (torch.Tensor): Predicted probabilities [B, C].
        target (torch.Tensor): Binary target labels [B, C].
        gamma (float, optional): Focusing parameter to reduce loss for easy examples. Defaults to 2.0.
        alpha (float, optional): Class balancing weight. Defaults to 0.25.

    Returns:
        torch.Tensor: Scalar focal loss value.
    """
    bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
    modulator = torch.pow(1 - pred, gamma) * target + torch.pow(pred, gamma) * (1 - target)
    return (alpha * modulator * bce).mean()

def loss_fn(
    pred_props: torch.Tensor,
    true_props: torch.Tensor,
    reconstructed: torch.Tensor,
    marker_values: torch.Tensor,
    coverage: torch.Tensor,
    valid_mask: torch.Tensor,
    presence_probs: torch.Tensor,
    presence_logits: torch.Tensor,
    x_nnls: torch.Tensor,
    dl_props: torch.Tensor,
    combination_weight: torch.Tensor,
    marker_quality_weights: torch.Tensor,
    marker_selection: torch.Tensor,
    alpha: float = 0.85,
    beta: float = 0.07,
    gamma: float = 0.05,  # Increased to enforce sparsity
    presence_threshold: float = 0.01,
    low_snr_indices=[3, 4, 9, 11],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """Compute the loss for cell-type deconvolution, optimising for low-SNR and low-coverage scenarios.

    This function combines multiple loss terms:
    - Proportion error with coverage-based and low-SNR weighting to improve accuracy.
    - Critical range loss targeting low concentrations (0.1–2%) for low-SNR cell types.
    - Coverage-weighted reconstruction loss to ensure accurate marker reconstruction.
    - Focal presence loss to enhance presence detection.
    - Sparsity penalty to encourage zero predictions for absent cell types.
    - L1 penalty on low-SNR cell types to further enforce sparsity.
    - NNLS regularisation and weight penalty to align with NNLS predictions, prioritising DeepConv for low-SNR types.
    - Marker weight regularisation to prevent collapse of marker quality and selection weights.

    Args:
        pred_props (torch.FloatTensor): Predicted proportions [B, C].
        true_props (torch.FloatTensor): Ground-truth proportions [B, C].
        reconstructed (torch.FloatTensor): Reconstructed marker values [B, M].
        marker_values (torch.FloatTensor): True marker values [B, M], NaN where coverage=0.
        coverage (torch.FloatTensor): Coverage values [B, M].
        valid_mask (torch.BoolTensor): Mask indicating valid markers [B, M].
        presence_probs (torch.FloatTensor): Presence probabilities [B, C].
        presence_logits (torch.FloatTensor): Raw logits [B, C].
        x_nnls (torch.FloatTensor): NNLS predictions [B, C], or None.
        dl_props (torch.FloatTensor): Deep learning-only proportions [B, C].
        combination_weight (torch.FloatTensor): Ensembling weights [C].
        marker_quality_weights (torch.Tensor): Marker quality weights [M].
        marker_selection (torch.Tensor): Marker selection weights [M].
        alpha (float, optional): Proportion error weight. Defaults to 0.85.
        beta (float, optional): Reconstruction loss weight. Defaults to 0.07.
        gamma (float, optional): Sparsity penalty weight. Defaults to 0.05.
        presence_threshold (float, optional): Presence threshold. Defaults to 0.01.
        low_snr_indices (list[int], optional): Low-SNR indices. Defaults to [3, 4, 9, 11].
        device (torch.device, optional): Device. Defaults to CUDA if available.

    Returns:
        tuple:
            - torch.Tensor: Combined loss scalar.
            - dict: Diagnostic statistics, including loss components and metrics.
    """
    # Proportion Error
    errors = torch.abs(pred_props - true_props)
    importance_weights = torch.ones_like(true_props)

    # Apply concentration-specific weighting
    low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
    med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
    high_conc_mask = true_props > 0.05
    importance_weights = torch.where(low_conc_mask, 1.8, importance_weights)
    importance_weights = torch.where(med_conc_mask, 1.4, importance_weights)
    importance_weights = torch.where(high_conc_mask, 1.0, importance_weights)

    # Apply additional weighting for low-SNR cell types
    for idx in low_snr_indices:
        capped_fraction = torch.clamp(true_props[:, idx], max=0.10)
        importance_weights[:, idx] *= (1.0 + 10.0 * capped_fraction)

    # Weight proportion errors by coverage to enhance low-coverage robustness
    coverage_weights = torch.sigmoid(coverage / 10.0).mean(dim=1, keepdim=True)
    weighted_errors = errors * importance_weights * coverage_weights

    # Add underestimation and overestimation penalties
    underestimation = F.relu(true_props - pred_props)
    overestimation = F.relu(pred_props - true_props)
    low_snr_mask = torch.zeros_like(true_props)
    low_snr_mask[:, low_snr_indices] = 1.0
    underestimation_penalty = 1.3 * underestimation
    low_snr_under_penalty = low_snr_mask * underestimation * 0.7
    loss_props = (weighted_errors + underestimation_penalty + low_snr_under_penalty).mean()

    # Apply critical range loss for low concentrations (0.1–2%)
    critical_ranges = [
        ((true_props >= 0.001) & (true_props < 0.005), 1.0),
        ((true_props >= 0.005) & (true_props < 0.02), 2.0),
        ((true_props >= 0.02) & (true_props < 0.05), 0.5),
    ]
    range_losses = []
    for mask, weight in critical_ranges:
        if mask.sum() > 0:
            range_error = (torch.abs(pred_props - true_props) * mask.float()).sum() / mask.sum()
            range_losses.append(weight * range_error)
    critical_range_loss = sum(range_losses) if range_losses else torch.tensor(0.0, device=device)

    # Apply coverage-weighted reconstruction loss
    safe_marker_values = torch.where(valid_mask, marker_values, reconstructed)
    recon_loss = torch.sum(
        valid_mask * coverage * torch.abs(safe_marker_values - reconstructed)
    ) / (torch.sum(valid_mask * coverage) + 1e-8)

    # Apply focal loss for presence detection
    presence_targets = (true_props > presence_threshold).float()
    presence_loss = focal_loss(presence_probs, presence_targets)

    # Apply sparsity penalty to encourage zero predictions for absent cell types
    sparsity_penalty = torch.mean(torch.abs(pred_props))

    # Apply L1 penalty on low-SNR cell types to further enforce sparsity
    l1_penalty = 0.01 * torch.mean(torch.abs(pred_props) * low_snr_mask)

    # Apply NNLS regularisation if provided
    reg_loss = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        c = torch.sigmoid(coverage / 10.0).mean(dim=1)
        reg_loss = (c * (pred_props - x_nnls).pow(2).sum(dim=1)).mean()

    # Apply weight penalty to prioritise DeepConv for low-SNR types
    weight_penalty = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        nnls_errors = torch.abs(x_nnls - true_props).detach()
        dl_errors = torch.abs(dl_props - true_props).detach()
        weights = torch.sigmoid(combination_weight).unsqueeze(0)
        low_snr_mask = torch.zeros_like(weights)
        low_snr_mask[:, low_snr_indices] = 1.0
        weight_penalty = (weights * nnls_errors * (1 - low_snr_mask) + (1 - weights) * dl_errors * low_snr_mask).mean()

    # Regularise marker weights to prevent collapse
    marker_weight_reg = (torch.sigmoid(marker_quality_weights).mean() + torch.sigmoid(marker_selection).mean()) / 2.0

    # Combine loss terms
    total_loss = (
        alpha * loss_props +
        0.3 * critical_range_loss +
        beta * recon_loss +
        0.15 * presence_loss +
        gamma * sparsity_penalty +
        l1_penalty +
        0.15 * weight_penalty +
        0.01 * marker_weight_reg
    )

    # Diagnostics
    with torch.no_grad():
        presence_preds = (presence_probs > 0.5).float()
        true_positives = torch.sum(presence_preds * presence_targets, dim=0)
        false_positives = torch.sum(presence_preds * (1 - presence_targets), dim=0)
        false_negatives = torch.sum((1 - presence_preds) * presence_targets, dim=0)
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        avg_precision = torch.mean(precision)
        avg_recall = torch.mean(recall)
        avg_f1 = torch.mean(f1)
        accuracy = torch.mean((presence_preds == presence_targets).float())
        low_conc_error = torch.mean(torch.masked_select(errors, low_conc_mask))
        med_conc_error = torch.mean(torch.masked_select(errors, med_conc_mask))
        high_conc_error = torch.mean(torch.masked_select(errors, high_conc_mask))

    details = {
        'total_loss': total_loss.item(),
        'loss_props': loss_props.item(),
        'critical_range_loss': critical_range_loss.item(),
        'recon_loss': recon_loss.item(),
        'presence_loss': presence_loss.item(),
        'sparsity_loss': sparsity_penalty.item(),
        'l1_penalty': l1_penalty.item(),
        'reg_loss': reg_loss.item(),
        'weight_penalty': weight_penalty.item(),
        'marker_weight_reg': marker_weight_reg.item(),
        'low_snr_under': underestimation[:, low_snr_indices].mean().item(),
        'low_snr_over': overestimation[:, low_snr_indices].mean().item(),
        'error_low_conc': low_conc_error.item() if not torch.isnan(low_conc_error) else 0.0,
        'error_med_conc': med_conc_error.item() if not torch.isnan(med_conc_error) else 0.0,
        'error_high_conc': high_conc_error.item() if not torch.isnan(high_conc_error) else 0.0,
        'presence_accuracy': accuracy.item(),
        'presence_precision': avg_precision.item(),
        'presence_recall': avg_recall.item(),
        'presence_f1': avg_f1.item(),
        'valid_ratio': torch.mean(valid_mask.float()).item()
    }

    return total_loss, details