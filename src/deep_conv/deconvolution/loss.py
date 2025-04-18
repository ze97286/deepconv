import torch
import torch.nn.functional as F

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """Compute focal loss for binary classification to enhance low-SNR cell type detection.

    Args:
        pred (torch.Tensor): Predicted probabilities [B, C].
        target (torch.Tensor): Binary target labels (0 or 1) [B, C].
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
    alpha: float = 0.92,
    beta: float = 0.07,
    gamma: float = 0.01,
    presence_threshold: float = 0.01,
    low_snr_indices=[11],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """Compute the loss for cell-type deconvolution, optimising for low-SNR and low-coverage scenarios.

    This loss function combines multiple terms to improve detection of low-SNR cell types (e.g., T-cells
    at 1–2%), handle low-coverage markers, and refine NNLS ensembling:
    - Proportion error: Weighted MAE with coverage-based and low-SNR penalties.
    - Critical range loss: Targets specific concentration ranges (0.1–2%) for low-SNR types.
    - Reconstruction loss: Coverage-weighted L1 loss to ensure accurate marker reconstruction.
    - Presence loss: Focal loss to enhance binary presence detection.
    - Sparsity penalty: Encourages sparse proportion predictions.
    - NNLS regularisation and weight penalty: Aligns predictions with NNLS priors, prioritising DeepConv for low-SNR types.

    Args:
        pred_props (torch.FloatTensor): Predicted proportions [B, C], ensembled if x_nnls provided.
        true_props (torch.FloatTensor): Ground-truth proportions [B, C].
        reconstructed (torch.FloatTensor): Reconstructed marker methylation values [B, M].
        marker_values (torch.FloatTensor): True marker methylation values [B, M], NaN where coverage=0.
        coverage (torch.FloatTensor): Read coverage values [B, M].
        valid_mask (torch.BoolTensor): Mask indicating valid markers (coverage > 0) [B, M].
        presence_probs (torch.FloatTensor): Presence probabilities [B, C].
        presence_logits (torch.FloatTensor): Raw logits before sigmoid [B, C].
        x_nnls (torch.FloatTensor): NNLS predictions [B, C], or None if ensembling disabled.
        dl_props (torch.FloatTensor): Deep learning-only proportions before ensembling [B, C].
        combination_weight (torch.FloatTensor): Ensembling weights for DL and NNLS predictions [C].
        alpha (float, optional): Weight for proportion error term. Defaults to 0.92.
        beta (float, optional): Weight for reconstruction loss term. Defaults to 0.07.
        gamma (float, optional): Weight for sparsity penalty term. Defaults to 0.01.
        presence_threshold (float, optional): Threshold for presence detection. Defaults to 0.01.
        low_snr_indices (list[int], optional): Indices of low-SNR cell types (e.g., colon, oesophagus, OAC, T-cells).
            Defaults to [11] (T-cells).
        device (torch.device, optional): Device for computation. Defaults to CUDA if available, else CPU.

    Returns:
        tuple:
            - torch.Tensor: Combined loss scalar.
            - dict: Diagnostic statistics, including loss components and presence metrics.
    """
    # Proportion Error
    errors = torch.abs(pred_props - true_props)
    importance_weights = torch.ones_like(true_props)

    # Concentration-specific weighting
    low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
    med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
    high_conc_mask = true_props > 0.05
    importance_weights = torch.where(low_conc_mask, 1.8, importance_weights)
    importance_weights = torch.where(med_conc_mask, 1.4, importance_weights)
    importance_weights = torch.where(high_conc_mask, 1.0, importance_weights)

    # Low-SNR cell type weighting
    for idx in low_snr_indices:
        capped_fraction = torch.clamp(true_props[:, idx], max=0.10)
        importance_weights[:, idx] *= (1.0 + 10.0 * capped_fraction)

    # Coverage-based weighting
    coverage_weights = coverage.mean(dim=1, keepdim=True)  # [B, 1]
    weighted_errors = errors * importance_weights * coverage_weights

    # Underestimation and overestimation penalties
    underestimation = F.relu(true_props - pred_props)
    overestimation = F.relu(pred_props - true_props)  # Added to fix undefined variable
    low_snr_mask = torch.zeros_like(true_props)
    low_snr_mask[:, low_snr_indices] = 1.0
    underestimation_penalty = 1.3 * underestimation
    low_snr_under_penalty = low_snr_mask * underestimation * 0.7
    loss_props = (weighted_errors + underestimation_penalty + low_snr_under_penalty).mean()

    # Critical Range Loss for low-SNR concentrations
    critical_ranges = [
        ((true_props >= 0.001) & (true_props < 0.005), 1.0),
        ((true_props >= 0.005) & (true_props < 0.02), 2.0),  # Emphasise 0.5–2%
        ((true_props >= 0.02) & (true_props < 0.05), 0.5),
    ]
    range_losses = []
    for mask, weight in critical_ranges:
        if mask.sum() > 0:
            range_error = (torch.abs(pred_props - true_props) * mask.float()).sum() / mask.sum()
            range_losses.append(weight * range_error)
    critical_range_loss = sum(range_losses) if range_losses else torch.tensor(0.0, device=device)

    # Coverage-Weighted Reconstruction Loss
    safe_marker_values = torch.where(valid_mask, marker_values, reconstructed)
    recon_loss = torch.sum(
        valid_mask * coverage * torch.abs(safe_marker_values - reconstructed)
    ) / (torch.sum(valid_mask * coverage) + 1e-8)

    # Focal Presence Loss
    presence_targets = (true_props > presence_threshold).float()
    presence_loss = focal_loss(presence_probs, presence_targets)

    # Sparsity Penalty
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1))

    # NNLS Regularisation
    reg_loss = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        c = coverage.mean(dim=1)
        reg_loss = (c * (pred_props - x_nnls).pow(2).sum(dim=1)).mean()

    # Weight Penalty for Ensembling
    weight_penalty = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        nnls_errors = torch.abs(x_nnls - true_props).detach()
        dl_errors = torch.abs(dl_props - true_props).detach()
        weights = torch.sigmoid(combination_weight).unsqueeze(0)
        low_snr_mask = torch.zeros_like(weights)
        low_snr_mask[:, low_snr_indices] = 1.0
        weight_penalty = (weights * nnls_errors * (1 - low_snr_mask) + (1 - weights) * dl_errors * low_snr_mask).mean()

    # Combine Loss Terms
    total_loss = (
        alpha * loss_props +
        0.2 * critical_range_loss +
        beta * recon_loss +
        0.1 * presence_loss +
        gamma * sparsity_penalty +
        0.1 * reg_loss +
        0.1 * weight_penalty
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
        'reg_loss': reg_loss.item(),
        'weight_penalty': weight_penalty.item(),
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