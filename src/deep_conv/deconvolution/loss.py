import torch
import torch.nn.functional as F

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """
    Compute focal loss for binary classification.
    
    Args:
        pred (torch.Tensor): Predicted probabilities, shape [B, C]
        target (torch.Tensor): Target labels (0 or 1), shape [B, C]
        gamma (float): Focusing parameter
        alpha (float): Class balancing weight
        
    Returns:
        torch.Tensor: Focal loss value
    """
    bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
    modulator = torch.pow(1 - pred, gamma) * target + torch.pow(pred, gamma) * (1 - target)
    return (alpha * modulator * bce).mean()

def loss_fn(
    pred_props: torch.Tensor,
    true_props: torch.Tensor,
    presence_probs: torch.Tensor,
    reconstructed: torch.Tensor,
    marker_values: torch.Tensor,
    coverage: torch.Tensor,
    valid_mask: torch.Tensor,
    x_nnls: torch.Tensor,
    dl_props: torch.Tensor,
    combination_weight: torch.Tensor,
    presence_threshold: float = 0.01,
    low_snr_indices=[3, 4, 9, 11],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    Loss function for cell type deconvolution with reconstruction and optional NNLS ensembling.

    Args:
        pred_props: [B, C] tensor of predicted proportions (after ensembling)
        true_props: [B, C] tensor of true proportions
        presence_probs: [B, C] tensor of presence probabilities
        reconstructed: [B, M] tensor of reconstructed marker values
        marker_values: [B, M] tensor of true marker values
        coverage: [B, M] tensor of coverage values
        valid_mask: [B, M] tensor indicating valid markers
        x_nnls: [B, C] tensor of NNLS predictions
        dl_props: [B, C] tensor of DL predictions before ensembling
        combination_weight: [C] tensor of per-cell-type weights
        presence_threshold: Threshold for presence detection
        low_snr_indices: Indices of low-SNR cell types
        device: Device for computations
    """
    # Proportion error
    cell_errors = torch.abs(pred_props - true_props)
    importance_weights = torch.ones_like(true_props)
    low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
    med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
    high_conc_mask = true_props > 0.05
    importance_weights = torch.where(low_conc_mask, 2.0, importance_weights)
    importance_weights = torch.where(med_conc_mask, 1.5, importance_weights)
    importance_weights = torch.where(high_conc_mask, 1.0, importance_weights)
    
    low_snr_mask = torch.zeros_like(true_props)
    low_snr_mask[:, low_snr_indices] = 1.0
    underestimation = F.relu(true_props - pred_props)
    underestimation_penalty = 1.3 * underestimation
    low_snr_under_penalty = low_snr_mask * underestimation * 0.7
    
    weighted_errors = importance_weights * (cell_errors + underestimation_penalty + low_snr_under_penalty)
    loss_props = weighted_errors.mean()

    # Reconstruction loss
    safe_marker_values = torch.where(valid_mask, marker_values, reconstructed)
    recon_loss = torch.sum(
        valid_mask * coverage * torch.abs(safe_marker_values - reconstructed)
    ) / (torch.sum(valid_mask * coverage) + 1e-8)

    # Presence loss
    presence_targets = (true_props > presence_threshold).float()
    presence_loss = focal_loss(presence_probs, presence_targets)

    # Sparsity penalty
    sparsity_penalty = torch.abs(pred_props).mean()

    # NNLS regularization
    reg_loss = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        c = coverage.mean(dim=1)
        reg_loss = (c * (pred_props - x_nnls).pow(2).sum(dim=1)).mean()

    # Weight penalty
    weight_penalty = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        nnls_cell_errors = torch.abs(x_nnls - true_props).detach()
        dl_cell_errors = torch.abs(dl_props - true_props).detach()
        weights = torch.sigmoid(combination_weight).unsqueeze(0)
        weight_penalty = (weights * nnls_cell_errors + (1 - weights) * dl_cell_errors).mean()

    # Combine losses
    total_loss = (
        0.6 * loss_props +        # Emphasize proportion accuracy
        0.2 * recon_loss +        # Regularize with reconstruction
        0.1 * presence_loss +     # Maintain presence accuracy
        0.05 * sparsity_penalty + # Encourage sparsity
        0.01 * reg_loss +         # Light NNLS regularization
        0.05 * weight_penalty     # Light weight penalty
    )

    # Logging details (flattened)
    details = {
        'loss_props': loss_props.item(),
        'recon_loss': recon_loss.item(),
        'presence_loss': presence_loss.item(),
        'sparsity_loss': sparsity_penalty.item(),
        'reg_loss': reg_loss.item(),
        'weight_penalty': weight_penalty.item(),
        'low_snr_under': underestimation[:, low_snr_indices].mean().item(),
        'low_snr_over': F.relu(pred_props - true_props)[:, low_snr_indices].mean().item(),
        'concentration_low_conc': torch.mean(torch.masked_select(cell_errors, low_conc_mask)).item() if low_conc_mask.any() else 0.0,
        'concentration_med_conc': torch.mean(torch.masked_select(cell_errors, med_conc_mask)).item() if med_conc_mask.any() else 0.0,
        'concentration_high_conc': torch.mean(torch.masked_select(cell_errors, high_conc_mask)).item() if high_conc_mask.any() else 0.0
    }

    return total_loss, details