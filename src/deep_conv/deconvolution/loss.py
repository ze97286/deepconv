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
    x_nnls: torch.Tensor,
    dl_props: torch.Tensor,
    combination_weight: torch.Tensor,
    alpha: float = 0.92,
    beta: float = 0.07,
    gamma: float = 0.01,
    presence_threshold: float = 0.005,
    low_snr_indices: list = [3, 4, 9, 11],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> tuple[torch.Tensor, dict]:
    """Calculate the loss for a cell-type deconvolution model with integrated presence models.

    This loss function combines multiple components to train the model effectively:
    1. Proportion error, penalising discrepancies between predicted and true cell-type proportions,
       with enhanced weighting for low-concentration and low-signal-to-noise ratio (SNR) cell types.
    2. Coverage-weighted reconstruction error, ensuring predicted proportions align with input
       methylation data.
    3. Sparsity penalty, encouraging fewer active cell types in predictions.
    4. Optional regularisation terms for ensembling with non-negative least squares (NNLS)
       predictions, though currently weighted to zero.

    The function also computes detailed diagnostics for monitoring training, including presence
    detection metrics and errors across concentration ranges.

    Args:
        pred_props (torch.FloatTensor): Predicted cell-type proportions [B, C], ensembled if x_nnls is provided.
        true_props (torch.FloatTensor): Ground-truth cell-type proportions [B, C].
        reconstructed (torch.FloatTensor): Reconstructed marker methylation values [B, M].
        marker_values (torch.FloatTensor): True marker methylation values [B, M], NaN where coverage=0.
        coverage (torch.FloatTensor): Read coverage values [B, M].
        valid_mask (torch.BoolTensor): Mask indicating valid markers (coverage > 0) [B, M].
        presence_probs (torch.FloatTensor): Presence probabilities from pre-trained models [B, C].
        presence_logits (torch.FloatTensor): Raw logits before sigmoid for presence probabilities [B, C].
        x_nnls (torch.FloatTensor): NNLS predictions [B, C], or None if ensembling is disabled.
        dl_props (torch.FloatTensor): Deep learning-only proportions before ensembling [B, C].
        combination_weight (torch.FloatTensor): Ensembling weights for DL and NNLS predictions [C].
        alpha (float, optional): Weight for proportion error term. Defaults to 0.92.
        beta (float, optional): Weight for reconstruction error term. Defaults to 0.07.
        gamma (float, optional): Weight for sparsity penalty term. Defaults to 0.01.
        presence_threshold (float, optional): Threshold for defining cell-type presence. Defaults to 0.005.
        low_snr_indices (list, optional): Indices of low-SNR cell types (e.g., T-cells, OAC). Defaults to [3, 4, 9, 11].
        device (torch.device, optional): Device for computation. Defaults to CUDA if available, else CPU.

    Returns:
        tuple:
            - torch.Tensor: Combined loss scalar.
            - dict: Flattened dictionary of diagnostic statistics, including loss components and performance metrics.
    """
    # 1. Proportion Error: Penalise differences between predicted and true proportions
    cell_errors = torch.abs(pred_props - true_props)

    # Initialise importance weights for different concentration ranges
    importance_weights = torch.ones_like(true_props)

    # Define concentration masks to prioritise low and medium concentrations
    low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
    med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
    high_conc_mask = true_props > 0.05

    # Assign higher weights to lower concentrations to improve sensitivity
    importance_weights = torch.where(low_conc_mask, 1.8, importance_weights)
    importance_weights = torch.where(med_conc_mask, 1.4, importance_weights)
    importance_weights = torch.where(high_conc_mask, 1.0, importance_weights)

    # Apply additional weighting for low-SNR cell types, capped to avoid overemphasis
    for idx in low_snr_indices:
        capped_fraction = torch.clamp(true_props[:, idx], max=0.10)
        importance_weights[:, idx] *= (1.0 + 10.0 * capped_fraction)

    # Calculate underestimation and overestimation errors
    underestimation = F.relu(true_props - pred_props)
    overestimation = F.relu(pred_props - true_props)

    # Create mask for low-SNR cell types
    low_snr_mask = torch.zeros_like(true_props)
    low_snr_mask[:, low_snr_indices] = 1.0

    # Apply penalties for underestimation, with extra penalty for low-SNR cell types
    underestimation_penalty = 1.3 * underestimation
    low_snr_under_penalty = low_snr_mask * underestimation * 0.7

    # Combine errors with weights to compute proportion loss
    weighted_errors = importance_weights * (cell_errors + underestimation_penalty + low_snr_under_penalty)
    loss_props = weighted_errors.mean()

    # 2. Coverage-Weighted Reconstruction Loss: Ensure predicted proportions align with input data
    safe_marker_values = torch.where(valid_mask, marker_values, reconstructed)
    recon_loss = torch.sum(
        valid_mask * coverage * torch.abs(safe_marker_values - reconstructed)
    ) / (torch.sum(valid_mask * coverage) + 1e-8)

    # 3. Sparsity Penalty: Encourage fewer active cell types
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1))

    # 4. NNLS Regularisation: Penalise deviation from NNLS predictions (currently disabled)
    reg_loss = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        c = coverage.mean(dim=1)
        reg_loss = (c * (pred_props - x_nnls).pow(2).sum(dim=1)).mean()

    # 5. Weight Penalty: Balance DL and NNLS contributions in ensembling (currently disabled)
    weight_penalty = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        nnls_cell_errors = torch.abs(x_nnls - true_props).detach()
        dl_cell_errors = torch.abs(dl_props - true_props).detach()
        weights = torch.sigmoid(combination_weight).unsqueeze(0)
        weight_penalty = (weights * nnls_cell_errors + (1 - weights) * dl_cell_errors).mean()

    # Combine all loss terms with specified weights
    total_loss = (
        alpha * loss_props +
        beta * recon_loss +
        gamma * sparsity_penalty +
        0.0 * reg_loss +  # Disabled to match current model behaviour
        0.0 * weight_penalty  # Disabled to match current model behaviour
    )

    # 6. Diagnostics: Compute detailed metrics for monitoring
    with torch.no_grad():
        # Convert true proportions to binary presence/absence
        presence_targets = (true_props > presence_threshold).float()
        presence_preds = (presence_probs > 0.5).float()

        # Calculate confusion matrix components
        true_positives = torch.sum(presence_preds * presence_targets, dim=0)
        false_positives = torch.sum(presence_preds * (1 - presence_targets), dim=0)
        false_negatives = torch.sum((1 - presence_preds) * presence_targets, dim=0)
        true_negatives = torch.sum((1 - presence_preds) * (1 - presence_targets), dim=0)

        # Compute presence detection metrics
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        avg_precision = torch.mean(precision)
        avg_recall = torch.mean(recall)
        avg_f1 = torch.mean(f1)
        accuracy = torch.mean((presence_preds == presence_targets).float())

        # Calculate errors for different concentration ranges
        low_conc_error = torch.mean(torch.masked_select(cell_errors, low_conc_mask))
        med_conc_error = torch.mean(torch.masked_select(cell_errors, med_conc_mask))
        high_conc_error = torch.mean(torch.masked_select(cell_errors, high_conc_mask))

    # Compile diagnostics into a flattened dictionary
    details = {
        'total_loss': total_loss.item(),
        'loss_props': loss_props.item(),
        'recon_loss': recon_loss.item(),
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