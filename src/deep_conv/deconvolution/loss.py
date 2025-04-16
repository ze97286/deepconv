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
    low_snr_indices=[11],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    Loss function for deconvolution model with integrated presence models (restored with NEW x_nnls support).
    
    Args:
        pred_props (FloatTensor): [B, C], predicted proportions (ensembled if x_nnls provided).
        true_props (FloatTensor): [B, C], ground-truth proportions.
        reconstructed (FloatTensor): [B, M], reconstructed marker methylation.
        marker_values (FloatTensor): [B, M], true marker values (NaN if coverage=0).
        coverage (FloatTensor): [B, M], read coverage.
        valid_mask (BoolTensor): [B, M], True where coverage>0.
        presence_probs (FloatTensor): [B, C], presence probabilities.
        presence_logits (FloatTensor): [B, C], raw logits.
        x_nnls (FloatTensor): [B, C], NNLS predictions (NEW).
        dl_props (FloatTensor): [B, C], DL-only proportions (NEW).
        combination_weight (FloatTensor): [C], ensembling weights (NEW).
        alpha (float): Weight for proportion error.
        beta (float): Weight for reconstruction error.
        gamma (float): Weight for sparsity penalty.
        presence_threshold (float): Threshold for presence detection.
        low_snr_indices (list[int]): Low-SNR cell type indices.
        device (torch.device): Computation device.

    Returns:
        total_loss (Tensor): Combined loss.
        details (dict): Flat dictionary of statistics (NEW: flattened for train.py).
    """
    # -----------------------------
    # (1) Proportion Error 
    # -----------------------------
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
    
    loss_props = weighted_errors.mean()
    
    # -----------------------------
    # (2) Coverage-Weighted Reconstruction Loss
    # -----------------------------
    safe_marker_values = torch.where(valid_mask, marker_values, reconstructed)
    
    recon_loss = torch.sum(
        valid_mask * coverage * torch.abs(safe_marker_values - reconstructed)
    ) / (torch.sum(valid_mask * coverage) + 1e-8)
    
    # -----------------------------
    # (3) Sparsity Regularisation
    # -----------------------------
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1))
    
    # NEW: NNLS regularization (from current loss)
    reg_loss = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        c = coverage.mean(dim=1)
        reg_loss = (c * (pred_props - x_nnls).pow(2).sum(dim=1)).mean()

    # NEW: Weight penalty (from current loss)
    weight_penalty = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        nnls_cell_errors = torch.abs(x_nnls - true_props).detach()
        dl_cell_errors = torch.abs(dl_props - true_props).detach()
        weights = torch.sigmoid(combination_weight).unsqueeze(0)
        weight_penalty = (weights * nnls_cell_errors + (1 - weights) * dl_cell_errors).mean()

    # -----------------------------
    # Combine All Terms
    # -----------------------------
    total_loss = (
        alpha * loss_props +
        beta * recon_loss +
        gamma * sparsity_penalty +
        0.0 * reg_loss +  # NEW: Zero weight to match current loss when x_nnls=None
        0.0 * weight_penalty  # NEW: Zero weight
    )
    
    # -----------------------------
    # (4) Detailed Monitoring / Diagnostics
    # -----------------------------
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
        'reg_loss': reg_loss.item(),  # NEW
        'weight_penalty': weight_penalty.item(),  # NEW
        'low_snr_under': underestimation[:, low_snr_indices].mean().item(),
        'low_snr_over': overestimation[:, low_snr_indices].mean().item(),
        'error_low_conc': low_conc_error.item() if not torch.isnan(low_conc_error) else 0.0,  # NEW: Flattened
        'error_med_conc': med_conc_error.item() if not torch.isnan(med_conc_error) else 0.0,  # NEW: Flattened
        'error_high_conc': high_conc_error.item() if not torch.isnan(high_conc_error) else 0.0,  # NEW: Flattened
        'presence_accuracy': accuracy.item(),  # NEW: Flattened
        'presence_precision': avg_precision.item(),  # NEW: Flattened
        'presence_recall': avg_recall.item(),  # NEW: Flattened
        'presence_f1': avg_f1.item(),  # NEW: Flattened
        'valid_ratio': torch.mean(valid_mask.float()).item()
    }
    
    return total_loss, details