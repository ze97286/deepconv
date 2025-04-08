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
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    Loss function for deconvolution model with integrated presence models.
    
    This updated version focuses on proportion accuracy with presence models
    treated as features rather than binary gates.

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
            Sigmoid probabilities from the pre-trained presence detection models.
        presence_logits (FloatTensor): [B, C]
            Logits (before sigmoid) from the pre-trained presence detection models.
        alpha (float):
            Weight for the proportion error term (often near 1.0).
        beta (float):
            Weight for the reconstruction term (marker-level error).
        gamma (float):
            Weight for the sparsity penalty (discourage spread-out predictions).
        presence_threshold (float):
            Threshold on true_props to determine presence (for evaluation metrics).
        low_snr_indices (list[int]):
            Indices of cell types considered "low SNR" or difficult to detect.
        device (torch.device):
            Computation device.

    Returns:
        total_loss (Tensor):
            A scalar tensor representing the combined loss.
        details (dict):
            A dictionary of intermediate scalars/statistics for monitoring.
    """
    # -----------------------------
    # (1) Proportion Error 
    # -----------------------------
    # Standard mean absolute error
    cell_errors = torch.abs(pred_props - true_props)
    
    # Create importance weights for different concentration levels
    importance_weights = torch.ones_like(true_props)
    
    # Concentration-dependent masks for weighting
    low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
    med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
    high_conc_mask = true_props > 0.05
    
    # Scale importance by concentration range - gentler scaling
    importance_weights = torch.where(low_conc_mask, 1.8, importance_weights)   # Lower weight than before
    importance_weights = torch.where(med_conc_mask, 1.4, importance_weights)   # Lower weight than before
    importance_weights = torch.where(high_conc_mask, 1.0, importance_weights)
    
    # Special handling for low-SNR cell types - less aggressive
    for idx in low_snr_indices:
        capped_fraction = torch.clamp(true_props[:, idx], max=0.10)
        importance_weights[:, idx] *= (1.0 + 10.0 * capped_fraction)  # Reduced multiplier
    
    # Calculate underestimation/overestimation
    underestimation = F.relu(true_props - pred_props)  # only positive if true>pred
    overestimation = F.relu(pred_props - true_props)   # only positive if pred>true
    
    # Create low-SNR mask
    low_snr_mask = torch.zeros_like(true_props)
    low_snr_mask[:, low_snr_indices] = 1.0
    
    # Apply moderate penalty for underestimation 
    underestimation_penalty = 1.3 * underestimation  # Reduced from 1.5
    
    # Less aggressive special penalty for low-SNR underestimation
    low_snr_under_penalty = low_snr_mask * underestimation * 0.7  # Reduced further
    
    # Combine into weighted errors
    weighted_errors = importance_weights * (cell_errors + underestimation_penalty + low_snr_under_penalty)
    
    # Mean across all cells in the batch
    loss_props = weighted_errors.mean()
    
    # -----------------------------
    # (2) Coverage-Weighted Reconstruction Loss
    # -----------------------------
    # Replace NaN marker values with reconstructed values where invalid
    safe_marker_values = torch.where(valid_mask, marker_values, reconstructed)
    
    # Coverage-weighted L1 loss
    recon_loss = torch.sum(
        valid_mask * coverage * torch.abs(safe_marker_values - reconstructed)
    ) / (torch.sum(valid_mask * coverage) + 1e-8)
    
    # -----------------------------
    # (3) Sparsity Regularisation (very low weight)
    # -----------------------------
    # Encourages the model to predict fewer cell types present
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1))
    
    # -----------------------------
    # Combine All Terms
    # -----------------------------
    total_loss = alpha * loss_props + beta * recon_loss + gamma * sparsity_penalty
    
    # -----------------------------
    # (4) Detailed Monitoring / Diagnostics
    # -----------------------------
    with torch.no_grad():
        # Convert true_props to presence vs. absence based on threshold
        presence_targets = (true_props > presence_threshold).float()
        
        # Use pre-trained presence probabilities for evaluation
        presence_preds = (presence_probs > 0.5).float()
        
        # Confusion counts
        true_positives = torch.sum(presence_preds * presence_targets, dim=0)
        false_positives = torch.sum(presence_preds * (1 - presence_targets), dim=0)
        false_negatives = torch.sum((1 - presence_preds) * presence_targets, dim=0)
        true_negatives = torch.sum((1 - presence_preds) * (1 - presence_targets), dim=0)
        
        # Precision / Recall / F1 
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