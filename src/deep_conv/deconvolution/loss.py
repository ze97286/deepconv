from typing import List
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
    alpha: float = 0.4,
    beta: float = 0.02,
    gamma: float = 0.2,
    presence_threshold: float = 0.01,
    low_snr_indices=[11],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    focal_loss_weight: float = 0.01,
    compute_diagnostics: bool = True,
    corr_weight: float = 0.5,
    target_cell_indices=None,
    log_vars=None
):
    # Get or initialiדe learnable log-variances for dynamic weighting
    if log_vars is None:
        # These will be created but not persisted between calls
        log_var_mae = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        log_var_corr = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        log_var_presence = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        log_var_sparsity = torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
    else:
        # Use the passed log_vars
        log_var_mae = log_vars['mae']
        log_var_corr = log_vars['corr']
        log_var_presence = log_vars['presence']
        log_var_sparsity = log_vars['sparsity']
    
    # Cap the task uncertainty weights to prevent any term from dominating
    task_uncertainty = {
        'mae': torch.clamp(torch.exp(-log_var_mae), max=1.0),
        'corr': torch.clamp(torch.exp(-log_var_corr), max=1.0),
        'presence': torch.clamp(torch.exp(-log_var_presence), max=1.0),
        'sparsity': torch.clamp(torch.exp(-log_var_sparsity), max=1.0)
    }
    
    # Sanitize coverage
    if torch.isnan(coverage).any() or torch.isinf(coverage).any():
        print("Warning: coverage contains nan or inf values")
    if (coverage < 0).any():
        print("Warning: coverage contains negative values")
    coverage = torch.nan_to_num(coverage, nan=0.0, posinf=0.0, neginf=0.0)
    coverage = torch.clamp(coverage, min=0.0)

    if torch.isnan(pred_props).any() or torch.isinf(pred_props).any():
        print("Warning: pred_props contains nan or inf values")
    if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
        print("Warning: reconstructed contains nan or inf values")
    if torch.isnan(presence_probs).any() or torch.isinf(presence_probs).any():
        print("Warning: presence_probs contains nan or inf values")

    # Ensure valid_mask is False for nan values in coverage
    valid_mask = (coverage > 0) & (~torch.isnan(coverage))

    # Compute average coverage for the batch
    avg_coverage = torch.mean(coverage[valid_mask]).item()

    # Check if we have a specialized dataset (T-cells or OAC)
    is_specialized = target_cell_indices is not None and len(target_cell_indices) == 1
    
    # Compute all basic components needed for both paths
    cell_errors = torch.abs(pred_props - true_props)
    underestimation = F.relu(true_props - pred_props)
    overestimation = F.relu(pred_props - true_props)
    
    # Focal Loss for Presence Detection - always compute this
    presence_probs_clipped = torch.clamp(presence_probs, 0.0, 1.0)
    presence_targets = (true_props > presence_threshold).float()
    presence_loss = focal_loss(
        presence_probs_clipped,
        presence_targets,
        alpha_pos=0.25,
        alpha_neg=0.75,
        gamma=2.0,
        fp_weight=1.5,
        class_weights=None
    )
    
    # For specialized datasets (T-cells, OAC), use targeted loss
    if is_specialized:
        target_idx = target_cell_indices[0]
        
        # 1. Check variance of target cell type
        target_true = true_props[:, target_idx]
        target_variance = torch.var(target_true)
        target_has_variance = target_variance >= 1e-4
        
        # 2. MAE for target cell type with underestimation penalty (if sufficient variance)
        target_error = cell_errors[:, target_idx]
        target_underestimation = underestimation[:, target_idx]
        target_overestimation = overestimation[:, target_idx]
        
        if target_has_variance:
            target_mae = target_error.mean() + 0.3 * target_underestimation.mean()
        else:
            target_mae = torch.tensor(0.0, device=device)
        
        # 3. Correlation loss for target cell type (if sufficient variance)
        if target_has_variance:
            target_pred = pred_props[:, target_idx]
            true_mean = torch.mean(target_true)
            pred_mean = torch.mean(target_pred)
            true_centered = target_true - true_mean
            pred_centered = target_pred - pred_mean
            cov = torch.mean(true_centered * pred_centered)
            true_std = torch.std(target_true, unbiased=False)
            pred_std = torch.std(target_pred, unbiased=False)
            
            if true_std < 1e-6 or pred_std < 1e-6:
                target_corr = torch.tensor(0.0, device=device)
                corr_loss = torch.tensor(1.0, device=device)
            else:
                target_corr = cov / (true_std * pred_std + 1e-8)
                corr_loss = 1.0 - target_corr
        else:
            target_corr = torch.tensor(0.0, device=device)
            corr_loss = torch.tensor(0.0, device=device)
        
        # 4. Target-specific presence detection loss
        target_presence_loss = focal_loss(
            presence_probs_clipped[:, target_idx].unsqueeze(1),
            presence_targets[:, target_idx].unsqueeze(1),
            alpha_pos=0.25,
            alpha_neg=0.75,
            gamma=2.0,
            fp_weight=1.5
        )
        
        # 5. Targeted sparsity penalty (only for non-target cell types)
        non_target_mask = torch.ones_like(pred_props)
        non_target_mask[:, target_idx] = 0.0
        sparsity_penalty = torch.mean(torch.abs(pred_props * non_target_mask))
        
        # Compute specialized loss with dynamic weighting
        total_loss = (
            task_uncertainty['mae'] * target_mae + 0.5 * log_var_mae +
            task_uncertainty['corr'] * corr_loss + 0.5 * log_var_corr +
            task_uncertainty['presence'] * target_presence_loss + 0.5 * log_var_presence +
            task_uncertainty['sparsity'] * sparsity_penalty + 0.5 * log_var_sparsity
        )
    else:
        # Standard loss calculation for general datasets
        
        # Dynamically adjust gamma based on coverage
        effective_gamma = gamma
        if avg_coverage < 10:
            effective_gamma = gamma * 0.5
        
        # Proportion Error (loss_props)
        importance_weights = torch.ones_like(true_props)
        low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
        med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
        high_conc_mask = true_props > 0.05
        importance_weights = torch.where(low_conc_mask, 2.0, importance_weights)
        importance_weights = torch.where(med_conc_mask, 1.6, importance_weights)
        importance_weights = torch.where(high_conc_mask, 1.2, importance_weights)

        # Increase importance for T-cells in low-coverage scenarios
        if avg_coverage < 10:
            importance_weights[:, low_snr_indices] *= 1.5
        
        capped_fraction = torch.clamp(true_props[:, low_snr_indices], max=0.10)
        importance_weights[:, low_snr_indices] *= (1.0 + 10.0 * capped_fraction)
        
        low_snr_mask = torch.zeros_like(true_props)
        low_snr_mask[:, low_snr_indices] = 1.0
        underestimation_penalty = 1.3 * underestimation
        low_snr_under_penalty = low_snr_mask * underestimation * 1.2
        weighted_errors = importance_weights * (cell_errors + underestimation_penalty + low_snr_under_penalty)
        
        # Focus on meaningful variance
        variance = torch.var(true_props, dim=0)
        variance_mask = (variance > 1e-4).float()
        if variance_mask.sum() > 0:
            loss_props = (weighted_errors * variance_mask.unsqueeze(0)).sum() / (variance_mask.sum() * pred_props.size(0))
        else:
            loss_props = weighted_errors.mean()

        # Reconstruction Loss
        errors = torch.abs(marker_values - reconstructed)
        errors = torch.where(valid_mask, errors, torch.zeros_like(errors))
        weighted_errors = valid_mask * coverage * errors
        denominator = torch.sum(valid_mask * coverage) + 1e-8
        if denominator < 1e-7:
            recon_loss = torch.tensor(0.0, device=device)
        else:
            recon_loss = torch.sum(weighted_errors) / denominator

        # Sparsity Regularization
        sparsity_penalty = torch.mean(torch.abs(pred_props))

        # General Correlation Loss
        num_cell_types = pred_props.shape[1]
        cell_indices = range(num_cell_types)
        if target_cell_indices is not None:
            cell_indices = target_cell_indices
            
        total_corr = 0.0
        valid_cell_types = 0
        for i in cell_indices:
            true_vals = true_props[:, i]
            pred_vals = pred_props[:, i]
            # Skip if almost constant (very low variance)
            if torch.var(true_vals) < 1e-4:
                continue
                
            true_mean = torch.mean(true_vals)
            pred_mean = torch.mean(pred_vals)
            true_centered = true_vals - true_mean
            pred_centered = pred_vals - pred_mean
            cov = torch.mean(true_centered * pred_centered)
            true_std = torch.std(true_vals, unbiased=False)
            pred_std = torch.std(pred_vals, unbiased=False)
            if true_std < 1e-6 or pred_std < 1e-6:
                corr = 0.0
            else:
                corr = cov / (true_std * pred_std + 1e-8)
                valid_cell_types += 1
            total_corr += corr
            
        avg_corr = total_corr / (valid_cell_types if valid_cell_types > 0 else 1)
        corr_loss = 1.0 - avg_corr

        # Combine all terms
        total_loss = alpha * loss_props + beta * recon_loss + effective_gamma * sparsity_penalty + focal_loss_weight * presence_loss + corr_weight * corr_loss

    # Ensure total loss is non-negative
    total_loss = torch.clamp(total_loss, min=0.0)

    # Diagnostics - keep all original calculations 
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

        # Standard details dictionary - keeping all original fields
        details = {
            "total_loss": total_loss.item(),
            "specialized_loss": is_specialized,
            "loss_props": loss_props.item() if not is_specialized else weighted_errors.mean().item(),
            "recon_loss": recon_loss.item() if not is_specialized else 0.0,
            "corr_loss": corr_loss.item(),
            "sparsity_loss": sparsity_penalty.item(),
            "presence_loss": presence_loss.item(),
            "low_snr_under": underestimation[:, low_snr_indices].mean().item(),
            "low_snr_over": overestimation[:, low_snr_indices].mean().item(),
            "alpha_stats": {
                "mean": torch.mean(pred_props).item(),
                "std": torch.std(pred_props).item(),
                "max": torch.max(pred_props).item(),
                "min": torch.min(pred_props).item(),
            },
            "concentration_errors": {
                "low_conc": (
                    low_conc_error.item() if not torch.isnan(low_conc_error) else 0.0
                ),
                "med_conc": (
                    med_conc_error.item() if not torch.isnan(med_conc_error) else 0.0
                ),
                "high_conc": (
                    high_conc_error.item() if not torch.isnan(high_conc_error) else 0.0
                ),
            },
            "presence_stats": {
                "accuracy": accuracy.item(),
                "avg_precision": avg_precision.item(),
                "avg_recall": avg_recall.item(),
                "avg_f1": avg_f1.item(),
                "true_positives": torch.sum(true_positives).item(),
                "false_positives": torch.sum(false_positives).item(),
                "false_negatives": torch.sum(false_negatives).item(),
                "true_negatives": torch.sum(true_negatives).item(),
            },
            "valid_ratio": torch.mean(valid_mask.float()).item(),
        }
        
        # Add specialized metrics if using specialized loss
        if is_specialized:
            target_idx = target_cell_indices[0]
            details["target_idx"] = target_idx
            details["target_mae"] = target_mae.item() 
            details["target_corr"] = target_corr.item() if hasattr(target_corr, 'item') else 0.0
            details["target_presence_loss"] = target_presence_loss.item() 
            details["target_has_variance"] = target_has_variance.item() if hasattr(target_has_variance, 'item') else False
            details["task_weights"] = {
                "mae": task_uncertainty['mae'].item(),
                "corr": task_uncertainty['corr'].item(),
                "presence": task_uncertainty['presence'].item(),
                "sparsity": task_uncertainty['sparsity'].item()
            }

    return total_loss, details, log_var_mae, log_var_corr, log_var_presence, log_var_sparsity