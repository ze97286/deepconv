import torch 
import torch.nn.functional as F

def coverage_adaptive_loss(
    pred_props, true_props, reconstructed, marker_values, coverage, valid_mask,
    presence_probs, presence_logits,
    # Higher-level loss component weights
    alpha=1.0, beta=0.05, gamma=0.02,
    # Coverage weighting parameters
    coverage_weight_enabled=False,
    coverage_weight_scale=0.2,
    cov_min_weight=0.2, cov_max_weight=1.5, cov_norm_factor=20.0,
    # Proportion loss parameters
    min_frac_weight=0.5, max_frac_weight=1.5, cov_threshold=10.0,
    # Sparsity parameters
    sparsity_min=1.0, sparsity_max=3.0
):
    """
    Coverage-adaptive loss function with tunable hyperparameters.
    
    Args:
        pred_props: [B, C] Predicted cell type proportions
        true_props: [B, C] Ground truth proportions
        reconstructed: [B, M] Reconstructed marker values
        marker_values: [B, M] Original marker values
        coverage: [B, M] Coverage values
        valid_mask: [B, M] Boolean mask where coverage > 0
        presence_probs: [B, C] Presence probabilities
        presence_logits: [B, C] Presence logits before sigmoid
        alpha, beta, gamma: Loss component weights
        Various hyperparameters for coverage-based weighting
        
    Returns:
        total_loss: Combined loss value
        details: Dictionary with component values for monitoring
    """
    # Standard proportion error
    cell_errors = torch.abs(pred_props - true_props)
    
    # Calculate average coverage for each sample
    avg_coverage = coverage.mean(dim=1, keepdim=True)
    
    # Apply coverage weighting if enabled
    if coverage_weight_enabled:
        coverage_factor = torch.clamp(
            (avg_coverage / cov_threshold) * coverage_weight_scale,
            min_frac_weight,
            max_frac_weight
        )
    else:
        # No coverage weighting - use uniform weights
        coverage_factor = torch.ones_like(avg_coverage)
    
    # Apply importance weights
    importance_weights = coverage_factor * torch.ones_like(true_props)
    
    # Weight different concentration ranges differently
    low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
    med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
    
    # Weight low concentrations more heavily
    importance_weights = torch.where(low_conc_mask, importance_weights * 1.3, importance_weights)
    importance_weights = torch.where(med_conc_mask, importance_weights * 1.1, importance_weights)
    
    # Apply weights to errors
    weighted_errors = importance_weights * cell_errors
    loss_props = weighted_errors.mean()
    
    # Sparsity penalty - stronger for lower coverage
    if coverage_weight_enabled:
        sparsity_strength = torch.clamp(
            cov_threshold / (avg_coverage + 1e-8), 
            sparsity_min, 
            sparsity_max
        )
    else:
        sparsity_strength = torch.ones_like(avg_coverage)
    
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1)) * sparsity_strength.mean()
    
    # Coverage-weighted reconstruction loss - only if beta > 0
    if beta > 0:
        # Handle missing values in marker_values
        safe_marker_values = torch.where(valid_mask, marker_values, reconstructed)
        
        if coverage_weight_enabled:
            marker_weights = torch.clamp(
                coverage / cov_norm_factor, 
                cov_min_weight, 
                cov_max_weight
            )
        else:
            marker_weights = torch.ones_like(coverage)
        
        recon_loss = torch.sum(
            valid_mask * marker_weights * torch.abs(safe_marker_values - reconstructed)
        ) / (torch.sum(valid_mask * marker_weights) + 1e-8)
    else:
        # Skip reconstruction completely
        recon_loss = torch.tensor(0.0, device=pred_props.device)
    
    # Dynamically adjust component weights based on coverage
    if coverage_weight_enabled:
        coverage_ratio = torch.clamp(avg_coverage / cov_threshold, 0.5, 1.5)
        alpha_adjusted = alpha * coverage_ratio.mean()
        beta_adjusted = beta * (2 - coverage_ratio.mean())
    else:
        # Create tensor versions of the constants for consistent behavior
        alpha_adjusted = torch.tensor(alpha, device=pred_props.device)
        beta_adjusted = torch.tensor(beta, device=pred_props.device)
    
    # Combine all terms
    total_loss = alpha_adjusted * loss_props + beta_adjusted * recon_loss + gamma * sparsity_penalty
    
    # Calculate statistics for monitoring
    with torch.no_grad():
        # Presence accuracy
        presence_targets = (true_props > 0.01).float()
        presence_preds = (presence_probs > 0.5).float()
        
        # Confusion matrix
        true_positives = torch.sum(presence_preds * presence_targets, dim=0)
        false_positives = torch.sum(presence_preds * (1 - presence_targets), dim=0)
        false_negatives = torch.sum((1 - presence_preds) * presence_targets, dim=0)
        true_negatives = torch.sum((1 - presence_preds) * (1 - presence_targets), dim=0)
        
        # Metrics
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        avg_precision = torch.mean(precision)
        avg_recall = torch.mean(recall)
        avg_f1 = torch.mean(f1)
    
    # Return loss and details
    details = {
        'total_loss': total_loss.item(),
        'loss_props': loss_props.item(),
        'recon_loss': recon_loss.item(),
        'sparsity_penalty': sparsity_penalty.item(),
        'alpha_adjusted': alpha_adjusted.item(),
        'beta_adjusted': beta_adjusted.item(),
        'avg_coverage': avg_coverage.mean().item(),
        'presence_metrics': {
            'precision': avg_precision.item(),
            'recall': avg_recall.item(),
            'f1': avg_f1.item()
        }
    }
    
    return total_loss, details