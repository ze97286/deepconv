import torch 
import torch.nn.functional as F

def zero_focused_adaptive_loss(
    pred_props, true_props, reconstructed, marker_values, 
    coverage, valid_mask, feature_quality, presence_probs=None,
    alpha=1.0, beta=0.05, gamma=0.03, delta=0.2, zero_weight=5.0
):
    """
    Loss function focusing on correct zero classification
    
    Args:
        pred_props: [B, C] Predicted cell type proportions
        true_props: [B, C] Ground truth proportions
        reconstructed: [B, M] Reconstructed marker values
        marker_values: [B, M] Original marker values
        coverage: [B, M] Coverage values
        valid_mask: [B, M] Valid marker mask
        feature_quality: [B, C] Feature quality scores
        presence_probs: [B, C] Predicted presence probabilities
        alpha: Weight for proportion error
        beta: Weight for reconstruction error
        gamma: Weight for sparsity penalty
        delta: Weight for quality-weighted error
        zero_weight: Extra weight for false positive errors
        
    Returns:
        total_loss: Combined loss value
        details: Dictionary with component values
    """
    # Calculate average coverage per sample
    avg_coverage = coverage.mean(dim=1, keepdim=True)
    log_coverage = torch.log1p(avg_coverage)
    
    # 1. Basic proportion error (absolute difference)
    basic_error = torch.abs(pred_props - true_props)
    
    # 2. Zero-focused error (extra penalty for false positives)
    zero_mask = (true_props <= 0.001)
    
    # Calculate false positive error (predicting non-zero when should be zero)
    # Stronger penalty based on cell type and coverage
    false_positive_error = torch.where(
        zero_mask,
        zero_weight * pred_props,  # Penalty proportional to prediction magnitude
        torch.zeros_like(pred_props)
    )
    
    # 3. Coverage-weighted error
    # Use log_coverage for smoother scaling with a reasonable range
    coverage_factor = torch.clamp(2.0 / (log_coverage + 0.5), 0.5, 2.0)
    weighted_error = (basic_error + false_positive_error) * coverage_factor
    
    # 4. Adaptive error weighting by concentration
    concentration_weights = torch.ones_like(true_props)
    low_mask = (true_props > 0.0001) & (true_props <= 0.01)
    mid_mask = (true_props > 0.01) & (true_props <= 0.1)
    high_mask = true_props > 0.1
    
    concentration_weights[low_mask] = 1.5    # Higher weight for low concentrations
    concentration_weights[mid_mask] = 1.2    # Slightly higher for mid concentrations
    concentration_weights[high_mask] = 1.0   # Normal weight for high concentrations
    
    # 5. Quality-weighted error
    quality_weighted_error = weighted_error * torch.sqrt(feature_quality)
    
    # 6. Combined proportion loss
    prop_loss = (weighted_error * concentration_weights).mean()
    quality_loss = quality_weighted_error.mean()
    
    # 7. Presence classification loss (if provided)
    if presence_probs is not None:
        presence_targets = (true_props > 0.001).float()
        presence_loss = F.binary_cross_entropy(
            presence_probs, presence_targets, reduction='mean'
        )
    else:
        presence_loss = torch.tensor(0.0, device=pred_props.device)
    
    # 8. Reconstruction loss (if used)
    if beta > 0:
        safe_markers = torch.where(valid_mask, marker_values, reconstructed)
        recon_error = valid_mask * torch.abs(safe_markers - reconstructed)
        
        # Use log coverage for weighting reconstruction loss too
        coverage_weights = torch.clamp(log_coverage / 2.0, 0.1, 1.0)
        weighted_recon = coverage_weights * recon_error.mean(dim=1, keepdim=True)
        recon_loss = weighted_recon.mean()
    else:
        recon_loss = torch.tensor(0.0, device=pred_props.device)
    
    # 9. Sparsity penalty
    # Stronger for low coverage to prevent over-prediction
    # Use log_coverage for smoother scaling
    sparsity_strength = torch.clamp(1.5 / (log_coverage + 0.5), 1.0, 3.0)
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1)) * sparsity_strength.mean()
    
    # 10. Combined loss
    total_loss = (
        alpha * prop_loss + 
        beta * recon_loss + 
        gamma * sparsity_penalty + 
        delta * quality_loss + 
        (1.0 if presence_probs is not None else 0.0) * presence_loss
    )
    
    # Return loss and components
    details = {
        'prop_loss': prop_loss.item(),
        'quality_loss': quality_loss.item(),
        'recon_loss': recon_loss.item(),
        'sparsity': sparsity_penalty.item(),
        'presence_loss': presence_loss.item() if presence_probs is not None else 0.0,
        'avg_coverage': avg_coverage.mean().item(),
        'log_coverage': log_coverage.mean().item()
    }
    
    return total_loss, details