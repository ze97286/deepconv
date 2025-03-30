import torch 
import torch.nn.functional as F

def coverage_adaptive_loss(
    pred_props, true_props, reconstructed, marker_values, 
    coverage, valid_mask, feature_quality,
    alpha=1.0, beta=0.05, gamma=0.02, delta=0.1
):
    """
    Loss function with coverage-adaptive components
    
    Args:
        pred_props: [B, C] Predicted cell type proportions
        true_props: [B, C] Ground truth proportions
        reconstructed: [B, M] Reconstructed marker values
        marker_values: [B, M] Original marker values
        coverage: [B, M] Coverage values
        valid_mask: [B, M] Valid marker mask
        feature_quality: [B, C] Feature quality scores
        alpha: Weight for proportion error
        beta: Weight for reconstruction error
        gamma: Weight for sparsity penalty
        delta: Weight for quality-weighted error
        
    Returns:
        total_loss: Combined loss value
        details: Dictionary with component values
    """
    # Calculate average coverage per sample
    avg_coverage = coverage.mean(dim=1, keepdim=True)
    log_coverage = torch.log1p(avg_coverage)
    
    # 1. Basic proportion error (absolute difference)
    basic_error = torch.abs(pred_props - true_props)
    
    # 2. Coverage-weighted error
    # Use log_coverage for smoother scaling with a reasonable range
    # Higher weight for low coverage samples
    coverage_factor = torch.clamp(2.0 / (log_coverage + 0.5), 0.5, 2.0)
    weighted_error = basic_error * coverage_factor
    
    # 3. Adaptive error weighting by concentration
    # More emphasis on concentrations in the middle range
    concentration_weights = torch.ones_like(true_props)
    low_mask = (true_props > 0.0001) & (true_props <= 0.01)
    mid_mask = (true_props > 0.01) & (true_props <= 0.1)
    high_mask = true_props > 0.1
    
    concentration_weights[low_mask] = 1.5    # Higher weight for low concentrations
    concentration_weights[mid_mask] = 1.2    # Slightly higher for mid concentrations
    concentration_weights[high_mask] = 1.0   # Normal weight for high concentrations
    
    # 4. Quality-weighted error
    # Lower quality should mean lower impact on loss
    quality_weighted_error = weighted_error * torch.sqrt(feature_quality)
    
    # 5. Combined proportion loss
    prop_loss = (weighted_error * concentration_weights).mean()
    quality_loss = quality_weighted_error.mean()
    
    # 6. Reconstruction loss (if used)
    if beta > 0:
        safe_markers = torch.where(valid_mask, marker_values, reconstructed)
        recon_error = valid_mask * torch.abs(safe_markers - reconstructed)
        
        # Use log coverage for weighting reconstruction loss too
        coverage_weights = torch.clamp(log_coverage / 2.0, 0.1, 1.0)
        weighted_recon = coverage_weights * recon_error.mean(dim=1, keepdim=True)
        recon_loss = weighted_recon.mean()
    else:
        recon_loss = torch.tensor(0.0, device=pred_props.device)
    
    # 7. Sparsity penalty
    # Stronger for low coverage to prevent over-prediction
    # Use log_coverage for smoother scaling
    sparsity_strength = torch.clamp(1.5 / (log_coverage + 0.5), 1.0, 3.0)
    sparsity_penalty = torch.mean(torch.sum(pred_props, dim=1)) * sparsity_strength.mean()
    
    # 8. Combined loss
    total_loss = alpha * prop_loss + beta * recon_loss + gamma * sparsity_penalty + delta * quality_loss
    
    # Return loss and components
    details = {
        'prop_loss': prop_loss.item(),
        'quality_loss': quality_loss.item(),
        'recon_loss': recon_loss.item(),
        'sparsity': sparsity_penalty.item(),
        'avg_coverage': avg_coverage.mean().item(),
        'log_coverage': log_coverage.mean().item()
    }
    
    return total_loss, details