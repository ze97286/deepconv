import torch 
import torch.nn.functional as F

def efficient_coverage_aware_loss(
    pred_props: torch.Tensor,
    true_props: torch.Tensor,
    reconstructed: torch.Tensor,
    marker_values: torch.Tensor,
    coverage: torch.Tensor,
    valid_mask: torch.Tensor,
    presence_probs: torch.Tensor,
    presence_logits: torch.Tensor,
    reliability_scores: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    log_space: bool = True,
    presence_threshold: float = 0.005,
    low_snr_indices = None
):
    """
    Efficient coverage-aware loss function for cell type deconvolution.
    
    Args:
        pred_props: [B, C] Predicted cell type proportions
        true_props: [B, C] Ground truth proportions
        reconstructed: [B, M] Reconstructed marker values
        marker_values: [B, M] Original marker values
        coverage: [B, M] Coverage values
        valid_mask: [B, M] Mask indicating valid markers (coverage > 0)
        presence_probs: [B, C] Presence probabilities from presence models
        presence_logits: [B, C] Presence logits from presence models
        reliability_scores: [B, M] Reliability scores for markers
        alpha: Weight for proportion error
        beta: Weight for reconstruction error
        log_space: Whether to use log-space for concentration error
        presence_threshold: Threshold for presence metrics
        low_snr_indices: List of indices for cell types with low SNR
    
    Returns:
        total_loss: Combined loss value
        details: Dictionary of metrics for monitoring
    """
    # Set default low SNR indices if not provided
    if low_snr_indices is None:
        low_snr_indices = []
    
    # 1. Proportion error calculation
    if log_space and (true_props > 0).all():
        # Calculate error in log space for better handling of small concentrations
        epsilon = 1e-6
        log_pred = torch.log(pred_props + epsilon)
        log_true = torch.log(true_props + epsilon)
        
        # Mean squared error in log space
        prop_errors = torch.pow(log_pred - log_true, 2)
        
        # Weight by original concentration (give more weight to higher concentrations)
        prop_weights = torch.sqrt(true_props + epsilon)
        weighted_prop_errors = prop_errors * prop_weights
    else:
        # Standard absolute error
        prop_errors = torch.abs(pred_props - true_props)
        
        # Create importance weights for different concentration levels
        prop_weights = torch.ones_like(prop_errors)
        
        # Concentration-dependent masks
        low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
        med_conc_mask = (true_props > 0.01) & (true_props <= 0.05)
        high_conc_mask = true_props > 0.05
        
        # Scale importance by concentration range
        prop_weights = torch.where(low_conc_mask, torch.tensor(1.8, device=prop_weights.device), prop_weights)
        prop_weights = torch.where(med_conc_mask, torch.tensor(1.4, device=prop_weights.device), prop_weights)
        prop_weights = torch.where(high_conc_mask, torch.tensor(1.0, device=prop_weights.device), prop_weights)
        
        # Special handling for low-SNR cell types
        for idx in low_snr_indices:
            capped_fraction = torch.clamp(true_props[:, idx], max=0.10)
            prop_weights[:, idx] *= (1.0 + 10.0 * capped_fraction)
        
        # Calculate underestimation/overestimation
        underestimation = F.relu(true_props - pred_props)  # only positive if true>pred
        overestimation = F.relu(pred_props - true_props)   # only positive if pred>true
        
        # Create low-SNR mask
        low_snr_mask = torch.zeros_like(true_props)
        if low_snr_indices:
            low_snr_mask[:, low_snr_indices] = 1.0
        
        # Apply penalty for underestimation
        underestimation_penalty = 1.3 * underestimation
        
        # Special penalty for low-SNR underestimation
        low_snr_under_penalty = low_snr_mask * underestimation * 0.7
        
        # Combine into weighted errors
        weighted_prop_errors = prop_weights * (prop_errors + underestimation_penalty + low_snr_under_penalty)
    
    # Mean proportion error across batch
    prop_loss = weighted_prop_errors.mean()
    
    # 2. Reconstruction error weighted by reliability
    # Replace invalid marker values with reconstructed values
    safe_markers = torch.where(valid_mask, marker_values, reconstructed)
    
    # Calculate reconstruction errors
    recon_errors = torch.abs(safe_markers - reconstructed)
    
    # Weight by reliability scores
    weighted_recon = recon_errors * reliability_scores * valid_mask.float()
    
    # Normalize by sum of weights
    recon_loss = weighted_recon.sum() / (reliability_scores * valid_mask.float()).sum().clamp(min=1e-8)
    
    # 3. Combine losses
    total_loss = alpha * prop_loss + beta * recon_loss
    
    # 4. Calculate metrics for monitoring
    with torch.no_grad():
        # Basic error metrics
        mae = torch.abs(pred_props - true_props).mean().item()
        mse = torch.pow(pred_props - true_props, 2).mean().item()
        
        # Presence metrics
        presence_targets = (true_props > presence_threshold).float()
        presence_preds = (presence_probs > 0.5).float()
        
        # Confusion matrix
        tp = torch.sum(presence_preds * presence_targets, dim=0)
        fp = torch.sum(presence_preds * (1 - presence_targets), dim=0)
        fn = torch.sum((1 - presence_preds) * presence_targets, dim=0)
        tn = torch.sum((1 - presence_preds) * (1 - presence_targets), dim=0)
        
        # Precision, recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        
        avg_precision = torch.mean(precision).item()
        avg_recall = torch.mean(recall).item()
        avg_f1 = torch.mean(f1).item()
        accuracy = torch.mean((presence_preds == presence_targets).float()).item()
        
        # Error in different concentration ranges
        low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
        med_conc_mask = (true_props > 0.01) & (true_props <= 0.1)
        high_conc_mask = true_props > 0.1
        
        low_conc_error = torch.mean(torch.masked_select(prop_errors, low_conc_mask))
        med_conc_error = torch.mean(torch.masked_select(prop_errors, med_conc_mask))
        high_conc_error = torch.mean(torch.masked_select(prop_errors, high_conc_mask))
        
        if torch.isnan(low_conc_error):
            low_conc_error = torch.tensor(0.0, device=prop_errors.device)
        if torch.isnan(med_conc_error):
            med_conc_error = torch.tensor(0.0, device=prop_errors.device)
        if torch.isnan(high_conc_error):
            high_conc_error = torch.tensor(0.0, device=prop_errors.device)
        
        # Error by coverage level
        mean_coverage = coverage.mean(dim=1)
        low_cov_mask = mean_coverage < 10.0
        med_cov_mask = (mean_coverage >= 10.0) & (mean_coverage < 30.0)
        high_cov_mask = mean_coverage >= 30.0
        
        # Expanded error tensor for masking by sample
        expanded_errors = torch.mean(prop_errors, dim=1)
        
        low_cov_error = torch.mean(expanded_errors[low_cov_mask]) if low_cov_mask.any() else torch.tensor(0.0, device=prop_errors.device)
        med_cov_error = torch.mean(expanded_errors[med_cov_mask]) if med_cov_mask.any() else torch.tensor(0.0, device=prop_errors.device)
        high_cov_error = torch.mean(expanded_errors[high_cov_mask]) if high_cov_mask.any() else torch.tensor(0.0, device=prop_errors.device)
        
        # Reliability stats
        avg_reliability = reliability_scores.mean().item()
        reliability_std = reliability_scores.std().item()
    
    # Detailed metrics dictionary
    details = {
        'total_loss': total_loss.item(),
        'prop_loss': prop_loss.item(),
        'recon_loss': recon_loss.item(),
        'mae': mae,
        'mse': mse,
        'alpha_stats': {
            'mean': torch.mean(pred_props).item(),
            'std': torch.std(pred_props).item(),
            'max': torch.max(pred_props).item(),
            'min': torch.min(pred_props).item()
        },
        'presence_stats': {
            'accuracy': accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'true_positives': torch.sum(tp).item(),
            'false_positives': torch.sum(fp).item(),
            'false_negatives': torch.sum(fn).item(),
            'true_negatives': torch.sum(tn).item()
        },
        'concentration_errors': {
            'low_conc': low_conc_error.item(),
            'med_conc': med_conc_error.item(),
            'high_conc': high_conc_error.item()
        },
        'coverage_errors': {
            'low_cov': low_cov_error.item(),
            'med_cov': med_cov_error.item(),
            'high_cov': high_cov_error.item()
        },
        'reliability_stats': {
            'mean': avg_reliability,
            'std': reliability_std
        },
        'valid_ratio': torch.mean(valid_mask.float()).item()
    }
    
    # Add low SNR metrics if applicable
    if low_snr_indices:
        details['low_snr_under'] = underestimation[:, low_snr_indices].mean().item()
        details['low_snr_over'] = overestimation[:, low_snr_indices].mean().item()
    
    return total_loss, details