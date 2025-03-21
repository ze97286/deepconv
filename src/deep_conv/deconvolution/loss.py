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
    reliability_scores: torch.Tensor = None,
    alpha: float = 0.7,
    beta: float = 0.3,
    log_concentration: bool = True,
    presence_threshold: float = 0.005
):
    """
    Coverage-aware loss function that leverages marker reliability information.
    
    Args:
        pred_props: [B, C] Predicted cell type proportions
        true_props: [B, C] Ground truth proportions
        reconstructed: [B, M] Reconstructed marker values
        marker_values: [B, M] Original marker values
        coverage: [B, M] Coverage values
        valid_mask: [B, M] Mask indicating valid markers (coverage > 0)
        presence_probs: [B, C] Presence probabilities from presence models
        presence_logits: [B, C] Presence logits from presence models
        reliability_scores: [B, M] Reliability scores for each marker (optional)
        alpha: Weight for proportion error
        beta: Weight for reconstruction error
        log_concentration: Whether to use log-space for concentration error
        presence_threshold: Threshold for presence metrics
    """
    # 1. Proportion error (optionally in log space)
    if log_concentration:
        # Use log-space to better handle the wide range of concentrations
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        log_pred = torch.log(pred_props + epsilon)
        log_true = torch.log(true_props + epsilon)
        
        # Error is mean squared error in log space
        cell_errors = (log_pred - log_true) ** 2
        
        # Concentration-dependent weights (more weight to higher concentrations)
        conc_weights = torch.sqrt(true_props + epsilon)
        weighted_errors = cell_errors * conc_weights
    else:
        # Standard mean absolute error
        cell_errors = torch.abs(pred_props - true_props)
        
        # Concentration-dependent weights
        low_conc_mask = (true_props > 0.001) & (true_props <= 0.01)
        med_conc_mask = (true_props > 0.01) & (true_props <= 0.1)
        high_conc_mask = true_props > 0.1
        
        weights = torch.ones_like(true_props)
        weights = torch.where(low_conc_mask, weights * 1.5, weights)
        weights = torch.where(med_conc_mask, weights * 1.2, weights)
        
        weighted_errors = cell_errors * weights
    
    # Mean across all cells in the batch
    prop_loss = weighted_errors.mean()
    
    # 2. Weighted reconstruction loss
    # Use reliability scores if provided, otherwise use coverage
    if reliability_scores is not None:
        weights = reliability_scores
    else:
        weights = coverage
    
    # Replace invalid marker values with reconstructed values
    safe_marker_values = torch.where(valid_mask, marker_values, reconstructed)
    
    # Weighted L1 loss
    recon_loss = torch.sum(
        valid_mask * weights * torch.abs(safe_marker_values - reconstructed)
    ) / (torch.sum(valid_mask * weights) + 1e-8)
    
    # 3. Total loss
    total_loss = alpha * prop_loss + beta * recon_loss
    
    # 4. Calculate metrics for monitoring
    with torch.no_grad():
        # Presence metrics
        presence_targets = (true_props > presence_threshold).float()
        presence_preds = (presence_probs > 0.5).float()
        
        # Basic metrics
        tp = torch.sum(presence_preds * presence_targets)
        fp = torch.sum(presence_preds * (1 - presence_targets))
        fn = torch.sum((1 - presence_preds) * presence_targets)
        tn = torch.sum((1 - presence_preds) * (1 - presence_targets))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Metrics by concentration range
        low_conc_mask = (true_props > 0) & (true_props <= 0.01)
        med_conc_mask = (true_props > 0.01) & (true_props <= 0.1)
        high_conc_mask = true_props > 0.1
        
        # Use absolute error for metrics
        abs_errors = torch.abs(pred_props - true_props)
        
        low_conc_error = torch.mean(torch.masked_select(abs_errors, low_conc_mask))
        med_conc_error = torch.mean(torch.masked_select(abs_errors, med_conc_mask))
        high_conc_error = torch.mean(torch.masked_select(abs_errors, high_conc_mask))
        
        # Metrics by coverage
        mean_coverage = coverage.mean(dim=1)
        low_cov_mask = mean_coverage < 10.0
        med_cov_mask = (mean_coverage >= 10.0) & (mean_coverage < 30.0)
        high_cov_mask = mean_coverage >= 30.0
        
        low_cov_error = torch.mean(abs_errors[low_cov_mask]) if low_cov_mask.any() else torch.tensor(0.0)
        med_cov_error = torch.mean(abs_errors[med_cov_mask]) if med_cov_mask.any() else torch.tensor(0.0)
        high_cov_error = torch.mean(abs_errors[high_cov_mask]) if high_cov_mask.any() else torch.tensor(0.0)
    
    # Return loss and metrics
    details = {
        'total_loss': total_loss.item(),
        'prop_loss': prop_loss.item(),
        'recon_loss': recon_loss.item(),
        'alpha_stats': {
            'mean': pred_props.mean().item(),
            'std': pred_props.std().item(),
            'max': pred_props.max().item(),
            'min': pred_props.min().item()
        },
        'presence_metrics': {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
        },
        'concentration_errors': {
            'low_conc': low_conc_error.item() if not torch.isnan(low_conc_error) else 0.0,
            'med_conc': med_conc_error.item() if not torch.isnan(med_conc_error) else 0.0,
            'high_conc': high_conc_error.item() if not torch.isnan(high_conc_error) else 0.0
        },
        'coverage_errors': {
            'low_cov': low_cov_error.item() if not torch.isnan(low_cov_error) else 0.0,
            'med_cov': med_cov_error.item() if not torch.isnan(med_cov_error) else 0.0,
            'high_cov': high_cov_error.item() if not torch.isnan(high_cov_error) else 0.0
        }
    }
    
    return total_loss, details