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
    alpha: float = 0.7,        # Weight for proportion error
    beta: float = 0.3,         # Weight for reconstruction
    presence_threshold: float = 0.005
):
    """
    Coverage-weighted loss function for deconvolution
    """
    # 1. Proportion error (weighted MAE)
    cell_errors = torch.abs(pred_props - true_props)
    
    # Weight by proportion magnitude (higher weight for larger proportions)
    prop_weights = torch.clamp(true_props * 10, 0.1, 2.0)
    weighted_errors = cell_errors * prop_weights
    
    loss_props = weighted_errors.mean()
    
    # 2. Coverage-weighted reconstruction loss (NNLS-style)
    # Replace NaN marker values with zeros where invalid
    marker_values_safe = torch.where(valid_mask, marker_values, torch.zeros_like(marker_values))
    
    # Apply coverage weighting
    recon_loss = torch.sum(
        valid_mask * coverage * torch.abs(marker_values_safe - reconstructed)
    ) / (torch.sum(valid_mask * coverage) + 1e-8)
    
    # 3. Combine losses
    total_loss = alpha * loss_props + beta * recon_loss
    
    # 4. Calculate metrics for monitoring
    with torch.no_grad():
        # Presence metrics
        presence_targets = (true_props > presence_threshold).float()
        presence_preds = (presence_probs > 0.5).float()
        
        # Calculate basic metrics
        tp = torch.sum(presence_preds * presence_targets)
        fp = torch.sum(presence_preds * (1 - presence_targets))
        fn = torch.sum((1 - presence_preds) * presence_targets)
        tn = torch.sum((1 - presence_preds) * (1 - presence_targets))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Calculate errors in different proportion ranges
        low_prop_mask = (true_props > 0) & (true_props <= 0.01)
        med_prop_mask = (true_props > 0.01) & (true_props <= 0.1)
        high_prop_mask = true_props > 0.1
        
        low_prop_error = torch.mean(torch.masked_select(cell_errors, low_prop_mask))
        med_prop_error = torch.mean(torch.masked_select(cell_errors, med_prop_mask))
        high_prop_error = torch.mean(torch.masked_select(cell_errors, high_prop_mask))
        
        # Calculate metrics by coverage
        mean_coverage = coverage.mean(dim=1)
        low_cov_mask = mean_coverage < 10.0
        med_cov_mask = (mean_coverage >= 10.0) & (mean_coverage < 30.0)
        high_cov_mask = mean_coverage >= 30.0
        
        low_cov_error = torch.mean(cell_errors[low_cov_mask]) if low_cov_mask.any() else torch.tensor(0.0)
        med_cov_error = torch.mean(cell_errors[med_cov_mask]) if med_cov_mask.any() else torch.tensor(0.0)
        high_cov_error = torch.mean(cell_errors[high_cov_mask]) if high_cov_mask.any() else torch.tensor(0.0)
    
    # Return loss and monitoring metrics
    details = {
        'total_loss': total_loss.item(),
        'prop_loss': loss_props.item(),
        'recon_loss': recon_loss.item(),
        'presence_metrics': {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
        },
        'prop_errors': {
            'low_prop': low_prop_error.item() if not torch.isnan(low_prop_error) else 0.0,
            'med_prop': med_prop_error.item() if not torch.isnan(med_prop_error) else 0.0,
            'high_prop': high_prop_error.item() if not torch.isnan(high_prop_error) else 0.0,
        },
        'coverage_errors': {
            'low_cov': low_cov_error.item() if not torch.isnan(low_cov_error) else 0.0,
            'med_cov': med_cov_error.item() if not torch.isnan(med_cov_error) else 0.0,
            'high_cov': high_cov_error.item() if not torch.isnan(high_cov_error) else 0.0,
        }
    }
    
    return total_loss, details