import torch

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """
    Compute focal loss for binary classification.
    
    Args:
        pred (torch.Tensor): Predicted probabilities, shape [B, C]
        target (torch.Tensor): Target labels (0 or 1), shape [B, C]
        gamma (float): Focusing parameter to reduce loss for easy examples
        alpha (float): Class balancing weight
        
    Returns:
        torch.Tensor: Focal loss value
    """
    bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
    modulator = torch.pow(1 - pred, gamma) * target + torch.pow(pred, gamma) * (1 - target)
    return (alpha * modulator * bce).mean()

def loss_fn(
    pred_props: torch.Tensor,
    true_props: torch.Tensor,
    presence_probs: torch.Tensor,
    coverage: torch.Tensor,
    x_nnls: torch.Tensor,
    dl_props: torch.Tensor,
    combination_weight: torch.Tensor,
    presence_threshold: float = 0.01,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    weight_penalty_lambda: float = 0.1,
    error_weighting: str = 'coverage',
):
    """
    Loss function with strong consistency between presence and proportion.
    
    Args:
        pred_props: [B, C] tensor of predicted proportions (after ensembling)
        true_props: [B, C] tensor of true proportions
        presence_probs: [B, C] tensor of presence probabilities
        coverage: [B, M] tensor of coverage values
        x_nnls: [B, C] tensor of NNLS predictions
        dl_props: [B, C] tensor of DeepConv predictions before ensembling
        combination_weight: [C] tensor of per-cell-type weights
        presence_threshold: Threshold for presence detection
        device: Device to run computations on
        weight_penalty_lambda: Weight for the weight penalty term
        error_weighting: How to weight the proportion errors ('coverage' or 'presence')
    """
    # Calculate basic proportion error
    errors = torch.abs(pred_props - true_props)  # Shape: [B, C]

    # Inverse frequency weighting for rare cell types
    avg_props = true_props.mean(dim=0)  # Shape: [C]
    cell_type_weights = 1.0 / (avg_props + 1e-6)  # Shape: [C], higher weight for rare cell types
    cell_type_weights = cell_type_weights / cell_type_weights.sum() * len(cell_type_weights)  # Normalize
    cell_type_weights = cell_type_weights.unsqueeze(0)  # Shape: [1, C]

    # Apply error weighting based on the chosen strategy
    if error_weighting == 'coverage':
        # Weight by average coverage per sample
        coverage_weights = coverage.mean(dim=1, keepdim=True)  # Shape: [B, 1]
        weighted_errors = errors * coverage_weights * cell_type_weights  # Broadcasting: [B, C] * [B, 1] * [1, C]
        loss_props = weighted_errors.mean()
    elif error_weighting == 'presence':
        # Weight by presence model confidence
        weighted_errors = errors * presence_probs * cell_type_weights  # Shape: [B, C] * [B, C] * [1, C]
        loss_props = weighted_errors.mean()
    else:
        raise ValueError(f"Unknown error_weighting option: {error_weighting}. Must be 'coverage' or 'presence'.")

    # Add concentration-specific loss components
    # Define critical concentration ranges with weights
    critical_ranges = [
        ((true_props >= 0.001) & (true_props < 0.005), 1.0),   # 0.1-0.5%
        ((true_props >= 0.005) & (true_props < 0.01), 2.0),    # 0.5-1% (highest weight)
        ((true_props >= 0.01) & (true_props < 0.05), 0.5),     # 1-5%
    ]
    
    # Problematic cell types and priority ranges
    # Tissues get special handling
    tissue_indices = [3, 4, 5, 10]  # Colon, Esophagus, Gastric, Small Intestine
    
    range_losses = []
    for mask, weight in critical_ranges:
        if mask.sum() > 0:
            # Create cell type-specific weights for this range
            cell_range_weights = torch.ones_like(true_props)
            
            # Higher weight for epithelial tissues
            for idx in tissue_indices:
                if idx < true_props.size(1):
                    cell_range_weights[:, idx] = 3.0  # Triple weight for epithelial tissues
            
            range_error = (torch.abs(pred_props - true_props) * mask.float() * cell_range_weights).sum() / (mask.sum() * cell_range_weights[mask].mean())
            range_losses.append(weight * range_error)
    
    critical_range_loss = sum(range_losses) if range_losses else torch.tensor(0.0, device=device)

    # Presence Loss (using focal loss)
    # Define true presence (1 if cell type's true proportion > threshold, 0 otherwise)
    true_presence = (true_props > presence_threshold).float()
    presence_loss = focal_loss(presence_probs, true_presence)

    # Add a STRONG presence consistency loss - heavily penalize predicting proportions for absent cell types
    presence_consistency_loss = torch.tensor(0.0, device=device)
    
    # For each cell type, check if it should be absent but has predicted proportions
    for idx in range(true_props.size(1)):
        # Identify samples where this cell type should be absent
        absent_mask = (true_presence[:, idx] == 0)
        
        if absent_mask.any():
            # Weight factor - highest for epithelial tissues
            weight_factor = 5.0 if idx in tissue_indices else 2.0
            
            # Penalize any non-zero prediction for absent cell types
            # Quadratic penalty makes large false positives much worse than small ones
            violation = pred_props[absent_mask, idx].pow(2)
            presence_consistency_loss += violation.mean() * weight_factor
    
    # Model Consistency Loss - penalize disagreement between presence model and proportions
    model_consistency_loss = torch.tensor(0.0, device=device)
    
    # Presence model should agree with proportion estimates
    model_presence_threshold = 0.3  # Threshold for presence model
    
    for idx in range(pred_props.size(1)):
        # Find inconsistencies - low presence prob but high proportion
        inconsistent_mask = (presence_probs[:, idx] < model_presence_threshold) & (pred_props[:, idx] > 0.05)
        
        if inconsistent_mask.any():
            # Weight factor - highest for epithelial tissues
            weight_factor = 5.0 if idx in tissue_indices else 2.0
            
            # Penalize inconsistency - higher penalty for larger proportions
            inconsistency = pred_props[inconsistent_mask, idx]
            model_consistency_loss += inconsistency.mean() * weight_factor

    # Sparsity Penalty - encourage zero predictions
    sparsity_penalty = torch.abs(pred_props).mean()

    # NNLS Regularization
    reg_loss = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        c = coverage.mean(dim=1)
        reg_loss = (c * (pred_props - x_nnls).pow(2).sum(dim=1)).mean()

    # Weight Penalty: Penalize when the worse method gets higher weight
    weight_penalty = torch.tensor(0.0, device=device)
    if x_nnls is not None and dl_props is not None:
        nnls_cell_errors = torch.abs(x_nnls - true_props).detach()
        dl_cell_errors = torch.abs(dl_props - true_props).detach()
        weights = torch.sigmoid(combination_weight).unsqueeze(0)
        weight_penalty = (weights * nnls_cell_errors + (1 - weights) * dl_cell_errors).mean()

    # Combine losses with fixed weights - heavier emphasis on consistency
    total_loss = (
        0.3 * loss_props +                     # General proportion accuracy
        0.1 * presence_loss +                  # Binary presence detection
        0.3 * presence_consistency_loss +      # Strong penalty for false positives
        0.2 * model_consistency_loss +         # Agreement with presence model
        0.05 * critical_range_loss +           # Focused on critical ranges
        0.02 * sparsity_penalty +              # Encourage sparsity
        0.03 * reg_loss +                      # NNLS regularization
        weight_penalty_lambda * weight_penalty # Adaptive ensemble weighting
    )

    # Details for logging
    details = {
        'loss_props': loss_props.item(),
        'presence_loss': presence_loss.item(),
        'presence_consistency_loss': presence_consistency_loss.item() if isinstance(presence_consistency_loss, torch.Tensor) else 0.0,
        'model_consistency_loss': model_consistency_loss.item() if isinstance(model_consistency_loss, torch.Tensor) else 0.0,
        'critical_range_loss': critical_range_loss.item() if isinstance(critical_range_loss, torch.Tensor) else 0.0,
        'sparsity_loss': sparsity_penalty.item(),
        'reg_loss': reg_loss.item(),
        'weight_penalty': weight_penalty.item()
    }

    return total_loss, details