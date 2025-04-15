import torch

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
    modulator = torch.pow(1 - pred, gamma) * target + torch.pow(pred, gamma) * (1 - target)
    return (alpha * modulator * bce).mean()

def loss_fn(
    pred_props: torch.Tensor,
    true_props: torch.Tensor,
    presence_probs: torch.Tensor,
    coverage: torch.Tensor,
    x_nnls: torch.Tensor,
    dl_props: torch.Tensor,  # DeepConv predictions before ensembling
    combination_weight: torch.Tensor,  # Per-cell-type weights
    presence_threshold: float = 0.01,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    weight_penalty_lambda: float = 0.1,  # Weight for the weight penalty term
):
    # Proportion Error (on final ensembled predictions)
    errors = torch.abs(pred_props - true_props)
    coverage_weights = coverage.mean(dim=1, keepdim=True)
    loss_props = (errors * coverage_weights).mean()

    # Presence Loss (using focal loss)
    presence_targets = (true_props > presence_threshold).float()
    presence_loss = focal_loss(presence_probs, presence_targets)

    # Sparsity Penalty
    sparsity_penalty = torch.abs(pred_props).mean()

    # NNLS Regularization
    reg_loss = torch.tensor(0.0, device=device)
    if x_nnls is not None:
        c = coverage.mean(dim=1)
        reg_loss = (c * (pred_props - x_nnls).pow(2).sum(dim=1)).mean()

    # Weight Penalty: Penalize when the worse method gets higher weight
    weight_penalty = torch.tensor(0.0, device=device)
    if x_nnls is not None and dl_props is not None:
        # Compute per-cell-type errors for NNLS and DeepConv
        nnls_cell_errors = torch.abs(x_nnls - true_props).detach()  # Shape: [B, C], detach to not train NNLS
        dl_cell_errors = torch.abs(dl_props - true_props).detach()  # Shape: [B, C], detach to not train DeepConv
        
        # Compute weights
        weights = torch.sigmoid(combination_weight)  # Shape: [C], values in [0, 1]
        weights = weights.unsqueeze(0)  # Shape: [1, C] for broadcasting
        
        # Penalize: weight * error_of_nnls + (1 - weight) * error_of_dl
        # If NNLS has higher error, we want weight to be low (favor DeepConv), and vice versa
        weight_penalty = (weights * nnls_cell_errors + (1 - weights) * dl_cell_errors).mean()

    # Combine losses with fixed weights
    total_loss = (
        0.5 * loss_props +
        0.1 * presence_loss +
        0.05 * sparsity_penalty +
        0.1 * reg_loss +
        weight_penalty_lambda * weight_penalty  # Add the weight penalty term
    )

    # Details for logging
    details = {
        'loss_props': loss_props.item(),
        'presence_loss': presence_loss.item(),
        'sparsity_loss': sparsity_penalty.item(),
        'reg_loss': reg_loss.item(),
        'weight_penalty': weight_penalty.item()  # Log the weight penalty
    }

    return total_loss, details