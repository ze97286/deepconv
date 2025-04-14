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
    presence_threshold: float = 0.01,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    # Proportion Error
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

    # Combine losses with fixed weights
    total_loss = (
        0.5 * loss_props +
        0.1 * presence_loss +
        0.05 * sparsity_penalty +
        0.1 * reg_loss
    )

    # Details for logging
    details = {
        'loss_props': loss_props.item(),
        'presence_loss': presence_loss.item(),
        'sparsity_loss': sparsity_penalty.item(),
        'reg_loss': reg_loss.item()
    }

    return total_loss, details