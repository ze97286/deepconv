import torch
import numpy as np

def predict_with_consensus(
    model,
    X,
    coverage,
    batch_size=256,
    device=None,
):
    """
    Makes predictions using the deconvolution model in evaluation mode.
    
    This function:
    1) Handles any W&B hooks that might be present
    2) Converts inputs to the right format and device
    3) Processes data in batches to avoid memory issues
    4) Returns cell type proportions as numpy array
    
    Args:
        model: The CellTypeDeconvolutionModel
        X: Marker methylation values [N, M]
        coverage: Coverage values [N, M]
        batch_size: Batch size for processing
        device: Device to run inference on (defaults to model's device)
        
    Returns:
        numpy.ndarray: Cell type proportions [N, C]
    """
    # 1) Remove any W&B hooks before prediction, to avoid logging or gradient issues during inference
    wandb_hooks = []
    if hasattr(model, '_forward_hooks'):
        wandb_hooks = [
            (k, v) for k, v in model._forward_hooks.items()
            if 'wandb' in str(v)
        ]
        for hook_id, _ in wandb_hooks:
            model._forward_hooks.pop(hook_id)
    
    # 2) Decide which device to use (CPU/GPU)
    if device is None:
        device = next(model.parameters()).device
    
    # 3) Convert inputs (X, coverage) to Torch tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(coverage, torch.Tensor):
        coverage = torch.tensor(coverage, dtype=torch.float32)
    
    # Ensure both inputs have a batch dimension
    if len(X.shape) == 1:
        X = X.unsqueeze(0)
    if len(coverage.shape) == 1:
        coverage = coverage.unsqueeze(0)
    
    model.eval()
    predictions_list = []
    presence_probs_list = []
    
    # 4) Process the data in batches
    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_X = X[start_idx:end_idx].to(device)
            batch_coverage = coverage[start_idx:end_idx].to(device)
            
            # Forward pass through the model - unpack the five return values
            props, reconstructed, valid_mask, presence_probs, presence_logits = model(batch_X, batch_coverage)
            
            # Move to CPU numpy and store
            predictions_list.append(props.cpu().numpy())
            presence_probs_list.append(presence_probs.cpu().numpy())
            
            # Optional GPU memory cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # 5) Combine all batch results
    if len(predictions_list) == 0:
        # Edge case: empty input
        return np.zeros((num_samples, model.num_celltypes))
    
    # Return the cell type proportions
    return np.vstack(predictions_list)


def predict_with_details(
    model,
    X,
    coverage,
    batch_size=256,
    device=None,
):
    """
    Extended prediction function that returns additional details
    beyond just the cell type proportions.
    
    This function returns:
    - Cell type proportions
    - Presence probabilities 
    - Reconstructed marker values
    
    Args:
        model: The CellTypeDeconvolutionModel
        X: Marker methylation values [N, M]
        coverage: Coverage values [N, M]
        batch_size: Batch size for processing
        device: Device to run inference on (defaults to model's device)
        
    Returns:
        tuple: (cell_props, presence_probs, reconstructed_markers)
    """
    # 1) Remove any W&B hooks before prediction, to avoid logging or gradient issues during inference
    wandb_hooks = []
    if hasattr(model, '_forward_hooks'):
        wandb_hooks = [
            (k, v) for k, v in model._forward_hooks.items()
            if 'wandb' in str(v)
        ]
        for hook_id, _ in wandb_hooks:
            model._forward_hooks.pop(hook_id)
    
    # 2) Decide which device to use (CPU/GPU)
    if device is None:
        device = next(model.parameters()).device
    
    # 3) Convert inputs (X, coverage) to Torch tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(coverage, torch.Tensor):
        coverage = torch.tensor(coverage, dtype=torch.float32)
    
    # Ensure both inputs have a batch dimension
    if len(X.shape) == 1:
        X = X.unsqueeze(0)
    if len(coverage.shape) == 1:
        coverage = coverage.unsqueeze(0)
    
    model.eval()
    predictions_list = []
    presence_probs_list = []
    reconstructed_list = []
    
    # 4) Process the data in batches
    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_X = X[start_idx:end_idx].to(device)
            batch_coverage = coverage[start_idx:end_idx].to(device)
            
            # Forward pass through the model - unpack the five return values
            props, reconstructed, valid_mask, presence_probs, presence_logits = model(batch_X, batch_coverage)
            
            # Move to CPU numpy and store
            predictions_list.append(props.cpu().numpy())
            presence_probs_list.append(presence_probs.cpu().numpy())
            reconstructed_list.append(reconstructed.cpu().numpy())
            
            # Optional GPU memory cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # 5) Combine all batch results
    if len(predictions_list) == 0:
        # Edge case: empty input
        return (np.zeros((num_samples, model.num_celltypes)), 
                np.zeros((num_samples, model.num_celltypes)),
                np.zeros((num_samples, model.num_markers)))
    
    # Return the combined results
    return (
        np.vstack(predictions_list),           # Cell type proportions
        np.vstack(presence_probs_list),        # Presence probabilities
        np.vstack(reconstructed_list)          # Reconstructed marker values
    )