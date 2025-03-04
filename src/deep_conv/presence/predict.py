import torch
import torch.nn as nn

def predict_presence(
    model: nn.Module,
    X: torch.Tensor,
    coverage: torch.Tensor,
    threshold: float = 0.5,
    batch_size: int = 256,
    device=None
):
    """
    Predict cell type presence from methylation data.
    
    Args:
        model: Trained presence detection model
        X: Methylation values [N, M]
        coverage: Coverage values [N, M]
        threshold: Probability threshold for presence
        batch_size: Batch size for prediction
        device: Device for computation
        
    Returns:
        presence_probs: Probabilities of presence [N, C]
        presence_binary: Binary presence/absence [N, C]
    """
    # Determine device
    if device is None:
        device = next(model.parameters()).device
    
    # Convert inputs to tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(coverage, torch.Tensor):
        coverage = torch.tensor(coverage, dtype=torch.float32)
    
    # Ensure batch dimension
    if len(X.shape) == 1:
        X = X.unsqueeze(0)
    if len(coverage.shape) == 1:
        coverage = coverage.unsqueeze(0)
    
    # Set model to evaluation mode
    model.eval()
    
    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    num_cell_types = model.num_celltypes
    
    # Prepare outputs
    all_probs = torch.zeros(num_samples, num_cell_types)
    
    # Process in batches
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # Get batch data
            batch_X = X[start_idx:end_idx].to(device)
            batch_coverage = coverage[start_idx:end_idx].to(device)
            
            # Forward pass
            presence_logits, _ = model(batch_X, batch_coverage)
            
            # Convert to probabilities
            presence_probs = torch.sigmoid(presence_logits)
            
            # Store results
            all_probs[start_idx:end_idx] = presence_probs.cpu()
    
    # Convert to binary predictions
    presence_binary = (all_probs > threshold).float()
    
    return all_probs, presence_binary