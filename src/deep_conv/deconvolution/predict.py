import torch 
import numpy as np

def predict_with_consensus(
    model, 
    X, 
    coverage, 
    batch_size=256,
    device=None,
    low_snr_indices=[3, 4, 9, 11],
    concentration_thresholds={
        'very_low': {'threshold': 0.001, 'min_confidence': 0.7},
        'low': {'threshold': 0.01, 'min_confidence': 0.5},
        'medium': {'threshold': 0.05, 'min_confidence': 0.3},
        'high': {'threshold': 0.1, 'min_confidence': 0.2}
    }
):
    """
    Predicts cell-type proportions from marker-level input using a model, then applies
    post-processing logic based on dynamic thresholds. The post-processing modifies the 
    raw predictions using both concentration-based filtering (which checks the predicted 
    proportion) and confidence-based filtering (which checks the model's presence probability).

    Steps:
      1) Remove any WandB hooks that might log gradients (since we're doing inference).
      2) Split the data into batches for memory efficiency.
      3) Model forward pass:
         - For each batch, run the model to get:
             * props: The raw cell-type proportions (shape [B, C]).
             * presence_probs: Probability that each cell type is actually present (shape [B, C]).
      4) Post-processing (for each sample, each cell type):
         - Determine the predicted proportion (props[:, cell_idx]) and presence confidence (presence_probs[:, cell_idx]).
         - Compare the predicted proportion to thresholds that categorise it as 'very_low', 'low', 'medium', or 'high'.
         - For each category, require a minimum confidence. If presence_probs < (minimum confidence), 
           we scale down (by squared ratio) the predicted proportion to reflect uncertainty.
         - Optionally relax the minimum confidence if the cell type is known to be low SNR.
         - Zero out extremely small values (< 0.0005).
         - Re-normalise row-wise so the total proportion across cell types sums to 1.
      5) Combine results from all batches into a final NumPy array.

    Args:
        model (nn.Module): 
            The trained PyTorch model that outputs (proportions, presence_probs, etc.).
        X (np.ndarray or torch.Tensor): 
            Input marker methylation values. Shape should be [N, M] where N is #samples, M is #markers.
        coverage (np.ndarray or torch.Tensor): 
            Coverage values (same shape as X) used by the model.
        batch_size (int): 
            Number of samples per batch for inference (to limit memory usage).
        device (torch.device, optional): 
            The device to run the model on. If None, defaults to model's first parameter's device.
        low_snr_indices (List[int]): 
            Indices of cell types considered low-SNR. For these, we apply more lenient or scaled thresholds.
        concentration_thresholds (dict): 
            A mapping from category label to a dict with:
                'threshold': float => cutoff for predicted proportion,
                'min_confidence': float => minimum presence probability required for that category.
            Keys typically include 'very_low', 'low', 'medium', 'high', etc., but the user can customise.

    Returns:
        np.ndarray of shape [N, C] 
            The final predicted cell-type proportions per sample, after post-processing.
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

    # 4) Process the data in batches
    num_samples = X.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_X = X[start_idx:end_idx].to(device)
            batch_coverage = coverage[start_idx:end_idx].to(device)

            # (a) Forward pass through the model
            #     Model returns props, presence_probs, plus other outputs (which we ignore here).
            props, _, _, presence_probs, _ = model(batch_X, batch_coverage)

            # (b) Make a copy of the raw props to modify
            filtered_props = props.clone()

            # (c) For each cell type, apply dynamic thresholds for proportion & confidence
            for cell_idx in range(props.shape[1]):
                is_low_snr = (cell_idx in low_snr_indices)

                # For each concentration category
                for level_name, level_config in concentration_thresholds.items():
                    conc_threshold = level_config['threshold']
                    min_confidence = level_config['min_confidence']

                    # If cell is low-SNR, reduce the strictness of required confidence
                    if is_low_snr:
                        min_confidence *= 0.8

                    # Build a mask for samples that fit the current concentration range
                    if level_name == 'very_low':
                        # "very_low" => proportion <= 'low' threshold from the dictionary
                        conc_mask = props[:, cell_idx] <= concentration_thresholds['low']['threshold']
                    elif level_name == 'high':
                        # "high" => proportion > threshold for "high"
                        conc_mask = props[:, cell_idx] > conc_threshold
                    else:
                        # For e.g. "low" or "medium", find the next threshold above
                        # If there's no higher threshold, treat as infinite upper bound
                        next_level = next(
                            (v['threshold'] for k, v in concentration_thresholds.items()
                             if v['threshold'] > conc_threshold),
                            float('inf')
                        )
                        conc_mask = (
                            (props[:, cell_idx] > conc_threshold) &
                            (props[:, cell_idx] <= next_level)
                        )

                    # If any sample falls into this category, check presence confidence
                    if conc_mask.any():
                        confidence_ratio = presence_probs[:, cell_idx] / min_confidence

                        # Scale factor = 1 if confidence >= min_conf; else (confidence_ratio^2)
                        # i.e. we downscale the proportion more aggressively when confidence is lower
                        scaling_factor = torch.ones_like(confidence_ratio)
                        low_conf_mask = (confidence_ratio < 1.0)

                        scaling_factor[low_conf_mask] = torch.pow(confidence_ratio[low_conf_mask], 2)

                        # Apply the scaling factor only where conc_mask & low_conf_mask overlap
                        mask_to_apply = conc_mask & low_conf_mask
                        if mask_to_apply.any():
                            filtered_props[:, cell_idx][mask_to_apply] *= scaling_factor[mask_to_apply]

            # (d) Zero-out extremely small values, to remove noisy traces
            abs_threshold = 0.0005
            filtered_props[filtered_props < abs_threshold] = 0.0

            # (e) Re-normalise each sample so that proportions sum to 1
            row_sums = filtered_props.sum(dim=1, keepdim=True)
            valid_rows = (row_sums > 0).squeeze(-1)
            if valid_rows.any():
                filtered_props[valid_rows] /= row_sums[valid_rows]

            # (f) Move to CPU numpy and store
            predictions_list.append(filtered_props.cpu().numpy())

            # Optional GPU memory cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 5) Combine all batch results
    if len(predictions_list) == 0:
        # Edge case: empty input
        return np.zeros((num_samples, model.num_celltypes))
    
    # Return a single array of shape [N, C]
    return np.vstack(predictions_list)