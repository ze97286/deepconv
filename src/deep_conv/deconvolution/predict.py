import torch
import numpy as np


def predict_with_post_processing(
    model, 
    marker_values, 
    coverage, 
    marker_to_cell_mapping,
    batch_size=256,
    min_coverage_threshold=5.0,
    min_signal_threshold=0.01
):
    """
    Complete inference pipeline with post-processing.
    
    Args:
        model: Trained deconvolution model
        marker_values: Input methylation values [samples, markers]
        coverage: Coverage values [samples, markers]
        marker_to_cell_mapping: Mapping from markers to cell types
        batch_size: Batch size for processing large datasets
        min_coverage_threshold: Minimum coverage to trust predictions
        min_signal_threshold: Minimum signal to keep prediction non-zero
        
    Returns:
        filtered_predictions: Post-processed cell type proportions
    """
    # Get raw model predictions
    raw_predictions = model.predict(marker_values, coverage, batch_size)
    
    # Calculate marker-level quality scores
    marker_quality = np.clip(coverage / 20.0, 0, 1)  # Coverage-based quality
    
    # Process each sample
    filtered = raw_predictions.copy()
    
    for i in range(len(filtered)):
        # Calculate average coverage for this sample
        avg_cov = coverage[i].mean()
        
        # Calculate cell type quality based on its markers
        cell_quality = np.zeros(model.num_celltypes)
        for cell_idx in range(model.num_celltypes):
            # Find markers corresponding to this cell type
            cell_markers = marker_to_cell_mapping == cell_idx
            if np.any(cell_markers):
                # Cell quality is average of its marker qualities
                cell_quality[cell_idx] = marker_quality[i, cell_markers].mean()
        
        # Handle zero or NaN quality
        cell_quality = np.nan_to_num(cell_quality)
        
        # Apply stricter filtering for low coverage
        if avg_cov < min_coverage_threshold:
            # For low coverage, use a more aggressive threshold
            threshold_factor = 1.5 - (avg_cov / min_coverage_threshold)
            dynamic_threshold = min_signal_threshold * threshold_factor
            
            # Filter based on quality and threshold
            quality_mask = cell_quality > 0.3
            filtered[i, ~quality_mask] = 0
            filtered[i, filtered[i] < dynamic_threshold] = 0
            
            # If everything got zeroed, keep just the top predictions
            if np.sum(filtered[i] > 0) <= 1:
                top_n = max(1, int(avg_cov / 2))  # Adaptive based on coverage
                top_idx = np.argsort(raw_predictions[i])[::-1][:top_n]
                
                # Zero everything except top N
                mask = np.zeros_like(filtered[i], dtype=bool)
                mask[top_idx] = True
                filtered[i, ~mask] = 0
        else:
            # For higher coverage, apply standard thresholding
            filtered[i, filtered[i] < min_signal_threshold] = 0
        
        # Re-normalize to sum to 1
        if filtered[i].sum() > 0:
            filtered[i] = filtered[i] / filtered[i].sum()
    
    return filtered