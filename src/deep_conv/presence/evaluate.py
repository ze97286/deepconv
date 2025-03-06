import torch
import torch.nn as nn 
import numpy as np
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_binary_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device=None,
    threshold=0.5,
    decision_thresholds=None,
    concentration_key='concentration'
):
    """
    Evaluate a binary classifier on a dataset.
    
    Args:
        model: Binary classifier model
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        threshold: Classification threshold
        decision_thresholds: Dictionary mapping concentration ranges to thresholds
                            e.g., {(0.1, 1.0): 0.7, (0.01, 0.1): 0.5, (0.0, 0.01): 0.3}
        concentration_key: Key in batch dictionary for concentration values
    
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Initialize metrics
    metrics = {
        'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
        'all_labels': [],
        'all_probs': [],
        'all_concs': []
    }
    
    # Evaluate
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            target_markers_mask = batch.get('target_markers_mask', None)
            if target_markers_mask is not None and target_markers_mask.dim() == 1:
                target_markers_mask = target_markers_mask.to(device)
            
            # Get labels
            labels = batch['label'].to(device).view(-1, 1)
            
            # Get concentrations if available
            concentrations = None
            if concentration_key in batch:
                concentrations = batch[concentration_key].cpu().numpy()
                metrics['all_concs'].append(concentrations)
            
            # Forward pass
            logits, _ = model(marker_values, coverage, target_markers_mask)
            probabilities = torch.sigmoid(logits)
            
            # Store predictions and labels
            metrics['all_labels'].append(labels.cpu().numpy())
            metrics['all_probs'].append(probabilities.cpu().numpy())
            
            # Apply concentration-specific thresholds if provided
            if decision_thresholds is not None and concentrations is not None:
                # Get threshold for each sample based on concentration
                sample_thresholds = []
                for conc in concentrations:
                    # Find matching threshold range
                    for (min_conc, max_conc), thresh in decision_thresholds.items():
                        if min_conc <= conc < max_conc:
                            sample_thresholds.append(thresh)
                            break
                    else:
                        # Use default if no range matches
                        sample_thresholds.append(threshold)
                
                # Convert to tensor
                thresh_tensor = torch.tensor(sample_thresholds, device=device).view(-1, 1)
                predictions = (probabilities >= thresh_tensor).float()
            else:
                # Use single threshold
                predictions = (probabilities >= threshold).float()
            
            # Update confusion matrix
            metrics['tp'] += torch.sum((predictions == 1) & (labels == 1)).item()
            metrics['fp'] += torch.sum((predictions == 1) & (labels == 0)).item()
            metrics['tn'] += torch.sum((predictions == 0) & (labels == 0)).item()
            metrics['fn'] += torch.sum((predictions == 0) & (labels == 1)).item()
    
    # Calculate derived metrics
    tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
    
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    
    # Concatenate arrays
    if metrics['all_labels']:
        all_labels = np.concatenate(metrics['all_labels']).flatten()
        all_probs = np.concatenate(metrics['all_probs']).flatten()
        
        # Calculate AUROC and AUPRC (if there are positive and negative examples)
        if len(np.unique(all_labels)) > 1:
            metrics['auroc'] = roc_auc_score(all_labels, all_probs)
            metrics['auprc'] = average_precision_score(all_labels, all_probs)
        else:
            metrics['auroc'] = 0.0
            metrics['auprc'] = metrics['precision']  # If only one class, AUPRC = precision
    
    # Aggregate concentrations if available
    if metrics['all_concs']:
        metrics['all_concs'] = np.concatenate(metrics['all_concs']).flatten()
    
    # Print results
    print(f"Evaluation Results:")
    print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}, Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    if 'auroc' in metrics:
        print(f"AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
    print(f"Confusion Matrix - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    return metrics


