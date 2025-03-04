import torch
import torch.nn as nn 
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)

def evaluate_presence_model(
    model: nn.Module,
    val_loaders: dict,
    presence_threshold: float = 0.0005,
    decision_threshold: float = 0.5,
    device: torch.device = None,
    cell_type_names=None
):
    """
    Evaluate presence detection model performance.
    
    Args:
        model: Trained presence detection model
        val_loaders: Dict of validation DataLoaders
        presence_threshold: Threshold for ground truth presence
        decision_threshold: Threshold for predicted presence
        device: Device for computation
        cell_type_names: List of cell type names
        
    Returns:
        results: Dict of evaluation results
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = {}
    
    for val_name, val_loader in val_loaders.items():
        print(f"\nEvaluating on {val_name}:")
        
        # Get all predictions and ground truth
        all_preds_probs = []
        all_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                marker_values = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                true_props = batch['y'].to(device)
                
                # Forward pass
                presence_logits, _ = model(marker_values, coverage)
                presence_probs = torch.sigmoid(presence_logits)
                
                # Convert to numpy
                presence_probs = presence_probs.cpu().numpy()
                true_props = true_props.cpu().numpy()
                
                all_preds_probs.append(presence_probs)
                all_true.append(true_props)
        
        # Concatenate results
        all_preds_probs = np.vstack(all_preds_probs)
        all_true = np.vstack(all_true)
        
        # Convert to binary presence
        all_true_binary = (all_true > presence_threshold).astype(float)
        all_preds_binary = (all_preds_probs > decision_threshold).astype(float)
        
        # Compute overall metrics
        true_flat = all_true_binary.flatten()
        pred_flat = all_preds_binary.flatten()
        prob_flat = all_preds_probs.flatten()
        
        # Overall metrics
        tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(true_flat, prob_flat)
        roc_auc = auc(fpr, tpr)
        
        # PR curve and AUC
        pr_precision, pr_recall, _ = precision_recall_curve(true_flat, prob_flat)
        pr_auc = auc(pr_recall, pr_precision)
        
        # Per-cell type metrics
        per_cell_metrics = []
        
        for i in range(all_true_binary.shape[1]):
            true_i = all_true_binary[:, i]
            pred_i = all_preds_binary[:, i]
            prob_i = all_preds_probs[:, i]
            
            # Check if this cell type is ever present or ever absent in this dataset
            cell_present = np.any(true_i > 0)
            cell_absent = np.any(true_i == 0)
            
            cell_name = cell_type_names[i] if cell_type_names else f"Cell type {i}"
            
            # Only calculate complete metrics if cell type is both present and absent
            if cell_present and cell_absent:
                tn_i, fp_i, fn_i, tp_i = confusion_matrix(true_i, pred_i).ravel()
                precision_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
                recall_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
                specificity_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
                f1_i = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0
                
                # ROC and PR curves
                fpr_i, tpr_i, _ = roc_curve(true_i, prob_i)
                roc_auc_i = auc(fpr_i, tpr_i)
                
                pr_precision_i, pr_recall_i, _ = precision_recall_curve(true_i, prob_i)
                pr_auc_i = auc(pr_recall_i, pr_precision_i)
                
                status = "normal"
            # Handle always-present case (can only measure false negatives)
            elif cell_present and not cell_absent:
                tp_i = np.sum((pred_i > 0) & (true_i > 0))
                fn_i = np.sum((pred_i == 0) & (true_i > 0))
                tn_i = fp_i = 0
                
                recall_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
                precision_i = 1.0 if tp_i > 0 else 0  # No false positives possible
                specificity_i = float('nan')  # Not applicable
                f1_i = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0
                
                roc_auc_i = float('nan')  # Not applicable for all-positive
                pr_auc_i = float('nan')
                
                status = "always_present"
            # Handle always-absent case (can only measure false positives)
            elif not cell_present and cell_absent:
                tn_i = np.sum((pred_i == 0) & (true_i == 0))
                fp_i = np.sum((pred_i > 0) & (true_i == 0))
                tp_i = fn_i = 0
                
                specificity_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
                precision_i = 0  # No true positives
                recall_i = float('nan')  # Not applicable
                f1_i = 0  # No true positives
                
                roc_auc_i = float('nan')  # Not applicable for all-negative
                pr_auc_i = float('nan')
                
                status = "always_absent"
            else:
                # Should never happen - no samples for this cell type
                print(f"Warning: No samples for {cell_name}")
                continue
            
            per_cell_metrics.append({
                'name': cell_name,
                'precision': precision_i,
                'recall': recall_i,
                'specificity': specificity_i,
                'f1': f1_i,
                'roc_auc': roc_auc_i if not np.isnan(roc_auc_i) else None,
                'pr_auc': pr_auc_i if not np.isnan(pr_auc_i) else None,
                'true_positives': int(tp_i),
                'false_positives': int(fp_i),
                'true_negatives': int(tn_i),
                'false_negatives': int(fn_i),
                'status': status
            })
        
        # Calculate meaningful averages only for applicable cell types
        valid_cells_metrics = [m for m in per_cell_metrics if m['status'] == "normal"]
        present_cells_metrics = [m for m in per_cell_metrics if m['status'] in ["normal", "always_present"]]
        absent_cells_metrics = [m for m in per_cell_metrics if m['status'] in ["normal", "always_absent"]]
        
        # Calculate averages
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in valid_cells_metrics]) if valid_cells_metrics else np.nan,
            'recall': np.mean([m['recall'] for m in present_cells_metrics]) if present_cells_metrics else np.nan,
            'specificity': np.mean([m['specificity'] for m in absent_cells_metrics]) if absent_cells_metrics else np.nan,
            'f1': np.mean([m['f1'] for m in valid_cells_metrics]) if valid_cells_metrics else np.nan,
        }
        
        # Store results
        results[val_name] = {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            },
            'cell_type_averages': avg_metrics,
            'per_cell_type': per_cell_metrics
        }
        
        # Print summary
        print(f"Overall dataset metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall/Sensitivity: {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  PR AUC: {pr_auc:.4f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        # Print cell type average metrics
        print("\nCell type average metrics:")
        print(f"  Precision: {avg_metrics['precision']:.4f}")
        print(f"  Recall: {avg_metrics['recall']:.4f}")
        print(f"  Specificity: {avg_metrics['specificity']:.4f}")
        print(f"  F1: {avg_metrics['f1']:.4f}")
        
        # Print per-cell type metrics for important cell types
        print("\nMetrics for notable cell types:")
        for cell_metric in per_cell_metrics:
            # Skip cell types that are always absent in specialized datasets
            if cell_metric['status'] == "always_absent" and val_name in ['cd4', 'cd8', 'oac']:
                continue
                
            print(f"  {cell_metric['name']} ({cell_metric['status']}):")
            
            # Print applicable metrics based on status
            if cell_metric['status'] == "normal":
                print(f"    Precision: {cell_metric['precision']:.4f}")
                print(f"    Recall: {cell_metric['recall']:.4f}")
                print(f"    Specificity: {cell_metric['specificity']:.4f}")
                print(f"    F1: {cell_metric['f1']:.4f}")
            elif cell_metric['status'] == "always_present":
                print(f"    Always present - Recall: {cell_metric['recall']:.4f}")
                print(f"    TP: {cell_metric['true_positives']}, FN: {cell_metric['false_negatives']}")
            elif cell_metric['status'] == "always_absent":
                print(f"    Always absent - Specificity: {cell_metric['specificity']:.4f}")
                print(f"    TN: {cell_metric['true_negatives']}, FP: {cell_metric['false_positives']}")
    
    return results