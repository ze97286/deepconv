import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


def concentration_weighted_focal_loss(logits, targets, concentration, gamma=2.0, alpha=0.25,
                                      critical_range=(0.01, 0.03), critical_weight=3.0):
    """
    Focal loss with additional weighting based on sample concentration.
    
    Args:
        logits: [B, 1] Logits from the model
        targets: [B, 1] Binary ground truth
        concentration: [B, 1] Estimated concentration for each sample
        gamma: Focusing parameter for focal loss
        alpha: Class balancing parameter
        critical_range: Tuple (min_conc, max_conc) defining the critical concentration range
        critical_weight: Weight multiplier for samples in the critical range
        
    Returns:
        Weighted focal loss
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    # Focal loss term
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_term = (1 - p_t) ** gamma
    
    # Alpha weighting for class imbalance
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # Concentration-based weighting
    conc_weights = torch.ones_like(logits)
    critical_mask = (concentration >= critical_range[0]) & (concentration <= critical_range[1])
    conc_weights[critical_mask] = critical_weight
    
    # Additional weighting for very low concentrations (below critical range)
    very_low_mask = (concentration < critical_range[0]) & (concentration > 0)
    conc_weights[very_low_mask] = 2.0
    
    # Combine all weights
    weighted_focal_loss = alpha_t * focal_term * ce_loss * conc_weights
    
    return weighted_focal_loss.mean()


def train_binary_classifier(
    model: nn.Module,
    dataloaders: dict,
    model_path: str,
    target_cell_type: str,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    warmup_epochs: int = 5,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: torch.device = None,
    fp16_training: bool = True,
    gradient_accumulation: int = 1,
    concentration_balance: bool = True,
    curriculum_learning: bool = True,
    eval_metric: str = 'balanced_accuracy'
):
    """
    Enhanced training function for the low-concentration cell type detector.
    
    Args:
        model: Enhanced cell type detector model
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        model_path: Path to save model checkpoints
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        warmup_epochs: Number of warmup epochs for learning rate
        weight_decay: L2 regularization weight
        patience: Early stopping patience
        device: Device to run on (GPU/CPU)
        fp16_training: Whether to use mixed precision training
        gradient_accumulation: Number of batches to accumulate gradients
        concentration_balance: Whether to balance samples across concentration ranges
        curriculum_learning: Whether to use curriculum learning (start with easy samples)
        eval_metric: Metric for model selection ('balanced_accuracy', 'f1', 'auc')
        
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Set up mixed precision training
    scaler = torch.cuda.amp.GradScaler() if fp16_training and torch.cuda.is_available() else None
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Main scheduler for after warmup
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience//2, verbose=True
    )
    
    # Create directory for saving models
    os.makedirs(model_path, exist_ok=True)
    
    # Initialize early stopping variables
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Create concentration thresholds for evaluation
    concentration_thresholds = {
        (0.0, 0.01): 0.25,   # Very low concentration: much lower threshold
        (0.01, 0.02): 0.35,  # Low concentration: lower threshold
        (0.02, 0.05): 0.45,  # Medium concentration: near standard threshold
        (0.05, 1.0): 0.5     # High concentration: standard threshold
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_metrics = {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'loss': 0.0
        }
        
        # Implement curriculum learning if enabled
        if curriculum_learning:
            # In early epochs, focus on higher concentration samples (easier)
            # In later epochs, include more low-concentration samples (harder)
            curr_progress = min(1.0, epoch / (num_epochs * 0.5))  # 0 to 1 over first half of training
            
            # Adjust critical range for loss function based on curriculum progress
            # Start with higher concentration range, then gradually lower it
            critical_min = max(0.01, 0.05 - 0.04 * curr_progress)
            critical_max = max(0.03, 0.1 - 0.07 * curr_progress)
            critical_range = (critical_min, critical_max)
        else:
            # Fixed critical range for all epochs
            critical_range = (0.01, 0.03)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Critical range: {critical_range}")
        
        # Training
        for batch_idx, batch in enumerate(tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{num_epochs}")):
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            target_markers_mask = batch.get('target_markers_mask', None)
            if target_markers_mask is not None and target_markers_mask.dim() == 1:
                target_markers_mask = target_markers_mask.to(device)
            
            labels = batch['label'].to(device).view(-1, 1)
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, concentration, _ = model(marker_values, coverage, target_markers_mask)
                    
                    # Calculate loss with concentration weighting
                    loss = concentration_weighted_focal_loss(
                        logits, labels, concentration, 
                        critical_range=critical_range
                    )
                    
                    # Scale for gradient accumulation
                    loss = loss / gradient_accumulation
            else:
                logits, concentration, _ = model(marker_values, coverage, target_markers_mask)
                
                # Calculate loss with concentration weighting
                loss = concentration_weighted_focal_loss(
                    logits, labels, concentration,
                    critical_range=critical_range
                )
                
                # Scale for gradient accumulation
                loss = loss / gradient_accumulation
            
            # Backward pass with mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation == 0 or batch_idx == len(dataloaders['train']) - 1:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation == 0 or batch_idx == len(dataloaders['train']) - 1:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Track metrics
            with torch.no_grad():
                train_losses.append(loss.item() * gradient_accumulation)
                
                # Calculate adaptive threshold based on concentration
                probabilities = torch.sigmoid(logits)
                predictions = model.predict_with_adaptive_threshold(
                    marker_values, coverage, target_markers_mask
                )[0]
                
                # Update confusion matrix
                train_metrics['tp'] += torch.sum((predictions == 1) & (labels == 1)).item()
                train_metrics['fp'] += torch.sum((predictions == 1) & (labels == 0)).item()
                train_metrics['tn'] += torch.sum((predictions == 0) & (labels == 0)).item()
                train_metrics['fn'] += torch.sum((predictions == 0) & (labels == 1)).item()
        
        # Update learning rate for warmup
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        
        # Calculate training metrics
        train_metrics['loss'] = np.mean(train_losses)
        tp, fp, tn, fn = train_metrics['tp'], train_metrics['fp'], train_metrics['tn'], train_metrics['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        balanced_accuracy = (recall + specificity) / 2
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train: loss={train_metrics['loss']:.4f}, " +
              f"precision={precision:.4f}, recall={recall:.4f}, specificity={specificity:.4f}, " +
              f"f1={f1:.4f}, balanced_acc={balanced_accuracy:.4f}")
        
        # Validation
        model.eval()
        val_metrics = {}
        
        for val_name, val_loader in dataloaders['val'].items():
            val_set_metrics = {
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                'loss': 0.0,
                'all_labels': [],
                'all_probs': [],
                'all_conc': [],
                'per_conc_metrics': {}
            }
            
            val_losses = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating {val_name}"):
                    marker_values = batch['X'].to(device)
                    coverage = batch['coverage'].to(device)
                    target_markers_mask = batch.get('target_markers_mask', None)
                    if target_markers_mask is not None and target_markers_mask.dim() == 1:
                        target_markers_mask = target_markers_mask.to(device)
                    
                    labels = batch['label'].to(device).view(-1, 1)
                    
                    # Forward pass
                    logits, concentration, _ = model(marker_values, coverage, target_markers_mask)
                    
                    # Calculate loss
                    loss = concentration_weighted_focal_loss(
                        logits, labels, concentration,
                        critical_range=critical_range
                    )
                    
                    # Use adaptive thresholding for predictions
                    predictions, probabilities, _ = model.predict_with_adaptive_threshold(
                        marker_values, coverage, target_markers_mask
                    )
                    
                    val_losses.append(loss.item())
                    
                    # Store predictions and labels for ROC and PR curves
                    val_set_metrics['all_labels'].append(labels.cpu().numpy())
                    val_set_metrics['all_probs'].append(probabilities.cpu().numpy())
                    val_set_metrics['all_conc'].append(concentration.cpu().numpy())
                    
                    # Update confusion matrix
                    val_set_metrics['tp'] += torch.sum((predictions == 1) & (labels == 1)).item()
                    val_set_metrics['fp'] += torch.sum((predictions == 1) & (labels == 0)).item()
                    val_set_metrics['tn'] += torch.sum((predictions == 0) & (labels == 0)).item()
                    val_set_metrics['fn'] += torch.sum((predictions == 0) & (labels == 1)).item()
                    
                    # Per-concentration metrics
                    for conc_range, threshold in concentration_thresholds.items():
                        min_conc, max_conc = conc_range
                        conc_mask = (concentration >= min_conc) & (concentration < max_conc)
                        
                        if conc_mask.any():
                            # Calculate metrics for this concentration range
                            range_labels = labels[conc_mask]
                            range_probs = probabilities[conc_mask]
                            range_preds = (range_probs >= threshold).float()
                            
                            # Confusion matrix
                            tp = torch.sum((range_preds == 1) & (range_labels == 1)).item()
                            fp = torch.sum((range_preds == 1) & (range_labels == 0)).item()
                            tn = torch.sum((range_preds == 0) & (range_labels == 0)).item()
                            fn = torch.sum((range_preds == 0) & (range_labels == 1)).item()
                            
                            # Metrics
                            conc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            conc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            conc_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                            conc_f1 = 2 * conc_precision * conc_recall / (conc_precision + conc_recall) if (conc_precision + conc_recall) > 0 else 0.0
                            
                            # Store metrics
                            conc_key = f"{min_conc:.4f}-{max_conc:.4f}"
                            if conc_key not in val_set_metrics['per_conc_metrics']:
                                val_set_metrics['per_conc_metrics'][conc_key] = {
                                    'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                                    'count': 0
                                }
                            
                            val_set_metrics['per_conc_metrics'][conc_key]['tp'] += tp
                            val_set_metrics['per_conc_metrics'][conc_key]['fp'] += fp
                            val_set_metrics['per_conc_metrics'][conc_key]['tn'] += tn
                            val_set_metrics['per_conc_metrics'][conc_key]['fn'] += fn
                            val_set_metrics['per_conc_metrics'][conc_key]['count'] += conc_mask.sum().item()
            
            # Calculate validation metrics
            val_set_metrics['loss'] = np.mean(val_losses)
            tp, fp, tn, fn = val_set_metrics['tp'], val_set_metrics['fp'], val_set_metrics['tn'], val_set_metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            balanced_accuracy = (recall + specificity) / 2
            
            # Concat all labels and probabilities
            all_labels = np.concatenate(val_set_metrics['all_labels']).flatten()
            all_probs = np.concatenate(val_set_metrics['all_probs']).flatten()
            all_conc = np.concatenate(val_set_metrics['all_conc']).flatten()
            
            # Calculate AUROC and AUPRC (if there are positive and negative examples)
            if len(np.unique(all_labels)) > 1:
                auroc = roc_auc_score(all_labels, all_probs)
                auprc = average_precision_score(all_labels, all_probs)
            else:
                auroc = 0.0
                auprc = precision  # If only one class, AUPRC = precision
            
            # Store metrics
            val_set_metrics.update({
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'balanced_accuracy': balanced_accuracy,
                'auroc': auroc,
                'auprc': auprc
            })
            
            # Calculate per-concentration metrics
            for conc_key, conc_metrics in val_set_metrics['per_conc_metrics'].items():
                tp, fp, tn, fn = conc_metrics['tp'], conc_metrics['fp'], conc_metrics['tn'], conc_metrics['fn']
                conc_metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                conc_metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                conc_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                conc_metrics['f1'] = 2 * conc_metrics['precision'] * conc_metrics['recall'] / (conc_metrics['precision'] + conc_metrics['recall']) if (conc_metrics['precision'] + conc_metrics['recall']) > 0 else 0.0
                conc_metrics['balanced_accuracy'] = (conc_metrics['recall'] + conc_metrics['specificity']) / 2
            
            val_metrics[val_name] = val_set_metrics
            
            # Print validation metrics
            print(f"Validation ({val_name}): loss={val_set_metrics['loss']:.4f}, " +
                  f"precision={precision:.4f}, recall={recall:.4f}, specificity={specificity:.4f}, " +
                  f"f1={f1:.4f}, balanced_acc={balanced_accuracy:.4f}, " +
                  f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
            
            # Print confusion matrix
            print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            
            # Print per-concentration metrics
            print(f"Per-concentration metrics for {val_name}:")
            for conc_key, conc_metrics in sorted(val_set_metrics['per_conc_metrics'].items()):
                print(f"  Concentration {conc_key}: " +
                      f"count={conc_metrics['count']}, " +
                      f"recall={conc_metrics['recall']:.4f}, " +
                      f"specificity={conc_metrics['specificity']:.4f}, " +
                      f"f1={conc_metrics['f1']:.4f}")
        
        # Calculate average metric for validation sets
        if eval_metric == 'balanced_accuracy':
            avg_metric = np.mean([m['balanced_accuracy'] for m in val_metrics.values()])
        elif eval_metric == 'f1':
            avg_metric = np.mean([m['f1'] for m in val_metrics.values()])
        elif eval_metric == 'auroc':
            avg_metric = np.mean([m['auroc'] for m in val_metrics.values()])
        else:
            raise ValueError(f"Unknown evaluation metric: {eval_metric}")
        
        # Update learning rate scheduler (after warmup)
        if epoch >= warmup_epochs:
            main_scheduler.step(avg_metric)
        
        # Check for improvement
        if avg_metric > best_metric:
            best_metric = avg_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'metric_name': eval_metric,
                'concentration_thresholds': concentration_thresholds
            }
            torch.save(checkpoint, os.path.join(model_path, f'best_{target_cell_type}_model.pt'))
            
            print(f"New best model saved! {eval_metric}={best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_path, f'best_{target_cell_type}_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with {eval_metric}={checkpoint['best_metric']:.4f}")
    
    return model

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


