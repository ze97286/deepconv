import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Simple focal loss without concentration weighting.
    
    Args:
        logits: Prediction logits
        targets: Ground truth labels
        alpha: Class balancing parameter (higher values focus more on positive class)
        gamma: Focusing parameter (higher values focus more on hard examples)
        
    Returns:
        Focal loss value
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    # Focal term
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_term = (1 - p_t) ** gamma
    
    # Alpha term for class imbalance
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # Combine
    loss = alpha_t * focal_term * ce_loss
    return loss.mean()


def train_binary_classifier(
    model: nn.Module,
    dataloaders: dict,
    model_path: str,
    target_cell_type: str,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    warmup_epochs: int = 5,
    weight_decay: float = 1e-4,
    patience: int = 20,  
    alpha: float = 0.25, 
    gamma: float = 2.0,  
    device: torch.device = None,
    fp16_training: bool = True,
    gradient_accumulation: int = 1,
    eval_metric: str = 'balanced_accuracy'
):
    """
    Train the simplified binary cell type classifier.
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
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_metrics = {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'loss': 0.0
        }
        
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
                    logits, _ = model(marker_values, coverage, target_markers_mask)
                    
                    # Calculate loss with focal loss
                    loss = focal_loss(logits, labels, alpha=alpha, gamma=gamma)
                    
                    # Scale for gradient accumulation
                    loss = loss / gradient_accumulation
            else:
                logits, _ = model(marker_values, coverage, target_markers_mask)
                
                # Calculate loss with focal loss
                loss = focal_loss(logits, labels, alpha=alpha, gamma=gamma)
                
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
                
                # Calculate metrics
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).float()
                
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
                'all_probs': []
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
                    logits, _ = model(marker_values, coverage, target_markers_mask)
                    
                    # Calculate loss
                    loss = focal_loss(logits, labels, alpha=alpha, gamma=gamma)
                    
                    # Calculate metrics
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities >= 0.5).float()
                    
                    val_losses.append(loss.item())
                    
                    # Store predictions and labels for ROC and PR curves
                    val_set_metrics['all_labels'].append(labels.cpu().numpy())
                    val_set_metrics['all_probs'].append(probabilities.cpu().numpy())
                    
                    # Update confusion matrix
                    val_set_metrics['tp'] += torch.sum((predictions == 1) & (labels == 1)).item()
                    val_set_metrics['fp'] += torch.sum((predictions == 1) & (labels == 0)).item()
                    val_set_metrics['tn'] += torch.sum((predictions == 0) & (labels == 0)).item()
                    val_set_metrics['fn'] += torch.sum((predictions == 0) & (labels == 1)).item()
            
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
            
            val_metrics[val_name] = val_set_metrics
            
            # Print validation metrics
            print(f"Validation ({val_name}): loss={val_set_metrics['loss']:.4f}, " +
                  f"precision={precision:.4f}, recall={recall:.4f}, specificity={specificity:.4f}, " +
                  f"f1={f1:.4f}, balanced_acc={balanced_accuracy:.4f}, " +
                  f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
            
            # Print confusion matrix
            print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
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
                'metric_name': eval_metric
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
                            (Only used if 'concentration' is available in batch)
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
            
            # Apply concentration-specific thresholds if provided and concentrations available
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
