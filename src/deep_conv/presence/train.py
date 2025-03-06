import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

def train_binary_classifier(
    model: nn.Module,
    dataloaders: dict,
    model_path: str,
    target_cell_type: str,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weight: float = None,  # Positive class weight (for imbalance)
    patience: int = 10,
    device: torch.device = None,
    fp16_training: bool = True,  # Use mixed precision
    gradient_accumulation: int = 1,  # Number of batches to accumulate
    eval_metric: str = 'balanced_accuracy'  # 'balanced_accuracy', 'f1', 'auroc'
):
    """
    Train a binary classifier for cell type detection.
    
    Args:
        model: Binary classifier model
        dataloaders: Dictionary containing 'train' and 'val' dataloaders
        model_path: Path to save model checkpoints
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        class_weight: Weight for positive class (None = auto-calculate)
        patience: Early stopping patience
        device: Training device (GPU/CPU)
        fp16_training: Whether to use mixed precision training
        gradient_accumulation: Number of batches to accumulate gradients
        eval_metric: Metric to use for model selection
    
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
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience//2, verbose=True
    )
    
    # Create directory for saving models
    os.makedirs(model_path, exist_ok=True)
    
    # Automatically calculate class weight if not provided
    if class_weight is None:
        pos_count = 0
        total_count = 0
        
        for batch in dataloaders['train']:
            labels = batch['label']
            pos_count += torch.sum(labels).item()
            total_count += len(labels)
        
        pos_ratio = pos_count / total_count
        class_weight = (1 - pos_ratio) / pos_ratio
        print(f"Calculated positive class weight: {class_weight:.4f} (ratio: {pos_ratio:.4f})")
    
    # Create loss function with class weights
    weights = torch.tensor([1.0, class_weight], device=device)
    
    def weighted_bce_loss(logits, targets):
        per_sample_weights = torch.ones_like(targets)
        per_sample_weights[targets == 1] = weights[1]
        return F.binary_cross_entropy_with_logits(
            logits, targets, weight=per_sample_weights, reduction='mean'
        )
    
    # Tracking variables
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
            labels = batch['label'].to(device).view(-1, 1)
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, _ = model(marker_values, coverage)
                    loss = weighted_bce_loss(logits, labels)
                    loss = loss / gradient_accumulation  # Scale for gradient accumulation
            else:
                logits, _ = model(marker_values, coverage)
                loss = weighted_bce_loss(logits, labels)
                loss = loss / gradient_accumulation  # Scale for gradient accumulation
            
            # Backward pass with mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation == 0 or batch_idx == len(dataloaders['train']) - 1:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation == 0 or batch_idx == len(dataloaders['train']) - 1:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Track metrics
            with torch.no_grad():
                train_losses.append(loss.item() * gradient_accumulation)
                
                # Calculate confusion matrix
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).float()
                
                train_metrics['tp'] += torch.sum((predictions == 1) & (labels == 1)).item()
                train_metrics['fp'] += torch.sum((predictions == 1) & (labels == 0)).item()
                train_metrics['tn'] += torch.sum((predictions == 0) & (labels == 0)).item()
                train_metrics['fn'] += torch.sum((predictions == 0) & (labels == 1)).item()
        
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
                    labels = batch['label'].to(device).view(-1, 1)
                    
                    # Forward pass
                    logits, _ = model(marker_values, coverage)
                    loss = weighted_bce_loss(logits, labels)
                    
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
        
        # Update learning rate scheduler
        scheduler.step(avg_metric)
        
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


