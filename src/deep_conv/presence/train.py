import torch 
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from deep_conv.presence.loss import *
from deep_conv.presence.model import *
import time
from deep_conv.presence.loss import FocalLoss

def train_single_cell_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loaders: dict,
    model_path: str,
    target_cell_type: int,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    presence_threshold: float = 0.0005,
    patience: int = 10,
    fp_weight: float = 8.0,  # Increased from 2.0 to 8.0
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    batch_accumulation: int = 1,
    fp16_training: bool = True,
    eval_every: int = 1,
    use_focal_loss: bool = True,
    class_ratio_estimation: bool = True
):
    """
    Balanced training function addressing class imbalance issues.
    """
    # Enable mixed precision training if requested and available
    scaler = torch.cuda.amp.GradScaler() if fp16_training and torch.cuda.is_available() else None
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create focal loss if requested (or use BCE with higher weights)
    if use_focal_loss:
        # Alpha controls class balance - 0.25 means more weight on positives
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience//2, verbose=True
    )
    
    # Directory for saving models
    os.makedirs(model_path, exist_ok=True)
    
    # Estimate class ratio if enabled
    if class_ratio_estimation:
        print("Estimating class ratio in training data...")
        positive_count = 0
        total_count = 0
        with torch.no_grad():
            for batch in train_loader:
                true_props = batch['y'].to(device)
                true_props_target = true_props[:, target_cell_type]
                # Convert to binary presence/absence
                presence_target = (true_props_target > presence_threshold).float()
                positive_count += presence_target.sum().item()
                total_count += len(presence_target)
                
        class_ratio = positive_count / total_count
        negative_ratio = 1.0 - class_ratio
        print(f"Class ratio - Positive: {class_ratio:.4f}, Negative: {negative_ratio:.4f}")
        
        # Adjust fp_weight based on class ratio
        if negative_ratio < 0.3:  # Very imbalanced
            fp_weight = max(10.0, 1.0 / negative_ratio)
        elif negative_ratio < 0.4:  # Moderately imbalanced
            fp_weight = max(5.0, 0.5 / negative_ratio)
            
        print(f"Using false positive weight: {fp_weight:.2f}")
    
    # Track best model
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Track best specificity (to focus on improving this)
    best_specificity = 0.0
    
    # Create a simple profiler to track time spent in different parts
    time_metrics = {'data_loading': 0, 'forward': 0, 'backward': 0, 'validation': 0}
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'f1': 0.0,
            'confusion': {
                'tp': 0,
                'fp': 0,
                'tn': 0,
                'fn': 0
            }
        }
        num_batches = 0
        optimizer.zero_grad()
        
        # Reset time metrics
        for k in time_metrics:
            time_metrics[k] = 0
            
        batch_start = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            data_time = time.time() - batch_start
            time_metrics['data_loading'] += data_time
            
            # Get batch data
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            true_props = batch['y'].to(device)
            
            # Extract ground truth for target cell type
            true_props_target = true_props[:, target_cell_type].view(-1, 1)
            # Convert to binary presence/absence
            presence_target = (true_props_target > presence_threshold).float()
            
            # Create weights (emphasis on false positives)
            weights = torch.ones_like(presence_target)
            weights[presence_target == 0] = fp_weight
            
            # Forward pass with mixed precision if enabled
            forward_start = time.time()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Get model outputs (with L2 penalty for balanced model)
                    if hasattr(model, 'l2_reg'):
                        presence_logit, valid_mask, l2_penalty = model(marker_values, coverage)
                    else:
                        presence_logit, valid_mask = model(marker_values, coverage)
                        l2_penalty = 0.0
                        
                    # Calculate loss - focal or weighted BCE
                    if use_focal_loss:
                        loss = focal_loss(presence_logit, presence_target, weights)
                    else:
                        loss = F.binary_cross_entropy_with_logits(
                            presence_logit, presence_target, weight=weights, reduction='mean'
                        )
                    
                    # Add L2 penalty if applicable
                    if l2_penalty > 0:
                        loss += l2_penalty
            else:
                # Same logic but without mixed precision
                if hasattr(model, 'l2_reg'):
                    presence_logit, valid_mask, l2_penalty = model(marker_values, coverage)
                else:
                    presence_logit, valid_mask = model(marker_values, coverage)
                    l2_penalty = 0.0
                    
                if use_focal_loss:
                    loss = focal_loss(presence_logit, presence_target, weights)
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        presence_logit, presence_target, weight=weights, reduction='mean'
                    )
                
                if l2_penalty > 0:
                    loss += l2_penalty
            
            forward_time = time.time() - forward_start
            time_metrics['forward'] += forward_time
            
            # Scale loss based on gradient accumulation
            loss = loss / batch_accumulation
            
            # Backward pass with mixed precision if enabled
            backward_start = time.time()
            if scaler is not None:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % batch_accumulation == 0 or batch_idx == len(train_loader) - 1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % batch_accumulation == 0 or batch_idx == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            backward_time = time.time() - backward_start
            time_metrics['backward'] += backward_time
            
            # Calculate metrics (outside of grad computation)
            with torch.no_grad():
                presence_prob = torch.sigmoid(presence_logit)
                presence_pred = (presence_prob > 0.5).float()
                
                # Accuracy
                accuracy = torch.mean((presence_pred == presence_target).float())
                
                # Confusion matrix
                tp = torch.sum((presence_pred == 1) & (presence_target == 1)).item()
                fp = torch.sum((presence_pred == 1) & (presence_target == 0)).item()
                tn = torch.sum((presence_pred == 0) & (presence_target == 0)).item()
                fn = torch.sum((presence_pred == 0) & (presence_target == 1)).item()
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Accumulate metrics
                train_metrics['loss'] += loss.item() * batch_accumulation
                train_metrics['accuracy'] += accuracy.item()
                train_metrics['precision'] += precision
                train_metrics['recall'] += recall
                train_metrics['specificity'] += specificity
                train_metrics['f1'] += f1
                train_metrics['confusion']['tp'] += tp
                train_metrics['confusion']['fp'] += fp
                train_metrics['confusion']['tn'] += tn
                train_metrics['confusion']['fn'] += fn
            
            num_batches += 1
            batch_start = time.time()
            
            # Print progress occasionally
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item() * batch_accumulation:.4f}, "
                      f"F1: {f1:.4f}, Spec: {specificity:.4f}")
        
        # Average training metrics
        for metric in train_metrics:
            if metric != 'confusion':
                train_metrics[metric] /= num_batches
        
        # Overall confusion matrix
        cm = train_metrics['confusion']
        train_tp, train_fp = cm['tp'], cm['fp']
        train_tn, train_fn = cm['tn'], cm['fn']
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}, "
              f"Specificity: {train_metrics['specificity']:.4f}")
        print(f"Train Confusion - TP: {train_tp}, FP: {train_fp}, TN: {train_tn}, FN: {train_fn}")
        
        # Only do validation every eval_every epochs or on last epoch
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            # Validation phase
            validation_start = time.time()
            model.eval()
            val_metrics = {}
            
            for val_name, val_loader in val_loaders.items():
                val_set_metrics = {
                    'loss': 0.0,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'specificity': 0.0,
                    'f1': 0.0,
                    'balanced_accuracy': 0.0,
                    'confusion': {
                        'tp': 0,
                        'fp': 0,
                        'tn': 0,
                        'fn': 0
                    }
                }
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Validating {val_name}"):
                        # Get batch data
                        marker_values = batch['X'].to(device)
                        coverage = batch['coverage'].to(device)
                        true_props = batch['y'].to(device)
                        
                        # Extract ground truth for target cell type
                        true_props_target = true_props[:, target_cell_type].view(-1, 1)
                        # Convert to binary presence/absence
                        presence_target = (true_props_target > presence_threshold).float()
                        
                        # Forward pass
                        if hasattr(model, 'l2_reg'):
                            presence_logit, valid_mask, _ = model(marker_values, coverage)
                        else:
                            presence_logit, valid_mask = model(marker_values, coverage)
                        
                        # Calculate loss (for monitoring)
                        if use_focal_loss:
                            loss = focal_loss(presence_logit, presence_target)
                        else:
                            loss = F.binary_cross_entropy_with_logits(
                                presence_logit, presence_target, reduction='mean'
                            )
                        
                        # Calculate metrics
                        presence_prob = torch.sigmoid(presence_logit)
                        presence_pred = (presence_prob > 0.5).float()
                        
                        # Accuracy
                        accuracy = torch.mean((presence_pred == presence_target).float())
                        
                        # Confusion matrix
                        tp = torch.sum((presence_pred == 1) & (presence_target == 1)).item()
                        fp = torch.sum((presence_pred == 1) & (presence_target == 0)).item()
                        tn = torch.sum((presence_pred == 0) & (presence_target == 0)).item()
                        fn = torch.sum((presence_pred == 0) & (presence_target == 1)).item()
                        
                        # Calculate metrics
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                        balanced_accuracy = (recall + specificity) / 2
                        
                        # Accumulate metrics
                        val_set_metrics['loss'] += loss.item()
                        val_set_metrics['accuracy'] += accuracy.item()
                        val_set_metrics['precision'] += precision
                        val_set_metrics['recall'] += recall
                        val_set_metrics['specificity'] += specificity
                        val_set_metrics['f1'] += f1
                        val_set_metrics['balanced_accuracy'] += balanced_accuracy
                        val_set_metrics['confusion']['tp'] += tp
                        val_set_metrics['confusion']['fp'] += fp
                        val_set_metrics['confusion']['tn'] += tn
                        val_set_metrics['confusion']['fn'] += fn
                        
                        num_val_batches += 1
                
                # Average validation metrics
                for metric in val_set_metrics:
                    if metric != 'confusion':
                        val_set_metrics[metric] /= num_val_batches
                
                # Store metrics
                val_metrics[val_name] = val_set_metrics
                
                # Print metrics
                print(f"{val_name} - F1: {val_set_metrics['f1']:.4f}, "
                      f"Precision: {val_set_metrics['precision']:.4f}, "
                      f"Recall: {val_set_metrics['recall']:.4f}, "
                      f"Specificity: {val_set_metrics['specificity']:.4f}, "
                      f"Balanced Acc: {val_set_metrics['balanced_accuracy']:.4f}")
                
                # Detailed confusion matrix
                cm = val_set_metrics['confusion']
                print(f"Confusion Matrix - TP: {cm['tp']}, FP: {cm['fp']}, TN: {cm['tn']}, FN: {cm['fn']}")
            
            time_metrics['validation'] = time.time() - validation_start
            
            # Calculate average F1 and specificity across validation sets
            avg_val_f1 = np.mean([metrics['f1'] for metrics in val_metrics.values()])
            avg_val_specificity = np.mean([metrics['specificity'] for metrics in val_metrics.values()])
            avg_balanced_acc = np.mean([metrics['balanced_accuracy'] for metrics in val_metrics.values()])
            
            # Update best specificity if improved
            if avg_val_specificity > best_specificity:
                best_specificity = avg_val_specificity
                print(f"New best specificity: {best_specificity:.4f}")
                
                # Save model with best specificity
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'specificity': best_specificity
                }
                torch.save(checkpoint, os.path.join(model_path, f'best_specificity_celltype_{target_cell_type}.pt'))
            
            # For model selection, prioritize balanced accuracy if specificity is decent
            selection_metric = avg_balanced_acc if avg_val_specificity > 0.5 else avg_val_specificity
            
            # Update learning rate scheduler based on balanced accuracy
            scheduler.step(selection_metric)
            
            # Check for improvement in the selection metric
            improved = False
            if selection_metric > best_val_f1:  # Reusing best_val_f1 variable
                best_val_f1 = selection_metric
                best_epoch = epoch
                patience_counter = 0
                improved = True
                
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'balanced_accuracy': avg_balanced_acc,
                    'specificity': avg_val_specificity,
                    'f1': avg_val_f1
                }
                torch.save(checkpoint, os.path.join(model_path, f'best_model_celltype_{target_cell_type}.pt'))
                
                print(f"New best model saved! Balanced Acc: {avg_balanced_acc:.4f}, Spec: {avg_val_specificity:.4f}, F1: {avg_val_f1:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            print(f"Skipping validation for epoch {epoch+1} (every {eval_every} epochs)")
        
        # Print epoch timing information
        epoch_time = time.time() - epoch_start
        print(f"Epoch completed in {epoch_time:.2f}s")
        if sum(time_metrics.values()) > 0:
            time_percentages = {k: time_metrics[k]/epoch_time*100 for k in time_metrics}
            print(f"Time breakdown: "
                  f"Data: {time_percentages['data_loading']:.1f}%, "
                  f"Forward: {time_percentages['forward']:.1f}%, "
                  f"Backward: {time_percentages['backward']:.1f}%, "
                  f"Validation: {time_percentages['validation']:.1f}%")
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_path, f'best_model_celltype_{target_cell_type}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Also return best specificity model if different
    spec_checkpoint_path = os.path.join(model_path, f'best_specificity_celltype_{target_cell_type}.pt')
    if os.path.exists(spec_checkpoint_path):
        spec_checkpoint = torch.load(spec_checkpoint_path)
        specificity_model = type(model)(
            num_markers=model.num_markers,
            target_ids=model.target_ids.cpu().numpy(),
            target_cell_type=model.target_cell_type
        ).to(device)
        specificity_model.load_state_dict(spec_checkpoint['model_state_dict'])
        print(f"Also loaded best specificity model from epoch {spec_checkpoint['epoch']+1}")
        return model, specificity_model
        
    return model


# def train_presence_model(
#     model: nn.Module,
#     train_loader: torch.utils.data.DataLoader,
#     val_loaders: dict,
#     model_path: str,
#     num_epochs: int = 1000,
#     learning_rate: float = 1e-3,
#     weight_decay: float = 1e-5,
#     presence_threshold: float = 0.0005,  # 0.05% threshold
#     patience: int = 10,
#     cell_types: list[str] = None, 
#     device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ):
#     """
#     Train the cell type presence detection model.
    
#     Args:
#         model: The model to train
#         train_loader: DataLoader for training data
#         val_loaders: Dict of validation DataLoaders
#         model_path: Path to save the model
#         num_epochs: Number of epochs to train
#         learning_rate: Learning rate
#         weight_decay: L2 regularization parameter
#         presence_threshold: Threshold for considering a cell type present
#         patience: Patience for early stopping
#         cell_types: List of cell type names for better reporting
#         device: Device to train on
        
#     Returns:
#         The trained model
#     """
#     # Move model to device
#     model = model.to(device)
    
#     # Create optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
#     # Learning rate scheduler
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=patience//2, verbose=True
#     )
    
#     # Create directory for saving models
#     os.makedirs(model_path, exist_ok=True)
    
#     # Track training history
#     history = defaultdict(list)
    
#     # Track best model
#     best_val_f1 = 0.0
#     best_epoch = 0
#     patience_counter = 0
    
#     # Training loop
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
        
#         # Training phase
#         model.train()
#         train_metrics = defaultdict(float)
#         num_batches = 0
        
#         for batch in tqdm(train_loader, desc="Training"):
#             # Get batch data
#             marker_values = batch['X'].to(device)
#             coverage = batch['coverage'].to(device)
#             true_props = batch['y'].to(device)
            
#             # Forward pass
#             presence_logits, valid_mask = model(marker_values, coverage)
            
#             # Calculate loss
#             loss, details = presence_loss_fn(
#                 presence_logits=presence_logits,
#                 true_props=true_props,
#                 valid_mask=valid_mask,
#                 presence_threshold=presence_threshold,
#                 device=device,
#                 cell_type_names=cell_types
#             )
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             # Accumulate metrics (only scalar metrics for training)
#             for metric, value in details.items():
#                 if isinstance(value, dict):
#                     continue
#                 train_metrics[metric] += value
            
#             num_batches += 1
        
#         # Average training metrics
#         for metric in train_metrics:
#             train_metrics[metric] /= num_batches
#             history[f"train_{metric}"].append(train_metrics[metric])
        
#         print(f"Train Loss: {train_metrics['loss']:.4f}, "
#               f"F1: {train_metrics['f1']:.4f}, "
#               f"Precision: {train_metrics['precision']:.4f}, "
#               f"Recall: {train_metrics['recall']:.4f}, "
#               f"Specificity: {train_metrics['specificity']:.4f}")
        
#         # Validation phase
#         model.eval()
#         val_metrics = defaultdict(lambda: defaultdict(float))
        
#         for val_name, val_loader in val_loaders.items():
#             val_set_metrics = {}
#             num_val_batches = 0
            
#             with torch.no_grad():
#                 for batch in tqdm(val_loader, desc=f"Validating {val_name}"):
#                     # Get batch data
#                     marker_values = batch['X'].to(device)
#                     coverage = batch['coverage'].to(device)
#                     true_props = batch['y'].to(device)
                    
#                     # Forward pass
#                     presence_logits, valid_mask = model(marker_values, coverage)
                    
#                     # Calculate metrics
#                     _, details = presence_loss_fn(
#                         presence_logits=presence_logits,
#                         true_props=true_props,
#                         valid_mask=valid_mask,
#                         presence_threshold=presence_threshold,
#                         device=device,
#                         cell_type_names=cell_types,
#                     )
                    
#                     # For the first batch, initialize all metrics
#                     if num_val_batches == 0:
#                         val_set_metrics = details.copy()
#                     else:
#                         # Accumulate scalar metrics
#                         for metric, value in details.items():
#                             if not isinstance(value, dict):
#                                 val_set_metrics[metric] += value
                            
#                             # Handle special dictionaries
#                             elif metric == 'confusion_counts':
#                                 for count_key, count_value in value.items():
#                                     val_set_metrics[metric][count_key] += count_value
                            
#                             elif metric == 'metrics_per_class':
#                                 for key, array_value in value.items():
#                                     val_set_metrics[metric][key] += array_value
                            
#                             # For worst_performers, just keep the last batch's values
#                             elif metric == 'worst_performers':
#                                 val_set_metrics[metric] = value
                    
#                     num_val_batches += 1
            
#             # Average scalar metrics and per-class metrics
#             for metric, value in val_set_metrics.items():
#                 if not isinstance(value, dict):
#                     val_set_metrics[metric] /= num_val_batches
#                 elif metric == 'metrics_per_class':
#                     for key in val_set_metrics[metric]:
#                         val_set_metrics[metric][key] /= num_val_batches
            
#             # Store in val_metrics and history
#             val_metrics[val_name] = val_set_metrics
            
#             # Store scalar metrics in history
#             for metric, value in val_set_metrics.items():
#                 if not isinstance(value, dict):
#                     history[f"val_{val_name}_{metric}"].append(value)
            
#             # Print evaluation metrics
#             print(f"{val_name} - F1: {val_set_metrics['f1']:.4f}, "
#                   f"Precision: {val_set_metrics['precision']:.4f}, "
#                   f"Recall: {val_set_metrics['recall']:.4f}, "
#                   f"Specificity: {val_set_metrics['specificity']:.4f}")
            
#             # Print detailed evaluation
#             print(f"\n===== Evaluation on {val_name} dataset =====")
#             print(f"Overall Metrics:")
#             print(f"  Accuracy: {val_set_metrics['accuracy']:.4f}")
#             print(f"  Precision: {val_set_metrics['precision']:.4f}")
#             print(f"  Recall: {val_set_metrics['recall']:.4f}")
#             print(f"  Specificity: {val_set_metrics['specificity']:.4f}")
#             print(f"  F1 Score: {val_set_metrics['f1']:.4f}")

#             # Confusion matrix counts
#             print(f"\nConfusion Matrix Counts:")
#             if 'confusion_counts' in val_set_metrics:
#                 print(f"  True Positives: {val_set_metrics['confusion_counts']['tp']}")
#                 print(f"  False Positives: {val_set_metrics['confusion_counts']['fp']}")
#                 print(f"  True Negatives: {val_set_metrics['confusion_counts']['tn']}")
#                 print(f"  False Negatives: {val_set_metrics['confusion_counts']['fn']}")

#             # Valid marker ratio
#             if 'valid_ratio_mean' in val_set_metrics:
#                 print(f"Mean Valid Marker Ratio: {val_set_metrics['valid_ratio_mean']:.4f}")

#             # Per-cell-type metrics
#             if 'metrics_per_class' in val_set_metrics:
#                 print("\nPer-Cell-Type Metrics:")
#                 metrics_per_class = val_set_metrics['metrics_per_class']
#                 if 'precision' in metrics_per_class:
#                     for i in range(len(metrics_per_class['precision'])):
#                         cell_name = cell_types[i] if cell_types else f"Cell type {i}"
#                         print(f"\n  {cell_name}:")
#                         if 'present_percent' in metrics_per_class:
#                             print(f"    Present in {metrics_per_class['present_percent'][i]:.2f}% of samples")
#                         if all(k in metrics_per_class for k in ['tp', 'fp', 'tn', 'fn']):
#                             print(f"    TP: {metrics_per_class['tp'][i]}, FP: {metrics_per_class['fp'][i]}")
#                             print(f"    TN: {metrics_per_class['tn'][i]}, FN: {metrics_per_class['fn'][i]}")
#                         print(f"    Precision: {metrics_per_class['precision'][i]:.4f}")
#                         print(f"    Recall: {metrics_per_class['recall'][i]:.4f}")
#                         print(f"    Specificity: {metrics_per_class['specificity'][i]:.4f}")
#                         print(f"    F1: {metrics_per_class['f1'][i]:.4f}")

#             # Worst performers
#             if 'worst_performers' in val_set_metrics:
#                 print("\nWorst Performing Cell Types:")
#                 for category, info in val_set_metrics['worst_performers'].items():
#                     print(f"  {category.replace('_', ' ').title()}: {info['name']} ({info['value']:.4f})")
        
#         # Calculate average F1 across validation sets
#         avg_val_f1 = np.mean([metrics['f1'] for metrics in val_metrics.values()])
        
#         # Update learning rate scheduler
#         scheduler.step(avg_val_f1)
        
#         # Check for improvement
#         if avg_val_f1 > best_val_f1:
#             best_val_f1 = avg_val_f1
#             best_epoch = epoch
#             patience_counter = 0
            
#             # Save best model
#             checkpoint = {
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_val_f1': best_val_f1,
#                 'history': dict(history)
#             }
#             torch.save(checkpoint, os.path.join(model_path, 'best_presence_model.pt'))
            
#             print(f"New best model saved! F1: {best_val_f1:.4f}")
#         else:
#             patience_counter += 1
#             print(f"No improvement. Patience: {patience_counter}/{patience}")
        
#         # Early stopping
#         if patience_counter >= patience:
#             print(f"Early stopping triggered after {epoch+1} epochs")
#             break
    
#     # Load best model
#     checkpoint = torch.load(os.path.join(model_path, 'best_presence_model.pt'))
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print(f"Loaded best model from epoch {checkpoint['epoch']+1} with F1 {checkpoint['best_val_f1']:.4f}")
    
#     return model