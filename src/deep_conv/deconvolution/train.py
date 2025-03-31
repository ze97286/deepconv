import os
from collections import defaultdict

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, List
from deep_conv.deconvolution.loss import loss_fn

import time

def init_wandb(config, project_name="cfDNA-Deconvolution", entity=None):
    """
    Initialise Weights & Biases (wandb) logging for experiment tracking.

    This function:
      1) Creates (or resumes) a wandb run with the given config and user/project info.
      2) Automatically logs config details (hyperparams, etc.) to wandb.
      3) Generates a run name based on the current datetime.

    Args:
        config (dict):
            Dictionary containing experiment configurations (e.g., model hyperparams,
            dataset paths, training flags).
        project_name (str):
            The wandb project in which this run will appear.
        entity (str, optional):
            The wandb entity (username or team name). If None, defaults to your wandb default.

    Returns:
        run (wandb.run):
            The wandb run object, which allows further logging.
    """
    from datetime import datetime as dt
    run = wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        name=f"run_{dt.now().strftime('%Y%m%d_%H%M%S')}"
    )
    return run


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: optim.Optimizer,
    device: torch.device,
    log_interval: int = 500,
    accumulation_steps: int = 4
) -> Dict[str, float]:
    """
    Performs one training epoch on the given data loader.

    Steps:
      1) Iterate over each batch (fraction, coverage, y).
      2) Forward pass the batch through the model to get:
         (proportions, reconstruction, presence info).
      3) Compute the composite loss from `loss_fn`.
      4) Accumulate gradients, optionally using `accumulation_steps`.
      5) Perform an optimiser step (update model params) after `accumulation_steps` mini-batches.
      6) Keep track of various statistics (loss, presence detection metrics, etc.) and log them.
      7) Print progress every `log_interval` batches.

    Args:
        model (nn.Module):
            The model to be trained (must be in `model.train()` mode outside this function).
        loader (DataLoader):
            A DataLoader yielding batches of training data, each containing:
              - 'X': cfDNA marker methylation values,
              - 'coverage': coverage array for each marker,
              - 'y': ground-truth cell-type proportions (if supervised).
        optimiser (torch.optim.Optimizer):
            The optimiser (e.g., Adam) used to update model parameters.
        device (torch.device):
            The target device (e.g., GPU) where model and data will reside.
        log_interval (int):
            Frequency (in mini-batches) with which progress is printed/logged.
        accumulation_steps (int):
            Number of mini-batches over which to accumulate gradients before taking an optimiser step.

    Returns:
        Dict[str, float]: 
            A dictionary of epoch-level metrics (averaged across all batches), e.g. 
            {
                'total_loss': <float>,
                'grad_norm': <float>,
                'alpha_stats/mean': ...,
                ...
            }
    """
    model.train()  # Ensure model is in training mode (affects dropout/BatchNorm, etc.)
    epoch_stats = defaultdict(float)  # Will aggregate sums that we later average
    num_batches = 0

    # Start fresh for gradient accumulation
    optimiser.zero_grad()

    # Iterate over all batches
    for batch_idx, batch in enumerate(tqdm(loader, desc='Training')):
        # 1) Move data to appropriate device
        fraction = batch['X'].to(device)
        coverage = batch['coverage'].to(device)
        y_true = batch['y'].to(device)
        
        # 2) Forward pass
        #    alpha = predicted proportions
        #    reconstructed = predicted marker data
        #    valid_mask = coverage>0
        #    presence_probs/logits = presence detection
        alpha, reconstructed, valid_mask, presence_probs, presence_logits = model(fraction, coverage)
        
        # 3) Compute loss using our composite loss function
        loss, details = loss_fn(
            pred_props=alpha,
            true_props=y_true,
            reconstructed=reconstructed,
            marker_values=fraction,
            coverage=coverage,
            valid_mask=valid_mask,
            presence_probs=presence_probs,
            presence_logits=presence_logits,
        )
        
        # 4) Scale the loss if using gradient accumulation
        scaled_loss = loss / accumulation_steps
        
        # 5) Backpropagation
        scaled_loss.backward()
        
        # 6) Update params after `accumulation_steps` or final batch
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(loader)):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            optimiser.zero_grad()
            
            # Record the gradient norm
            epoch_stats['grad_norm'] += grad_norm.item()
            
            # Log intermediate stats (batch-level) if wandb is active
            if wandb.run is not None and (batch_idx + 1) % (log_interval // 2) == 0:
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/grad_norm": grad_norm.item(),
                    "batch/lr": optimiser.param_groups[0]['lr'],
                    "batch/step": batch_idx
                })
        
        # 7) Aggregate stats for this batch
        epoch_stats['total_loss'] += loss.item()
        for key, value in details.items():
            if isinstance(value, dict):
                # Nested dict means we add e.g. "alpha_stats/mean"
                for subkey, subvalue in value.items():
                    epoch_stats[f"{key}/{subkey}"] += subvalue
            else:
                epoch_stats[key] += value
        num_batches += 1
        
        # 8) Print to console every `log_interval` mini-batches
        if batch_idx % log_interval == 0:
            print(f"\nBatch {batch_idx} | Loss: {loss.item():.8f}")
            print(f"Alpha Mean: {details['alpha_stats']['mean']:.8f} | "
                  f"Std: {details['alpha_stats']['std']:.8f}")
            # Example of optional info if your dictionary has such keys
            if 'cd48_under' in details and 'cd48_over' in details:
                print(f"CD4/CD8 Under: {details['cd48_under']:.8f} | Over: {details['cd48_over']:.8f}")
            if 'weight_stats' in details:
                print(f"Weight Mean: {details['weight_stats']['mean']:.8f} | "
                      f"Max: {details['weight_stats']['max']:.8f}")
    
    # 9) Average out stats across all batches
    for key in epoch_stats:
        epoch_stats[key] /= num_batches
    
    return dict(epoch_stats)


def validate(
    model: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    presence_threshold: float = 0.01  # Fixed threshold for consistent metrics
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Evaluate model on validation sets with consistent metrics.

    Args:
        model: The model to be evaluated
        val_loaders: Dictionary of validation DataLoaders
        device: Device to run validation on
        presence_threshold: Fixed threshold for evaluation metrics
        
    Returns:
        avg_val_loss: Average validation loss
        val_stats: Dictionary of validation statistics
    """
    model.eval()
    
    val_stats = {}
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    threshold_results = {t: {} for t in thresholds}

    print(f"Validating with presence threshold: {presence_threshold}")

    with torch.no_grad():
        # Evaluate each named validation set
        for val_name, val_loader in val_loaders.items():
            loader_stats = defaultdict(float)
            num_batches = 0

            # For presence detection, track confusion across cell types
            num_cell_types = model.num_celltypes
            confusion = {
                'tp': torch.zeros(num_cell_types, device=device),
                'fp': torch.zeros(num_cell_types, device=device),
                'tn': torch.zeros(num_cell_types, device=device),
                'fn': torch.zeros(num_cell_types, device=device)
            }

            # Sample-based confusion (aggregate)
            tp_sum = fp_sum = fn_sum = tn_sum = 0
            sample_f1_scores = []
            sample_precision_scores = []
            sample_recall_scores = []

            # Prepare an entry for each threshold in this val_name
            for t in thresholds:
                threshold_results[t][val_name] = {
                    'mse': 0.0,
                    'mae': 0.0,
                    'detection_accuracy': 0.0,
                    'count': 0
                }

            # Go through each batch in this val set
            for batch in tqdm(val_loader, desc=f'Validating {val_name}'):
                fraction = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                y_true = batch['y'].to(device)
                
                # Forward pass
                alpha, reconstructed, valid_mask, presence_probs, presence_logits = model(fraction, coverage)
        
                # Use our standard loss function
                loss, details = loss_fn(
                    pred_props=alpha,
                    true_props=y_true,
                    reconstructed=reconstructed,
                    marker_values=fraction,
                    coverage=coverage,
                    valid_mask=valid_mask,
                    presence_probs=presence_probs,
                    presence_logits=presence_logits,
                    presence_threshold=presence_threshold,
                )
                
                # --- Presence confusion matrix
                batch_size = y_true.size(0)
                true_present = (y_true > presence_threshold)
                pred_present = (presence_probs > 0.5)
                
                # Sample-based confusion
                for i in range(batch_size):
                    sample_tp = torch.sum((pred_present[i] & true_present[i]).float()).item()
                    sample_fp = torch.sum((pred_present[i] & ~true_present[i]).float()).item()
                    sample_fn = torch.sum((~pred_present[i] & true_present[i]).float()).item()
                    sample_tn = torch.sum((~pred_present[i] & ~true_present[i]).float()).item()
                    
                    tp_sum += sample_tp
                    fp_sum += sample_fp
                    fn_sum += sample_fn
                    tn_sum += sample_tn
                    
                    # Per-sample precision/recall/f1
                    if sample_tp + sample_fp > 0:
                        sample_precision = sample_tp / (sample_tp + sample_fp)
                    else:
                        sample_precision = 1.0
                    
                    if sample_tp + sample_fn > 0:
                        sample_recall = sample_tp / (sample_tp + sample_fn)
                    else:
                        sample_recall = 1.0
                    
                    if sample_precision + sample_recall > 0:
                        sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall)
                    else:
                        sample_f1 = 0.0
                    
                    sample_precision_scores.append(sample_precision)
                    sample_recall_scores.append(sample_recall)
                    sample_f1_scores.append(sample_f1)
                
                # Cell-type-level confusion
                for ct in range(num_cell_types):
                    ct_true_present = true_present[:, ct]
                    ct_pred_present = pred_present[:, ct]
                    confusion['tp'][ct] += torch.sum((ct_pred_present & ct_true_present).float())
                    confusion['fp'][ct] += torch.sum((ct_pred_present & ~ct_true_present).float())
                    confusion['tn'][ct] += torch.sum((~ct_pred_present & ~ct_true_present).float())
                    confusion['fn'][ct] += torch.sum((~ct_pred_present & ct_true_present).float())
                
                # --- Evaluate threshold-based metrics for alpha
                for t in thresholds:
                    batch_results = threshold_results[t][val_name]
                    
                    thresholded_preds = torch.where(alpha < t, torch.zeros_like(alpha), alpha)
                    
                    # Renormalise
                    row_sums = thresholded_preds.sum(dim=1, keepdim=True)
                    valid_rows = (row_sums > 0).squeeze(-1)
                    if valid_rows.any():
                        thresholded_preds[valid_rows] /= row_sums[valid_rows]
                    
                    # MSE, MAE
                    mse = F.mse_loss(thresholded_preds, y_true)
                    mae = torch.abs(thresholded_preds - y_true).mean()
                    
                    # "Detection accuracy": predicted presence vs. true presence
                    pred_present_t = (thresholded_preds > 0)
                    true_present_t = (y_true > presence_threshold)
                    detection_accuracy = (pred_present_t == true_present_t).float().mean()
                    
                    batch_results['mse'] += mse.item() * batch_size
                    batch_results['mae'] += mae.item() * batch_size
                    batch_results['detection_accuracy'] += detection_accuracy.item() * batch_size
                    batch_results['count'] += batch_size
                
                # Accumulate stats for standard loss details
                loader_stats['loss'] += loss.item()
                for key, value in details.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            loader_stats[f"{key}/{subkey}"] += subvalue
                    else:
                        loader_stats[key] += value
                
                num_batches += 1
            
            # Post-processing for this validation set

            # Compute sample-based presence metrics
            if len(sample_precision_scores) > 0:
                overall_precision = sum(sample_precision_scores) / len(sample_precision_scores)
                overall_recall = sum(sample_recall_scores) / len(sample_recall_scores)
                overall_f1 = sum(sample_f1_scores) / len(sample_f1_scores)
            else:
                overall_precision = 0.0
                overall_recall = 0.0
                overall_f1 = 0.0
            
            # Cell-type-level confusion => compute class-based precision/recall/f1
            class_precision = confusion['tp'] / (confusion['tp'] + confusion['fp'] + 1e-8)
            class_recall = confusion['tp'] / (confusion['tp'] + confusion['fn'] + 1e-8)
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)
            
            print(f"\nValidation set: {val_name}")
            print(f"Total: TP={tp_sum}, FP={fp_sum}, FN={fn_sum}, TN={tn_sum}")
            print(f"Sample-based metrics - Precision: {overall_precision:.4f}, "
                  f"Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}")
            print(f"Class-based metrics - Precision: {class_precision.mean().item():.4f}, "
                  f"Recall: {class_recall.mean().item():.4f}, F1: {class_f1.mean().item():.4f}")
            
            # Add sample-based presence metrics
            loader_stats['avg_precision'] = overall_precision
            loader_stats['avg_recall'] = overall_recall
            loader_stats['avg_f1'] = overall_f1
            
            # Store per-cell-type metrics
            for ct in range(num_cell_types):
                loader_stats[f'precision_ct{ct}'] = class_precision[ct].item()
                loader_stats[f'recall_ct{ct}'] = class_recall[ct].item()
                loader_stats[f'f1_ct{ct}'] = class_f1[ct].item()
            
            # Finalise threshold-based results
            for t in thresholds:
                batch_results = threshold_results[t][val_name]
                if batch_results['count'] > 0:
                    for key in ['mse', 'mae', 'detection_accuracy']:
                        batch_results[key] /= batch_results['count']
                    for key, value in batch_results.items():
                        if key != 'count':
                            loader_stats[f'thresh_{t}_{key}'] = value
            
            # Average across all batches
            for key in loader_stats:
                if key not in ['avg_precision', 'avg_recall', 'avg_f1']:
                    loader_stats[key] /= num_batches
            
            val_stats[val_name] = dict(loader_stats)
    
    # Compute mean val loss across sets
    avg_val_loss = sum(stats['total_loss'] for stats in val_stats.values()) / len(val_stats)
    return avg_val_loss, val_stats

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loaders: Dict[str, DataLoader],
    model_path: str,
    num_epochs: int = 1000,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    use_wandb: bool = True,
    wandb_project: str = "cfDNA-Deconvolution",
    wandb_entity: str = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[nn.Module, float]:
    """
    The main training loop for the cell-type deconvolution model.
    
    Features:
      - Warmup for learning rate (first few epochs)
      - Early stopping based on validation loss (patience)
      - Post-training best checkpoint restoration
      - W&B integration for logging/plotting if `use_wandb=True`

    Args:
        model: The cell-type model to train
        train_loader: Provides training batches
        val_loaders: Dictionary of validation loaders
        model_path: Directory to store best model checkpoints
        num_epochs: Max number of epochs to train
        patience: # of epochs to wait for improvement before early stopping
        lr: Base learning rate
        weight_decay: L2 penalty for Adam
        use_wandb: If True, logs metrics/plots to Weights & Biases
        wandb_project: W&B project name
        wandb_entity: W&B entity (team name or username)
        device: Where to run the training (CPU or GPU)

    Returns:
        model: The trained model, loaded from the best checkpoint
        best_threshold: The chosen presence threshold for classification
    """
    model = model.to(device)
    
    # Setup optimizer with a single parameter group
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler that reduces LR on plateau of validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=patience // 2,
        verbose=True
    )
    
    # Ensure model_path exists
    os.makedirs(model_path, exist_ok=True)
    
    # Initialize W&B (Optional)
    if use_wandb:
        config = {
            "model_type": model.__class__.__name__,
            "num_markers": getattr(model, "num_markers", "unknown"),
            "num_cell_types": getattr(model, "num_celltypes", "unknown"),
            "feature_dim": getattr(model, "feature_dim", "unknown"),
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": train_loader.batch_size if hasattr(train_loader, "batch_size") else "unknown",
            "num_epochs": num_epochs,
            "patience": patience,
            "device": str(device)
        }
        run = init_wandb(config, project_name=wandb_project, entity=wandb_entity)
        wandb.watch(model, log="all", log_freq=100)
    
    # Preparation
    initial_lr = lr
    warmup_epochs = 5  # # of epochs for linearly ramping LR from 0 to lr
    
    history = defaultdict(list)
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Track the "best presence threshold" by evaluating multiple thresholds
    best_threshold = 0.01
    best_threshold_f1 = 0.0
    
    # Fixed presence threshold for evaluation metrics
    eval_presence_threshold = 0.01
    
    # Main Training Loop
    for epoch in range(num_epochs):
        print(f"\n🔹 Epoch {epoch + 1}/{num_epochs}")
        
        # LR Warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = initial_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"LR Warmup: {current_lr:.1e}")
        
        # Training for one epoch
        train_stats = train_epoch(model, train_loader, optimizer, device)
        
        # Validation with fixed threshold for consistent metrics
        avg_val_loss, val_stats = validate(
            model,
            val_loaders,
            device,
            presence_threshold=eval_presence_threshold  # Fixed threshold for evaluation
        )
        
        # Evaluate multiple thresholds for best F1
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
        threshold_f1_scores = {t: 0.0 for t in thresholds}
        
        # Sum up detection metric for each threshold across all val sets
        for t in thresholds:
            for val_name, stats in val_stats.items():
                threshold_key = f'thresh_{t}_detection_accuracy'
                if threshold_key in stats:
                    threshold_f1_scores[t] += stats[threshold_key]
            
            threshold_f1_scores[t] /= len(val_stats)
            
            # Update best threshold if improved
            if threshold_f1_scores[t] > best_threshold_f1:
                best_threshold_f1 = threshold_f1_scores[t]
                best_threshold = t
                print(f"New best threshold: {best_threshold} (F1: {best_threshold_f1:.4f})")
        
        # Record stats into `history`
        for key, value in train_stats.items():
            history[key].append(value)
        for val_name, stats in val_stats.items():
            for k, v in stats.items():
                history[f"{val_name}/{k}"].append(v)
        
        # Print summary
        print(f"\n🔹 Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_stats['total_loss']:.8f} | Grad Norm: {train_stats['grad_norm']:.8f}")
        if 'alpha_stats/mean' in train_stats:
            print(f"Alpha Mean: {train_stats['alpha_stats/mean']:.8f} | Std: {train_stats['alpha_stats/std']:.8f}")
        
        for val_name, stats in val_stats.items():
            print(f"{val_name} Loss: {stats['total_loss']:.8f}")
            if 'avg_precision' in stats and 'avg_recall' in stats and 'avg_f1' in stats:
                print(f"{val_name} Detection: P={stats['avg_precision']:.4f}, "
                      f"R={stats['avg_recall']:.4f}, F1={stats['avg_f1']:.4f}")
        
        # Log to W&B
        if use_wandb:
            wandb_logs = {
                "epoch": epoch,
                "train/loss": train_stats['total_loss'],
                "train/grad_norm": train_stats['grad_norm'],
                "val/avg_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "best_threshold": best_threshold,
                "best_threshold_f1": best_threshold_f1
            }
            # Add validation stats
            for val_name, stats in val_stats.items():
                for k, v in stats.items():
                    wandb_logs[f"val/{val_name}/{k}"] = v
            
            # Table for threshold F1
            wandb_logs["threshold_comparison"] = wandb.Table(
                data=[[t, threshold_f1_scores[t]] for t in thresholds],
                columns=["threshold", "f1_score"]
            )
            wandb.log(wandb_logs)
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        # Early stopping on avg_val_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            print(f"New best model with loss: {best_val_loss:.8f}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_threshold': best_threshold,
                'history': dict(history)
            }
            torch.save(checkpoint, os.path.join(model_path, "best_model.pt"))
            if use_wandb:
                wandb.save(os.path.join(model_path, "best_model.pt"))
        else:
            patience_counter += 1
        
        # Check patience for early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load the Best Model
    print("Loading best model (best overall validation loss)")
    checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot Training History
    plot_training_history(dict(history), os.path.join(model_path, "model_training"))
    
    # Summarize final recommended presence threshold
    print(f"\nRecommended threshold for inference: {best_threshold}")
    print(f"(Based on best F1 score: {best_threshold_f1:.4f})")
    
    # Cleanup wandb
    if use_wandb:
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["total_epochs"] = epoch + 1
        wandb.run.summary["best_threshold"] = best_threshold
        wandb.run.summary["best_threshold_f1"] = best_threshold_f1
        
        def remove_wandb_hooks(model):
            # Remove forward/backward/pre-forward hooks
            for k in list(model._forward_hooks.keys()):
                model._forward_hooks.pop(k)
            for k in list(model._backward_hooks.keys()):
                model._backward_hooks.pop(k)
            for k in list(model._forward_pre_hooks.keys()):
                model._forward_pre_hooks.pop(k)
            # Recurse to child modules
            for child in model.children():
                remove_wandb_hooks(child)

        remove_wandb_hooks(model)
        wandb.finish()
    
    return model, best_threshold

def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """
    Plot training history (losses, stats) using Plotly.

    This function:
      1) Reads `history`, which is a dict of lists mapping e.g. 'train/loss' -> [val0, val1, ...].
      2) Creates a 2×2 subplot figure:
         - (row1, col1): 'Loss Evolution'
         - (row1, col2): 'Valid Marker Ratio'
         - (row2, col1): 'Alpha Statistics'
         - (row2, col2): 'Theta Evolution'
      3) Plots lines for any keys matching "loss", "valid", "alpha", "theta" in the appropriate subplot.
      4) Optionally saves the figure to an HTML file for offline viewing or logs it.

    Args:
        history (Dict[str, List[float]]):
            A dictionary where each key is a metric name and the value is a list of epoch-level measurements.
        save_path (str):
            If provided, the figure is saved to `save_path + ".html"`. Otherwise, a fig.show() might be done.

    Note: 
        The user can adapt which keys go to which subplot as needed. This is just an example layout.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create figure with 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Loss Evolution',
            'Valid Marker Ratio',
            'Alpha Statistics',
            'Theta Evolution'
        )
    )

    # (A) Plot loss evolution
    loss_keys = [k for k in history.keys() if 'loss' in k.lower()]
    for key in loss_keys:
        fig.add_trace(
            go.Scatter(y=history[key], name=key),
            row=1, col=1
        )

    # (B) Plot valid marker ratio (just an example)
    valid_keys = [k for k in history.keys() if 'valid' in k.lower()]
    for key in valid_keys:
        fig.add_trace(
            go.Scatter(y=history[key], name=key),
            row=1, col=2
        )

    # (C) Plot alpha statistics if they exist
    if 'train_alpha_mean' in history:
        fig.add_trace(
            go.Scatter(y=history['train_alpha_mean'], name='Mean'),
            row=2, col=1
        )
    if 'train_alpha_std' in history:
        fig.add_trace(
            go.Scatter(y=history['train_alpha_std'], name='Std'),
            row=2, col=1
        )

    # (D) Plot theta evolution (placeholder if some 'theta' key is in history)
    if 'theta_mean' in history:
        fig.add_trace(
            go.Scatter(y=history['theta_mean'], name='Theta'),
            row=2, col=2
        )

    # Tweak layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Training History"
    )

    # Y-axis labels
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=2)

    # X-axis labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Epoch", row=i, col=j)

    # Save or show
    if save_path:
        fig.write_html(f"{save_path}.html")
    else:
        fig.show()