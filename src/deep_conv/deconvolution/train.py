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
    accumulation_steps: int = 2,
    epoch: int = 0,
    focal_loss_weight: float = 0.01,
    presence_threshold: float = 0.01
) -> Dict[str, float]:
    model.train()
    epoch_stats = defaultdict(float)
    # Initialize nested dictionaries for per-cell-type stats
    epoch_stats['presence_stats_per_celltype'] = defaultdict(lambda: defaultdict(float))
    epoch_stats['alpha_stats_per_celltype'] = defaultdict(lambda: defaultdict(float))
    num_batches = 0

    optimiser.zero_grad()

    for batch_idx, batch in enumerate(tqdm(loader, desc='Training')):
        fraction = batch['X'].to(device)
        coverage = batch['coverage'].to(device)
        y_true = batch['y'].to(device)
        
        alpha, reconstructed, valid_mask, presence_probs, presence_logits = model(fraction, coverage)
        
        loss, details = loss_fn(
            pred_props=alpha,
            true_props=y_true,
            reconstructed=reconstructed,
            marker_values=fraction,
            coverage=coverage,
            valid_mask=valid_mask,
            presence_probs=presence_probs,
            presence_logits=presence_logits,
            focal_loss_weight=focal_loss_weight,
            presence_threshold=presence_threshold
        )
        
        # Compute proportion accuracy metrics for training
        mae = torch.abs(alpha - y_true).mean()
        mse = F.mse_loss(alpha, y_true)
        epoch_stats['mae'] += mae.item()
        epoch_stats['mse'] += mse.item()
        
        # Log individual loss components
        if batch_idx % log_interval == 0:
            print(f"\nBatch {batch_idx} | Loss Components - "
                  f"loss_props: {details['loss_props']:.4f}, "
                  f"recon_loss: {details['recon_loss']:.4f}, "
                  f"sparsity_penalty: {details['sparsity_loss']:.4f}, "
                  f"presence_loss: {details['presence_loss']:.4f}")
            print(f"Batch {batch_idx} | Proportion Accuracy - MAE: {mae.item():.4f}, MSE: {mse.item():.4f}")
        
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(loader)):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            optimiser.zero_grad()
            
            epoch_stats['grad_norm'] += grad_norm.item()
            
            if wandb.run is not None and (batch_idx + 1) % (log_interval // 2) == 0:
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/grad_norm": grad_norm.item(),
                    "batch/lr": optimiser.param_groups[0]['lr'],
                    "batch/step": batch_idx
                })
        
        epoch_stats['total_loss'] += scaled_loss.item() * accumulation_steps
        for key, value in details.items():
            if isinstance(value, dict):
                # Handle nested dictionaries (e.g., presence_stats_per_celltype, alpha_stats_per_celltype)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        # Handle per-cell-type stats
                        for subsubkey, subsubvalue in subvalue.items():
                            epoch_stats[f"{key}"][subkey][subsubkey] += subsubvalue
                    else:
                        epoch_stats[f"{key}/{subkey}"] += subvalue
            else:
                epoch_stats[key] += value
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            print(f"\nBatch {batch_idx} | Loss: {loss.item():.8f}")
            print(f"Alpha Mean: {details['alpha_stats']['mean']:.8f} | "
                  f"Std: {details['alpha_stats']['std']:.8f}")
            if 'cd48_under' in details and 'cd48_over' in details:
                print(f"CD4/CD8 Under: {details['cd48_under']:.8f} | Over: {details['cd48_over']:.8f}")
            if 'weight_stats' in details:
                print(f"Weight Mean: {details['weight_stats']['mean']:.8f} | "
                      f"Max: {details['weight_stats']['max']:.8f}")
    
    # Average the scalar stats
    for key in list(epoch_stats.keys()):
        if key not in ['presence_stats_per_celltype', 'alpha_stats_per_celltype']:
            epoch_stats[key] /= num_batches
    
    # Average the per-cell-type stats
    for key in ['presence_stats_per_celltype', 'alpha_stats_per_celltype']:
        for subkey in epoch_stats[key]:
            for subsubkey in epoch_stats[key][subkey]:
                epoch_stats[key][subkey][subsubkey] /= num_batches
    
    # Log average loss components for the epoch
    print(f"\nEpoch {epoch + 1} | Average Loss Components - "
          f"loss_props: {epoch_stats['loss_props']:.4f}, "
          f"recon_loss: {epoch_stats['recon_loss']:.4f}, "
          f"sparsity_penalty: {epoch_stats['sparsity_loss']:.4f}, "
          f"presence_loss: {epoch_stats['presence_loss']:.4f}")
    print(f"Epoch {epoch + 1} | Average Proportion Accuracy - MAE: {epoch_stats['mae']:.4f}, MSE: {epoch_stats['mse']:.4f}")
    
    return dict(epoch_stats)


def validate(
    model: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    presence_threshold: float = 0.01,
    focal_loss_weight: float = 0.01
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    model.eval()
    
    val_stats = {}
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    threshold_results = {t: {} for t in thresholds}
    total_samples = 0
    weighted_loss_sum = 0.0
    total_batches = 0

    print(f"Validating with presence threshold: {presence_threshold}")

    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            loader_stats = defaultdict(float)
            # Initialize nested dictionaries for per-cell-type stats
            loader_stats['presence_stats_per_celltype'] = defaultdict(lambda: defaultdict(float))
            loader_stats['alpha_stats_per_celltype'] = defaultdict(lambda: defaultdict(float))
            num_batches = 0

            confusion = {
                'tp': torch.zeros(model.num_celltypes, device=device),
                'fp': torch.zeros(model.num_celltypes, device=device),
                'tn': torch.zeros(model.num_celltypes, device=device),
                'fn': torch.zeros(model.num_celltypes, device=device)
            }

            tp_sum = fp_sum = fn_sum = tn_sum = 0
            sample_f1_scores = []
            sample_precision_scores = []
            sample_recall_scores = []
            mae_sum = 0.0
            mse_sum = 0.0

            for t in thresholds:
                threshold_results[t][val_name] = {
                    'mse': 0.0,
                    'mae': 0.0,
                    'detection_accuracy': 0.0,
                    'count': 0
                }

            # Determine absent indices from the first sample of the dataset
            first_batch = next(iter(val_loader))
            y_true_first = first_batch['y'].to(device)  # Shape: [batch_size, num_celltypes]
            # Use the first sample to determine absent cell types (proportion == 0)
            first_sample = y_true_first[0]  # Shape: [num_celltypes]
            absent_mask = (first_sample == 0)
            absent_indices = torch.nonzero(absent_mask, as_tuple=False).squeeze(-1).tolist()
            print(f"\nAbsent Cell Types for {val_name}: {absent_indices}")

            for batch in tqdm(val_loader, desc=f'Validating {val_name}'):
                fraction = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                y_true = batch['y'].to(device)
                
                alpha, reconstructed, valid_mask, presence_probs, presence_logits = model(fraction, coverage)
        
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
                    focal_loss_weight=focal_loss_weight
                )
                
                # Compute proportion accuracy metrics
                mae = torch.abs(alpha - y_true).mean()
                mse = F.mse_loss(alpha, y_true)
                batch_size = y_true.size(0)
                mae_sum += mae.item() * batch_size
                mse_sum += mse.item() * batch_size
                
                true_present = (y_true > presence_threshold)
                pred_present = (presence_probs > 0.5)
                
                for i in range(batch_size):
                    sample_tp = torch.sum((pred_present[i] & true_present[i]).float()).item()
                    sample_fp = torch.sum((pred_present[i] & ~true_present[i]).float()).item()
                    sample_fn = torch.sum((~pred_present[i] & true_present[i]).float()).item()
                    sample_tn = torch.sum((~pred_present[i] & ~true_present[i]).float()).item()
                    
                    tp_sum += sample_tp
                    fp_sum += sample_fp
                    fn_sum += sample_fn
                    tn_sum += sample_tn
                    
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
                
                for ct in range(model.num_celltypes):
                    ct_true_present = true_present[:, ct]
                    ct_pred_present = pred_present[:, ct]
                    confusion['tp'][ct] += torch.sum((ct_pred_present & ct_true_present).float())
                    confusion['fp'][ct] += torch.sum((ct_pred_present & ~ct_true_present).float())
                    confusion['tn'][ct] += torch.sum((~ct_pred_present & ~ct_true_present).float())
                    confusion['fn'][ct] += torch.sum((~ct_pred_present & ct_true_present).float())
                
                for t in thresholds:
                    batch_results = threshold_results[t][val_name]
                    
                    thresholded_preds = torch.where(alpha < t, torch.zeros_like(alpha), alpha)
                    
                    row_sums = thresholded_preds.sum(dim=1, keepdim=True)
                    valid_rows = (row_sums > 0).squeeze(-1)
                    if valid_rows.any():
                        thresholded_preds[valid_rows] /= row_sums[valid_rows]
                    
                    mse = F.mse_loss(thresholded_preds, y_true)
                    mae = torch.abs(thresholded_preds - y_true).mean()
                    
                    pred_present_t = (thresholded_preds > 0)
                    true_present_t = (y_true > presence_threshold)
                    detection_accuracy = (pred_present_t == true_present_t).float().mean()
                    
                    batch_results['mse'] += mse.item() * batch_size
                    batch_results['mae'] += mae.item() * batch_size
                    batch_results['detection_accuracy'] += detection_accuracy.item() * batch_size
                    batch_results['count'] += batch_size
                
                loader_stats['loss'] += loss.item()
                loader_stats['loss_props'] += details['loss_props']
                loader_stats['recon_loss'] += details['recon_loss']
                loader_stats['sparsity_loss'] += details['sparsity_loss']
                loader_stats['presence_loss'] += details['presence_loss']
                for key, value in details.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries (e.g., presence_stats_per_celltype, alpha_stats_per_celltype)
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, dict):
                                # Handle per-cell-type stats
                                for subsubkey, subsubvalue in subvalue.items():
                                    loader_stats[f"{key}"][subkey][subsubkey] += subsubvalue
                            else:
                                loader_stats[f"{key}/{subkey}"] += subvalue
                    else:
                        loader_stats[key] += value
                
                num_batches += 1
                total_batches += 1
                weighted_loss_sum += loss.item()
            
            if len(sample_precision_scores) > 0:
                overall_precision = sum(sample_precision_scores) / len(sample_precision_scores)
                overall_recall = sum(sample_recall_scores) / len(sample_recall_scores)
                overall_f1 = sum(sample_f1_scores) / len(sample_f1_scores)
            else:
                overall_precision = 0.0
                overall_recall = 0.0
                overall_f1 = 0.0
            
            class_precision = confusion['tp'] / (confusion['tp'] + confusion['fp'] + 1e-8)
            class_recall = confusion['tp'] / (confusion['tp'] + confusion['fn'] + 1e-8)
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)
            
            mae_avg = mae_sum / (num_batches * val_loader.batch_size)
            mse_avg = mse_sum / (num_batches * val_loader.batch_size)
            
            print(f"\nValidation set: {val_name}")
            print(f"Total: TP={tp_sum}, FP={fp_sum}, FN={fn_sum}, TN={tn_sum}")
            print(f"Sample-based metrics - Precision: {overall_precision:.4f}, "
                  f"Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}")
            print(f"Class-based metrics - Precision: {class_precision.mean().item():.4f}, "
                  f"Recall: {class_recall.mean().item():.4f}, F1: {class_f1.mean().item():.4f}")
            print(f"Proportion accuracy - MAE: {mae_avg:.4f}, MSE: {mse_avg:.4f}")
            print(f"Average Loss Components - "
                  f"loss_props: {loader_stats['loss_props']/num_batches:.4f}, "
                  f"recon_loss: {loader_stats['recon_loss']/num_batches:.4f}, "
                  f"sparsity_penalty: {loader_stats['sparsity_loss']/num_batches:.4f}, "
                  f"presence_loss: {loader_stats['presence_loss']/num_batches:.4f}")
            
            loader_stats['avg_precision'] = overall_precision
            loader_stats['avg_recall'] = overall_recall
            loader_stats['avg_f1'] = overall_f1
            loader_stats['mae'] = mae_avg
            loader_stats['mse'] = mse_avg
            
            for ct in range(model.num_celltypes):
                loader_stats[f'precision_ct{ct}'] = class_precision[ct].item()
                loader_stats[f'recall_ct{ct}'] = class_recall[ct].item()
                loader_stats[f'f1_ct{ct}'] = class_f1[ct].item()
            
            for t in thresholds:
                batch_results = threshold_results[t][val_name]
                if batch_results['count'] > 0:
                    for key in ['mse', 'mae', 'detection_accuracy']:
                        batch_results[key] /= batch_results['count']
                    for key, value in batch_results.items():
                        if key != 'count':
                            loader_stats[f'thresh_{t}_{key}'] = value
            
            # Average the scalar stats
            for key in list(loader_stats.keys()):
                if key not in ['presence_stats_per_celltype', 'alpha_stats_per_celltype']:
                    loader_stats[key] /= num_batches
            
            # Average the per-cell-type stats
            for key in ['presence_stats_per_celltype', 'alpha_stats_per_celltype']:
                for subkey in loader_stats[key]:
                    for subsubkey in loader_stats[key][subkey]:
                        loader_stats[key][subkey][subsubkey] /= num_batches
            
            # Store absent indices in val_stats
            loader_stats['absent_indices'] = absent_indices
            val_stats[val_name] = dict(loader_stats)
            
            dataset_size = len(val_loader.dataset)
            total_samples += dataset_size
    
    # Compute the average validation loss as the mean of per-batch losses
    avg_val_loss = weighted_loss_sum / total_batches if total_batches > 0 else 0.0
    return avg_val_loss, val_stats

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loaders: Dict[str, DataLoader],
    model_path: str,
    num_epochs: int = 1000,
    patience: int = 20,
    lr: float = 2e-3,
    weight_decay: float = 2e-4,
    use_wandb: bool = True,
    wandb_project: str = "cfDNA-Deconvolution",
    wandb_entity: str = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[nn.Module, float]:
    model = model.to(device)
    
    # Setup optimizer with AdamW, including presence model parameters
    optimizer = optim.AdamW(list(model.parameters()) + [param for pm in model.presence_models for param in pm.parameters()], lr=lr, weight_decay=weight_decay)
    
    # Cyclic learning rate scheduler (triangular policy)
    cycle_length = 10
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-4,
        max_lr=lr,
        step_size_up=cycle_length * len(train_loader) // 2,
        mode='triangular',
        cycle_momentum=False
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
    
    # Log presence model specificity thresholds
    print("\nPresence Model Specificity Thresholds:")
    for ct in range(model.num_celltypes):
        threshold = getattr(model.presence_models[ct], 'specificity_threshold', 0.5)  # Default to 0.5 if not found
        print(f"Cell Type {ct}: Specificity Threshold = {threshold:.4f}")
        if use_wandb:
            wandb.run.summary[f"presence_specificity_threshold_ct{ct}"] = threshold
    
    # Preparation
    initial_lr = lr
    warmup_epochs = 5
    
    history = defaultdict(list)
    best_val_loss = float('inf')
    best_tcells_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    best_threshold = 0.01
    best_threshold_f1 = 0.0
    
    eval_presence_threshold = 0.01
    
    for epoch in range(num_epochs):
        print(f"\n🔹 Epoch {epoch + 1}/{num_epochs}")
        
        # LR Warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = initial_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"LR Warmup: {current_lr:.1e}")
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Cyclic LR: {current_lr:.1e}")
        
        focal_loss_weight_train = 0.01
        focal_loss_weight_val = 0.01
        
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            focal_loss_weight=focal_loss_weight_train,
            presence_threshold=eval_presence_threshold
        )
        
        if epoch >= warmup_epochs:
            scheduler.step()
        
        avg_val_loss, val_stats = validate(
            model,
            val_loaders,
            device,
            presence_threshold=eval_presence_threshold,
            focal_loss_weight=focal_loss_weight_val
        )
        
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        threshold_f1_scores = {t: 0.0 for t in thresholds}
        
        for t in thresholds:
            for val_name, stats in val_stats.items():
                threshold_key = f'thresh_{t}_detection_accuracy'
                if threshold_key in stats:
                    threshold_f1_scores[t] += stats[threshold_key]
            
            threshold_f1_scores[t] /= len(val_stats)
            
            if threshold_f1_scores[t] > best_threshold_f1:
                best_threshold_f1 = threshold_f1_scores[t]
                best_threshold = t
                print(f"New best threshold: {best_threshold} (Detection Accuracy: {best_threshold_f1:.4f})")
        
        for key, value in train_stats.items():
            history[key].append(value)
        for val_name, stats in val_stats.items():
            for k, v in stats.items():
                history[f"{val_name}/{k}"].append(v)
        
        tcells_low_f1 = val_stats.get('t-cells_low', {}).get('avg_f1', 0.0)
        
        print(f"\n🔹 Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_stats['total_loss']:.8f} | Grad Norm: {train_stats['grad_norm']:.8f}")
        
        for val_name, stats in val_stats.items():
            print(f"{val_name} Loss: {stats['loss']:.8f}")
            if 'avg_precision' in stats and 'avg_recall' in stats and 'avg_f1' in stats:
                print(f"{val_name} Detection: P={stats['avg_precision']:.4f}, "
                      f"R={stats['avg_recall']:.4f}, F1: {stats['avg_f1']:.4f}")
            # Log per-cell-type presence stats for t-cells_* and oac_* datasets
            if 't-cells' in val_name or 'oac' in val_name:
                print(f"\nPer-Cell-Type Presence Stats for {val_name}:")
                for ct in range(model.num_celltypes):
                    ct_stats = stats.get('presence_stats_per_celltype', {}).get(f'celltype_{ct}', {})
                    if ct_stats:
                        print(f"Cell Type {ct}: TP={ct_stats['true_positives']}, FP={ct_stats['false_positives']}, "
                              f"FN={ct_stats['false_negatives']}, TN={ct_stats['true_negatives']}, "
                              f"Mean Prob Present={ct_stats['mean_prob_present']:.4f}, "
                              f"Std Prob Present={ct_stats['std_prob_present']:.4f}, "
                              f"Mean Prob Absent={ct_stats['mean_prob_absent']:.4f}, "
                              f"Std Prob Absent={ct_stats['std_prob_absent']:.4f}")
                # Log alpha stats for absent cell types (determined in validate)
                absent_indices = stats.get('absent_indices', [])
                print(f"\nAlpha Stats for Absent Cell Types {absent_indices} in {val_name}:")
                for ct in absent_indices:
                    alpha_stats = stats.get('alpha_stats_per_celltype', {}).get(f'celltype_{ct}', {})
                    if alpha_stats:
                        print(f"Cell Type {ct}: Mean Alpha={alpha_stats['mean_alpha']:.8f}, "
                              f"Std Alpha={alpha_stats['std_alpha']:.8f}")
        
        if use_wandb:
            wandb_logs = {
                "epoch": epoch,
                "train/loss": train_stats['total_loss'],
                "train/grad_norm": train_stats['grad_norm'],
                "val/avg_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "focal_loss_weight_train": focal_loss_weight_train,
                "focal_loss_weight_val": focal_loss_weight_val,
                "best_threshold": best_threshold,
                "best_threshold_f1": best_threshold_f1,
                "tcells_low_f1": tcells_low_f1
            }
            for val_name, stats in val_stats.items():
                for k, v in stats.items():
                    if k == 'presence_stats_per_celltype':
                        for ct, ct_stats in v.items():
                            for stat_name, stat_value in ct_stats.items():
                                wandb_logs[f"val/{val_name}/presence_{stat_name}_ct{ct}"] = stat_value
                    elif k == 'alpha_stats_per_celltype':
                        for ct, ct_stats in v.items():
                            for stat_name, stat_value in ct_stats.items():
                                wandb_logs[f"val/{val_name}/alpha_{stat_name}_ct{ct}"] = stat_value
                    elif k == 'absent_indices':
                        # Log absent indices as a list
                        wandb_logs[f"val/{val_name}/absent_indices"] = v
                    else:
                        wandb_logs[f"val/{val_name}/{k}"] = v
            
            wandb_logs["threshold_comparison"] = wandb.Table(
                data=[[t, threshold_f1_scores[t]] for t in thresholds],
                columns=["threshold", "detection_accuracy"]
            )
            wandb.log(wandb_logs)
        
        if avg_val_loss < best_val_loss or tcells_low_f1 > best_tcells_f1:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            if tcells_low_f1 > best_tcells_f1:
                best_tcells_f1 = tcells_low_f1
            patience_counter = 0
            best_epoch = epoch
            print(f"New best model with loss: {best_val_loss:.8f}, T-cells low F1: {best_tcells_f1:.4f}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_tcells_f1': best_tcells_f1,
                'best_threshold': best_threshold,
                'history': dict(history)
            }
            torch.save(checkpoint, os.path.join(model_path, "best_model.pt"))
            if use_wandb:
                wandb.save(os.path.join(model_path, "best_model.pt"))
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
            break
    
    print("Loading best model (best overall validation loss and T-cells F1)")
    checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    plot_training_history(dict(history), os.path.join(model_path, "model_training"))
    
    print(f"\nRecommended threshold for inference: {best_threshold}")
    print(f"(Based on best detection accuracy: {best_threshold_f1:.4f})")
    
    if use_wandb:
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_tcells_f1"] = best_tcells_f1
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["total_epochs"] = epoch + 1
        wandb.run.summary["best_threshold"] = best_threshold
        wandb.run.summary["best_threshold_f1"] = best_threshold_f1
        
        def remove_wandb_hooks(model):
            for k in list(model._forward_hooks.keys()):
                model._forward_hooks.pop(k)
            for k in list(model._backward_hooks.keys()):
                model._backward_hooks.pop(k)
            for k in list(model._forward_pre_hooks.keys()):
                model._forward_hooks.pop(k)
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
         - (row2, col2): 'T-Cells F1 Score'
      3) Plots lines for any keys matching "loss", "valid", "alpha", "tcells_low_f1" in the appropriate subplot.
      4) Optionally saves the figure to an HTML file for offline viewing or logs it.

    Args:
        history (Dict[str, List[float]]):
            A dictionary where each key is a metric name and the value is a list of epoch-level measurements.
        save_path (str):
            If provided, the figure is saved to `save_path + ".html"`. Otherwise, a fig.show() might be done.
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
            'T-Cells F1 Score'
        )
    )

    # (A) Plot loss evolution
    loss_keys = [k for k in history.keys() if 'loss' in k.lower()]
    for key in loss_keys:
        fig.add_trace(
            go.Scatter(y=history[key], name=key),
            row=1, col=1
        )

    # (B) Plot valid marker ratio
    valid_keys = [k for k in history.keys() if 'valid' in k.lower()]
    for key in valid_keys:
        fig.add_trace(
            go.Scatter(y=history[key], name=key),
            row=1, col=2
        )

    # (C) Plot alpha statistics if they exist
    if 'alpha_stats/mean' in history:
        fig.add_trace(
            go.Scatter(y=history['alpha_stats/mean'], name='Mean'),
            row=2, col=1
        )
    if 'alpha_stats/std' in history:
        fig.add_trace(
            go.Scatter(y=history['alpha_stats/std'], name='Std'),
            row=2, col=1
        )

    # (D) Plot T-cells F1 score for low coverage
    if 't-cells_low/avg_f1' in history:
        fig.add_trace(
            go.Scatter(y=history['t-cells_low/avg_f1'], name='T-Cells Low F1'),
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
    fig.update_yaxes(title_text="F1 Score", row=2, col=2)

    # X-axis labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Epoch", row=i, col=j)

    # Save or show
    if save_path:
        fig.write_html(f"{save_path}.html")
    else:
        fig.show()
