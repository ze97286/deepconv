import os
from collections import defaultdict
from pprint import pprint

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, List
from deep_conv.deconvolution.loss_function import loss_fn  # NEW: Correct import
from deep_conv.benchmark.benchmark_utils import evaluate_performance
import time

def init_wandb(config, project_name="cfDNA-Deconvolution", entity=None):
    """
    Initialise Weights & Biases (wandb) logging for experiment tracking.
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
    accumulation_steps: int = 8,
    epoch: int = 0,
    presence_threshold: float = 0.01
) -> Dict[str, float]:
    """
    Train the model for one epoch with added diagnostics.
    """
    model.train()
    epoch_stats = defaultdict(float)
    timing_stats = defaultdict(float)
    num_batches = 0

    optimiser.zero_grad()

    for batch_idx, batch in enumerate(tqdm(loader, desc='Training')):
        start_batch = time.time()

        # Data loading
        start_data = time.time()
        fraction = batch['X'].to(device)
        coverage = batch['coverage'].to(device)
        x_nnls = batch['x_nnls'].to(device) if 'x_nnls' in batch else None
        y_true = batch['y'].to(device)
        presence_probs = batch['presence_probs'].to(device) if 'presence_probs' in batch else None
        timing_stats['data_loading'] += time.time() - start_data

        # Forward pass
        start_forward = time.time()
        props, batch_presence_probs, _, dl_props, reconstructed, valid_mask = model(
            fraction, coverage, x_nnls, presence_probs
        )
        timing_stats['forward_pass'] += time.time() - start_forward

        # Loss computation
        start_loss = time.time()
        loss, details = loss_fn(
            pred_props=props,
            true_props=y_true,
            presence_probs=batch_presence_probs,
            reconstructed=reconstructed,
            marker_values=fraction,
            coverage=coverage,
            valid_mask=valid_mask,
            x_nnls=x_nnls,
            dl_props=dl_props,
            combination_weight=model.combination_weight,
            presence_threshold=presence_threshold,
            low_snr_indices=[11],
            device=device
        )
        timing_stats['loss_computation'] += time.time() - start_loss

        # Metrics calculation
        start_metrics = time.time()
        mae = torch.abs(props - y_true).mean()
        mse = F.mse_loss(props, y_true)
        props_flat = props.reshape(-1)
        y_true_flat = y_true.reshape(-1)
        props_centered = props_flat - props_flat.mean()
        y_true_centered = y_true_flat - y_true_flat.mean()
        props_std = props_centered.std() + 1e-8
        y_true_std = y_true_centered.std() + 1e-8
        correlation = (props_centered * y_true_centered).mean() / (props_std * y_true_std)
        
        true_present = (y_true > presence_threshold)
        pred_present = (props > presence_threshold)
        true_positives = (pred_present & true_present).float().sum()
        false_positives = (pred_present & ~true_present).float().sum()
        false_negatives = (~pred_present & true_present).float().sum()
        true_negatives = (~pred_present & ~true_present).float().sum()
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        epoch_stats['mae'] += mae.item()
        epoch_stats['mse'] += mse.item()
        epoch_stats['correlation'] += correlation.item()
        epoch_stats['precision'] += precision.item()
        epoch_stats['recall'] += recall.item()
        epoch_stats['f1_score'] += f1_score.item()
        timing_stats['metrics_calculation'] += time.time() - start_metrics

        # Backward pass
        start_backward = time.time()
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        timing_stats['backward_pass'] += time.time() - start_backward

        # Optimizer step
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(loader)):
            start_optim = time.time()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_stats['grad_norm'] += grad_norm.item()
            optimiser.step()
            optimiser.zero_grad()
            timing_stats['optimizer_step'] += time.time() - start_optim

            # WandB logging
            if wandb.run is not None and (batch_idx + 1) % (log_interval // 2) == 0:
                start_wandb = time.time()
                wandb_log = {
                    "batch/loss": loss.item(),
                    "batch/mae": mae.item(),
                    "batch/mse": mse.item(),
                    "batch/correlation": correlation.item(),
                    "batch/precision": precision.item(),
                    "batch/recall": recall.item(),
                    "batch/f1_score": f1_score.item(),
                    "batch/grad_norm": grad_norm.item(),
                    "batch/lr": optimiser.param_groups[0]['lr'],
                    "batch/step": batch_idx
                }
                for component, value in details.items():
                    wandb_log[f"batch/component_{component}"] = value
                wandb.log(wandb_log)
                timing_stats['wandb_logging'] += time.time() - start_wandb

        # Periodic logging with diagnostics
        if batch_idx % log_interval == 0:
            start_print = time.time()
            print(f"\nBatch {batch_idx}/{len(loader)} | Loss: {loss.item():.8f}")
            print(f"MAE: {mae.item():.4f}, MSE: {mse.item():.4f}, Correlation: {correlation.item():.4f}")
            print(f"Presence Detection - P: {precision.item():.4f}, R: {recall.item():.4f}, F1: {f1_score.item():.4f}")
            print(f"Gradient Norm: {grad_norm.item() if 'grad_norm' in locals() else 0.0:.4f}")
            print(f"Loss Details: {details}") 
            print(f"Props range: {props.min().item():.4f} - {props.max().item():.4f}")  # NEW: Diagnostic
            print(f"DL Props range: {dl_props.min().item():.4f} - {dl_props.max().item():.4f}")  # NEW: Diagnostic
            print(f"Fraction range: {fraction.min().item():.4f} - {fraction.max().item():.4f}")  # NEW: Diagnostic
            print(f"Coverage range: {coverage.min().item():.4f} - {coverage.max().item():.4f}")  # NEW: Diagnostic
            print(f"Valid mask ratio: {valid_mask.float().mean().item():.4f}")  # NEW: Diagnostic
            timing_stats['printing'] += time.time() - start_print

        epoch_stats['total_loss'] += scaled_loss.item() * accumulation_steps
        timing_stats['total_batch'] += time.time() - start_batch
        num_batches += 1

    for key in epoch_stats:
        epoch_stats[key] /= num_batches
    
    print(f"\n===== Epoch {epoch + 1} Summary =====")
    print(f"Average Loss: {epoch_stats['total_loss']:.8f}")
    print(f"MAE: {epoch_stats['mae']:.4f}, MSE: {epoch_stats['mse']:.4f}, Correlation: {epoch_stats['correlation']:.4f}")
    print(f"Combination Weights: {model.get_combination_weights()}, Precision: {epoch_stats['precision']:.4f}, Recall: {epoch_stats['recall']:.4f}, F1: {epoch_stats['f1_score']:.4f}")
    return dict(epoch_stats)

def validate(
    model: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    cell_types: List[str],
    presence_threshold: float = 0.01,
    alpha_threshold: float = 1e-4
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Validate the model on multiple validation sets.
    """
    model.eval()
    val_stats = {}
    
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    threshold_results = {}
    for t in thresholds:
        threshold_results[t] = {}
        for val_name in val_loaders.keys():
            threshold_results[t][val_name] = defaultdict(float)
    
    total_batches = 0
    weighted_loss_sum = 0.0

    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            print(f"\n----- Validating {val_name} -----")
            loader_stats = defaultdict(float)
            num_batches = 0
            confusion_matrix = {
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                'by_cell_type': {}
            }
            for ct_idx, ct_name in enumerate(cell_types):
                confusion_matrix['by_cell_type'][ct_name] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
            
            all_preds = []
            all_true = []
            all_dl_props = []
            
            for batch in tqdm(val_loader, desc=f'Validating {val_name}'):
                fraction = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                x_nnls = batch['x_nnls'].to(device) if 'x_nnls' in batch else None
                y_true = batch['y'].to(device)
                presence_probs = batch['presence_probs'].to(device) if 'presence_probs' in batch else None
                
                props, batch_presence_probs, _, dl_props, reconstructed, valid_mask = model(
                    fraction, coverage, x_nnls, presence_probs
                )
                
                loss, details = loss_fn(
                    pred_props=props,
                    true_props=y_true,
                    presence_probs=batch_presence_probs,
                    reconstructed=reconstructed,
                    marker_values=fraction,
                    coverage=coverage,
                    valid_mask=valid_mask,
                    x_nnls=x_nnls,
                    dl_props=dl_props,
                    combination_weight=model.combination_weight,
                    presence_threshold=presence_threshold,
                    low_snr_indices=[11],
                    device=device
                )
                
                all_preds.append(props.cpu().numpy())
                all_true.append(y_true.cpu().numpy())
                all_dl_props.append(dl_props.cpu().numpy())
                
                batch_size = y_true.size(0)
                mae = torch.abs(props - y_true).mean()
                mse = F.mse_loss(props, y_true)
                
                true_present = (y_true > presence_threshold)
                pred_present = (props > presence_threshold)
                tp = (pred_present & true_present).float().sum().item()
                fp = (pred_present & ~true_present).float().sum().item()
                fn = (~pred_present & true_present).float().sum().item()
                tn = (~pred_present & ~true_present).float().sum().item()
                
                confusion_matrix['tp'] += tp
                confusion_matrix['fp'] += fp
                confusion_matrix['fn'] += fn
                confusion_matrix['tn'] += tn
                
                for ct_idx, ct_name in enumerate(cell_types):
                    ct_tp = (pred_present[:, ct_idx] & true_present[:, ct_idx]).float().sum().item()
                    ct_fp = (pred_present[:, ct_idx] & ~true_present[:, ct_idx]).float().sum().item()
                    ct_fn = (~pred_present[:, ct_idx] & true_present[:, ct_idx]).float().sum().item()
                    ct_tn = (~pred_present[:, ct_idx] & ~true_present[:, ct_idx]).float().sum().item()
                    confusion_matrix['by_cell_type'][ct_name]['tp'] += ct_tp
                    confusion_matrix['by_cell_type'][ct_name]['fp'] += ct_fp
                    confusion_matrix['by_cell_type'][ct_name]['fn'] += ct_fn
                    confusion_matrix['by_cell_type'][ct_name]['tn'] += ct_tn
                
                for t in thresholds:
                    batch_results = threshold_results[t][val_name]
                    thresholded_props = props.clone()
                    thresholded_props[thresholded_props < t] = 0.0
                    row_sums = thresholded_props.sum(dim=1, keepdim=True)
                    valid_rows = (row_sums > 0).squeeze(-1)
                    if valid_rows.any():
                        thresholded_props[valid_rows] = thresholded_props[valid_rows] / row_sums[valid_rows]
                    mse_t = F.mse_loss(thresholded_props, y_true)
                    mae_t = torch.abs(thresholded_props - y_true).mean()
                    pred_present_t = (thresholded_props > 0)
                    true_present_t = (y_true > presence_threshold)
                    detection_accuracy_t = (pred_present_t == true_present_t).float().mean()
                    batch_results['mse'] += mse_t.item() * batch_size
                    batch_results['mae'] += mae_t.item() * batch_size
                    batch_results['detection_accuracy'] += detection_accuracy_t.item() * batch_size
                    batch_results['count'] += batch_size
                
                loader_stats['mae_sum'] += mae.item() * batch_size
                loader_stats['mse_sum'] += mse.item() * batch_size
                loader_stats['loss'] += loss.item()
                loader_stats['samples'] += batch_size
                for key, value in details.items():
                    if isinstance(value, (int, float)):
                        loader_stats[key] += value
                    else:
                        print(f"Warning: Skipping non-numeric detail '{key}' in {val_name}: {value}")
                
                num_batches += 1
                total_batches += 1
                weighted_loss_sum += loss.item()
            
            if num_batches == 0 or not all_preds:
                print(f"Warning: No valid batches in {val_name}")
                val_stats[val_name] = {"error": "No valid batches"}
                continue
            
            all_preds_np = np.concatenate(all_preds, axis=0)
            all_true_np = np.concatenate(all_true, axis=0)
            all_dl_props_np = np.concatenate(all_dl_props, axis=0)
            
            eval_metrics = evaluate_performance(all_true_np, all_preds_np, cell_types, alpha_threshold=alpha_threshold)
            r2 = eval_metrics["Overall"].get('Global R² (Flattened)', 0.0)
            overall_mae = eval_metrics["Overall"].get('Overall MAE', 0.0)
            per_cell_r2 = {ct: eval_metrics["Per_Cell_Type"][ct].get("R²", 0.0) 
                          for ct in cell_types if ct in eval_metrics["Per_Cell_Type"]}
            
            from sklearn.metrics import r2_score
            dl_r2 = r2_score(all_true_np, all_dl_props_np, multioutput='raw_values')
            mean_dl_r2 = np.mean(dl_r2)
            print(f"\nDL Props R² for {val_name}: Mean={mean_dl_r2:.4f}, Per-Cell={dl_r2}")
            
            precision = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp'] + 1e-8)
            recall = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'] + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            print(f"\nOverall Metrics for {val_name}:")
            print(f"Global R² (Flattened): {r2:.4f}")
            print(f"Overall MAE: {overall_mae:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            print(f"\nPer-Cell-Type Metrics for {val_name}:")
            for ct in cell_types:
                if ct in ["T-cells", "OAC"] or "t-cells" in val_name.lower() or "oac" in val_name.lower():
                    if ct in per_cell_r2:
                        ct_metrics = eval_metrics["Per_Cell_Type"][ct]
                        ct_cm = confusion_matrix['by_cell_type'][ct]
                        ct_precision = ct_cm['tp'] / (ct_cm['tp'] + ct_cm['fp'] + 1e-8)
                        ct_recall = ct_cm['tp'] / (ct_cm['tp'] + ct_cm['fn'] + 1e-8)
                        ct_f1 = 2 * ct_precision * ct_recall / (ct_precision + ct_recall + 1e-8)
                        print(f"{ct}:")
                        print(f"  R²: {per_cell_r2[ct]:.4f}")
                        print(f"  MAE: {ct_metrics.get('MAE', 0.0):.4f}")
                        print(f"  Detection - P: {ct_precision:.4f}, R: {ct_recall:.4f}, F1: {ct_f1:.4f}")
            
            for t in thresholds:
                batch_results = threshold_results[t][val_name]
                if batch_results['count'] > 0:
                    for key in ['mse', 'mae', 'detection_accuracy']:
                        batch_results[key] /= batch_results['count']
                        loader_stats[f'thresh_{t}_{key}'] = batch_results[key]
            
            loader_stats['r2'] = r2
            loader_stats['mae'] = overall_mae
            loader_stats['precision'] = precision
            loader_stats['recall'] = recall
            loader_stats['f1'] = f1
            loader_stats['per_cell_r2'] = per_cell_r2
            val_stats[val_name] = dict(loader_stats)
    
    avg_val_loss = weighted_loss_sum / total_batches if total_batches > 0 else float('inf')
    print(f"\n===== Validation Complete =====")
    print(f"Average validation loss: {avg_val_loss:.6f}")
    return avg_val_loss, val_stats

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loaders: Tuple[Dict[str, DataLoader], Dict[str, DataLoader]],
    model_path: str,
    cell_types: List[str],
    num_epochs: int = 1000,
    patience: int = 20,
    lr: float = 5e-5,
    weight_decay: float = 1e-3,
    use_wandb: bool = True,
    wandb_project: str = "cfDNA-Deconvolution",
    wandb_entity: str = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[nn.Module, float]:
    """
    Train the model over multiple epochs with early stopping.
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    os.makedirs(model_path, exist_ok=True)

    if use_wandb:
        config = {
            "model_type": model.__class__.__name__,
            "num_markers": getattr(model, "num_markers", "unknown"),
            "num_cell_types": getattr(model, "num_celltypes", "unknown"),
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": train_loader.batch_size,
            "num_epochs": num_epochs,
            "patience": patience,
            "device": str(device),
            "scheduler": "CosineAnnealingLR(T_max=10)"
        }
        run = init_wandb(config, project_name=wandb_project, entity=wandb_entity)
        wandb.watch(model, log="all", log_freq=100)

    best_val_loss = float('inf')
    best_tcells_f1 = 0.0
    best_tcells_r2 = -float('inf')
    best_oac_r2 = -float('inf')
    best_tier1_r2 = -float('inf')
    best_mae = float('inf')
    patience_counter = 0
    history = defaultdict(list)

    val_loaders_unaugmented, val_loaders_augmented = val_loaders
    for epoch in range(num_epochs):
        print(f"\n🔹 Epoch {epoch + 1}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            presence_threshold=0.01
        )
        scheduler.step()

        # Use unaugmented data for early epochs, augmented for later epochs
        current_val_loaders = val_loaders_unaugmented if epoch < 30 else val_loaders_augmented
        print(f"Validation with {'augmented' if epoch >= 30 else 'unaugmented'} data")

        avg_val_loss, val_stats = validate(
            model,
            current_val_loaders,
            device,
            cell_types=cell_types,
            presence_threshold=0.01,
            alpha_threshold=0.01
        )

        # Update history
        for key, value in train_stats.items():
            history[key].append(value)
        for val_name, stats in val_stats.items():
            for k, v in stats.items():
                if k != 'per_cell_r2':
                    history[f"{val_name}/{k}"].append(v)

        # Calculate key performance metrics across datasets
        tcells_r2_sum = 0.0
        tcells_count = 0
        tcells_f1_sum = 0.0
        oac_r2_sum = 0.0
        oac_count = 0
        tier1_r2_sum = 0.0
        tier1_count = 0
        mae_sum = 0.0
        mae_count = 0

        for val_name, stats in val_stats.items():
            if 't-cells' in val_name.lower():
                if 'per_cell_r2' in stats and 'T-cells' in stats['per_cell_r2']:
                    tcells_r2_sum += stats['per_cell_r2']['T-cells']
                    tcells_count += 1
                if 'f1' in stats:
                    tcells_f1_sum += stats['f1']
            elif 'oac' in val_name.lower():
                if 'per_cell_r2' in stats and 'OAC' in stats['per_cell_r2']:
                    oac_r2_sum += stats['per_cell_r2']['OAC']
                    oac_count += 1
            elif 'tier1' in val_name.lower():
                if 'r2' in stats:
                    tier1_r2_sum += stats['r2']
                    tier1_count += 1
            if 'mae' in stats:
                mae_sum += stats['mae']
                mae_count += 1

        tcells_r2_avg = tcells_r2_sum / tcells_count if tcells_count > 0 else 0.0
        tcells_f1_avg = tcells_f1_sum / tcells_count if tcells_count > 0 else 0.0
        oac_r2_avg = oac_r2_sum / oac_count if oac_count > 0 else 0.0
        tier1_r2_avg = tier1_r2_sum / tier1_count if tier1_count > 0 else 0.0
        mae_avg = mae_sum / mae_count if mae_count > 0 else float('inf')

        print(f"\n🔹 Performance Summary:")
        print(f"Average MAE: {mae_avg:.4f}")
        print(f"Average T-cells R²: {tcells_r2_avg:.4f}")
        print(f"Average T-cells F1: {tcells_f1_avg:.4f}")
        print(f"Average OAC R²: {oac_r2_avg:.4f}")
        print(f"Average Tier1 R²: {tier1_r2_avg:.4f}")

        if use_wandb:
            wandb_logs = {
                "epoch": epoch,
                "train/loss": train_stats['total_loss'],
                "val/avg_loss": avg_val_loss,
                "val/tcells_r2_avg": tcells_r2_avg,
                "val/tcells_f1_avg": tcells_f1_avg,
                "val/oac_r2_avg": oac_r2_avg,
                "val/tier1_r2_avg": tier1_r2_avg,
                "val/mae_avg": mae_avg,
                "lr": optimizer.param_groups[0]['lr']
            }
            for val_name, stats in val_stats.items():
                for k, v in stats.items():
                    if k == 'per_cell_r2':
                        for cell_type, r2 in v.items():
                            wandb_logs[f"val/{val_name}/r2_{cell_type}"] = r2
                    elif isinstance(v, (int, float)):
                        wandb_logs[f"val/{val_name}/{k}"] = v
            wandb.log(wandb_logs)

        improved = False
        improvement_reason = []
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True
            improvement_reason.append(f"loss: {best_val_loss:.6f}")
        
        if tcells_f1_avg > best_tcells_f1:
            best_tcells_f1 = tcells_f1_avg
            improved = True
            improvement_reason.append(f"T-cells F1: {best_tcells_f1:.4f}")
        
        if tcells_r2_avg > best_tcells_r2:
            best_tcells_r2 = tcells_r2_avg
            improved = True
            improvement_reason.append(f"T-cells R²: {best_tcells_r2:.4f}")
        
        if oac_r2_avg > best_oac_r2:
            best_oac_r2 = oac_r2_avg
            improved = True
            improvement_reason.append(f"OAC R²: {best_oac_r2:.4f}")
            
        if tier1_r2_avg > best_tier1_r2:
            best_tier1_r2 = tier1_r2_avg
            improved = True
            improvement_reason.append(f"Tier1 R²: {best_tier1_r2:.4f}")
            
        if mae_avg < best_mae:
            best_mae = mae_avg
            improved = True
            improvement_reason.append(f"MAE: {best_mae:.4f}")

        if improved:
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_tcells_f1': best_tcells_f1,
                'best_tcells_r2': best_tcells_r2,
                'best_oac_r2': best_oac_r2,
                'best_tier1_r2': best_tier1_r2,
                'best_mae': best_mae,
                'history': dict(history)
            }
            torch.save(checkpoint, os.path.join(model_path, "best_model.pt"))
            
            print(f"\n✅ Saved new best model with improvements in: {', '.join(improvement_reason)}")
            
            if use_wandb:
                wandb.save(os.path.join(model_path, "best_model.pt"))
        else:
            patience_counter += 1
            print(f"\nNo improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
            break

    print("\nLoading best model...")
    checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n🏆 Best Model Performance:")
    print(f"Loss: {best_val_loss:.6f}")
    print(f"T-cells F1: {best_tcells_f1:.4f}")
    print(f"T-cells R²: {best_tcells_r2:.4f}")
    print(f"OAC R²: {best_oac_r2:.4f}")
    print(f"Tier1 R²: {best_tier1_r2:.4f}")
    print(f"MAE: {best_mae:.4f}")

    if use_wandb:
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_tcells_f1"] = best_tcells_f1
        wandb.run.summary["best_tcells_r2"] = best_tcells_r2
        wandb.run.summary["best_oac_r2"] = best_oac_r2
        wandb.run.summary["best_tier1_r2"] = best_tier1_r2
        wandb.run.summary["best_mae"] = best_mae
        wandb.finish()

    return model, 0.01

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
