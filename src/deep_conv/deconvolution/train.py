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
from deep_conv.deconvolution.loss import loss_fn
from deep_conv.benchmark.benchmark_utils import evaluate_performance

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
    accumulation_steps: int = 8,
    epoch: int = 0,
    focal_loss_weight: float = 0.0,
    presence_threshold: float = 0.01,
    log_vars: Dict = None,
    k: float = 0.1,
) -> Dict[str, float]:
    model.train()
    epoch_stats = defaultdict(float)
    timing_stats = defaultdict(float)
    num_batches = 0

    # Initialize log variance parameters if not provided
    if log_vars is None:
        log_vars = {
            'mae': torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True),
            'corr': torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True),
            'presence': torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True),
            'sparsity': torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        }

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs')
    ) as prof:
        start_epoch = time.time()
        optimiser.zero_grad()

        for batch_idx, batch in enumerate(tqdm(loader, desc='Training')):
            start_batch = time.time()
            
            # 1. Data loading and transfer
            start_data = time.time()
            with torch.profiler.record_function("data_loading"):
                fraction = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                y_true = batch['y'].to(device)
            timing_stats['data_loading'] += time.time() - start_data
            
            # 2. Forward pass
            start_forward = time.time()
            with torch.profiler.record_function("forward_pass"):
                alpha, reconstructed, valid_mask, presence_probs, presence_logits, x_nnls = model(fraction, coverage)
            timing_stats['forward_pass'] += time.time() - start_forward
            
            # Determine target cell indices
            start_target = time.time()
            target_cell_types = None
            if hasattr(loader.dataset, 'name'):
                dataset_name = loader.dataset.name.lower() if hasattr(loader.dataset, 'name') else ""
            else:
                dataset_name = ""
                
            if "t-cells" in dataset_name:
                target_cell_types = [11]  # T-cells index
            elif "oac" in dataset_name:
                target_cell_types = [9]   # OAC index
            timing_stats['target_detection'] += time.time() - start_target
            
            # 3. Loss computation
            start_loss = time.time()
            with torch.profiler.record_function("loss_computation"):
                loss, details, log_vars['mae'], log_vars['corr'], log_vars['presence'], log_vars['sparsity'] = loss_fn(
                    pred_props=alpha,
                    true_props=y_true,
                    reconstructed=reconstructed,
                    marker_values=fraction,
                    coverage=coverage,
                    valid_mask=valid_mask,
                    presence_probs=presence_probs,
                    presence_logits=presence_logits,
                    focal_loss_weight=focal_loss_weight,
                    presence_threshold=presence_threshold,
                    compute_diagnostics=False,
                    target_cell_indices=target_cell_types,
                    log_vars=log_vars,
                    x_nnls=x_nnls,
                    k=k,
                )
            timing_stats['loss_computation'] += time.time() - start_loss
            
            # Add detailed loss timing if available
            if 'timing' in details:
                for key, value in details['timing'].items():
                    timing_stats[f'loss_{key}'] += value
            
            # 4. Metrics calculation
            start_metrics = time.time()
            mae = torch.abs(alpha - y_true).mean()
            mse = F.mse_loss(alpha, y_true)
            epoch_stats['mae'] += mae.item()
            epoch_stats['mse'] += mse.item()
            timing_stats['metrics_calculation'] += time.time() - start_metrics
            
            # 5. Backward pass
            start_backward = time.time()
            scaled_loss = loss / accumulation_steps
            with torch.profiler.record_function("backward_pass"):
                scaled_loss.backward()
            timing_stats['backward_pass'] += time.time() - start_backward
            
            # 6. Optimizer step (if applicable)
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(loader)):
                start_optim = time.time()
                with torch.profiler.record_function("optimizer_step"):
                    # Apply gradient clipping
                    start_clip = time.time()
                    all_params = list(model.parameters()) + [
                        log_vars['mae'], log_vars['corr'], 
                        log_vars['presence'], log_vars['sparsity']
                    ]
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    timing_stats['gradient_clipping'] += time.time() - start_clip
                    
                    # Actual optimizer step
                    start_step = time.time()
                    optimiser.step()
                    timing_stats['optimizer_step'] += time.time() - start_step
                    
                    # Zero gradients
                    start_zero = time.time()
                    optimiser.zero_grad()
                    timing_stats['zero_grad'] += time.time() - start_zero
                
                timing_stats['total_optimizer'] += time.time() - start_optim
                epoch_stats['grad_norm'] += grad_norm.item()
                
                # 7. Logging to wandb (if applicable)
                if wandb.run is not None and (batch_idx + 1) % (log_interval // 2) == 0:
                    start_wandb = time.time()
                    wandb_log = {
                        "batch/loss": loss.item(),
                        "batch/grad_norm": grad_norm.item(),
                        "batch/lr": optimiser.param_groups[0]['lr'],
                        "batch/step": batch_idx
                    }
                    
                    # Log task weights if available
                    if details.get('specialized_loss', False) and 'task_weights' in details:
                        tw = details['task_weights']
                        wandb_log.update({
                            "batch/weight_mae": tw['mae'],
                            "batch/weight_corr": tw['corr'],
                            "batch/weight_presence": tw['presence'],
                            "batch/weight_sparsity": tw['sparsity']
                        })
                    
                    wandb.log(wandb_log)
                    timing_stats['wandb_logging'] += time.time() - start_wandb
            
            # 8. Printing (if applicable)
            if batch_idx % log_interval == 0:
                start_print = time.time()
                print(f"\nBatch {batch_idx} | Loss: {loss.item():.8f}")
                print(f"Batch {batch_idx} | Proportion Accuracy - MAE: {mae.item():.4f}, MSE: {mse.item():.4f}")
                
                # Log task weights if using specialized loss
                if details.get('specialized_loss', False) and 'task_weights' in details:
                    tw = details['task_weights']
                    print(f"Task weights: MAE={tw['mae']:.4f}, Corr={tw['corr']:.4f}, "
                          f"Presence={tw['presence']:.4f}, Sparsity={tw['sparsity']:.4f}")
                
                # Print timing information
                print("\nTiming Information (seconds per batch):")
                for key, value in sorted(timing_stats.items()):
                    print(f"  {key}: {value/max(1, batch_idx):.6f}")
                timing_stats['printing'] += time.time() - start_print
            
            # Store total batch time
            epoch_stats['total_loss'] += scaled_loss.item() * accumulation_steps
            timing_stats['total_batch'] += time.time() - start_batch
            num_batches += 1
            
            prof.step()
    
    # Average stats
    for key in epoch_stats:
        epoch_stats[key] /= num_batches
    
    for key in timing_stats:
        timing_stats[key] /= num_batches
        epoch_stats[f'time_{key}'] = timing_stats[key]
    
    # Final epoch summary
    print(f"\nEpoch {epoch + 1} | Average Loss: {epoch_stats['total_loss']:.8f}")
    print(f"Epoch {epoch + 1} | Average Proportion Accuracy - MAE: {epoch_stats['mae']:.4f}, MSE: {epoch_stats['mse']:.4f}")
    
    print("\nAverage Timing Information (seconds per batch):")
    for key, value in sorted(timing_stats.items()):
        print(f"  {key}: {value:.6f}")
    
    epoch_stats['time_total_epoch'] = (time.time() - start_epoch) / num_batches
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    return dict(epoch_stats)

def validate(
    model: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    cell_types: List[str],
    presence_threshold: float = 0.01,
    focal_loss_weight: float = 0.0,
    alpha_threshold: float = 1e-4,
    log_vars: Dict = None,
    k: float = 0.1,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    model.eval()

    # Initialize log variance parameters if not provided
    if log_vars is None:
        log_vars = {
            'mae': torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True),
            'corr': torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True),
            'presence': torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True),
            'sparsity': torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        }

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

            # Collect predictions for R² and MAE computation
            all_preds = []
            all_true = []

            for t in thresholds:
                threshold_results[t][val_name] = {
                    'mse': 0.0,
                    'mae': 0.0,
                    'detection_accuracy': 0.0,
                    'count': 0
                }

            for batch in tqdm(val_loader, desc=f'Validating {val_name}'):
                fraction = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                y_true = batch['y'].to(device)

                alpha, reconstructed, valid_mask, presence_probs, presence_logits, x_nnls = model(fraction, coverage)

                # Determine target cell indices for specialized datasets
                target_cell_types = None
                if "t-cells" in val_name.lower():
                    target_cell_types = [11]  # T-cells index
                elif "oac" in val_name.lower():
                    target_cell_types = [9]    # OAC index

                # Use our enhanced loss function with log_vars
                loss, details, _, _, _, _ = loss_fn(
                    pred_props=alpha,
                    true_props=y_true,
                    reconstructed=reconstructed,
                    marker_values=fraction,
                    coverage=coverage,
                    valid_mask=valid_mask,
                    presence_probs=presence_probs,
                    presence_logits=presence_logits,
                    presence_threshold=presence_threshold,
                    focal_loss_weight=focal_loss_weight,
                    target_cell_indices=target_cell_types,
                    log_vars=log_vars,
                    x_nnls = x_nnls,
                    k=k,
                )

                # Collect predictions for R² and MAE
                all_preds.append(alpha.cpu().numpy())
                all_true.append(y_true.cpu().numpy())

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

                    thresholded_alpha = alpha.clone()
                    thresholded_alpha[thresholded_alpha < 1e-4] = 0.0

                    row_sums = thresholded_alpha.sum(dim=1, keepdim=True)
                    valid_rows = (row_sums > 0).squeeze(-1)
                    if valid_rows.any():
                        thresholded_alpha[valid_rows] /= row_sums[valid_rows]

                    mse = F.mse_loss(thresholded_alpha, y_true)
                    mae = torch.abs(thresholded_alpha - y_true).mean()

                    pred_present_t = (thresholded_alpha > 0)
                    true_present_t = (y_true > presence_threshold)
                    detection_accuracy = (pred_present_t == true_present_t).float().mean()

                    batch_results['mse'] += mse.item() * batch_size
                    batch_results['mae'] += mae.item() * batch_size
                    batch_results['detection_accuracy'] += detection_accuracy.item() * batch_size
                    batch_results['count'] += batch_size

                # Process loss details, including our new fields
                loader_stats['loss'] += loss.item()
                
                # Process standard fields from details
                for key, value in details.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            loader_stats[f"{key}/{subkey}"] += subvalue
                    else:
                        loader_stats[key] += value

                num_batches += 1
                total_batches += 1
                weighted_loss_sum += loss.item()

            # Compute R² and MAE for the dataset using evaluate_performance
            all_preds = np.concatenate(all_preds, axis=0)
            all_true = np.concatenate(all_true, axis=0)

            # Debug: Compare prediction statistics with deepconv_estimate
            print(f"Validate predictions for {val_name}: min={all_preds.min():.6f}, max={all_preds.max():.6f}, mean={all_preds.mean():.6f}")

            eval_metrics = evaluate_performance(
                all_true,
                all_preds,
                cell_types,
                alpha_threshold=alpha_threshold
            )

            global_metrics = eval_metrics["Overall"]
            pprint(global_metrics)
            r2 = eval_metrics["Overall"].get('Global R² (Flattened)', 0.0)
            overall_mae = eval_metrics["Overall"].get('Overall MAE', 0.0)

            # Extract per-cell-type R² values
            per_cell_r2 = {}
            for cell_type in cell_types:
                if cell_type in eval_metrics["Per_Cell_Type"]:
                    per_cell_r2[cell_type] = eval_metrics["Per_Cell_Type"][cell_type].get("R²", 0.0)

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
            print(f"R²: {r2:.4f}")
            print(f"Overall MAE (from evaluate_performance): {overall_mae:.4f}")
            
            # Display loss components
            print(f"Average Loss Components:")
            for component in ['loss_props', 'recon_loss', 'sparsity_loss', 'presence_loss', 'corr_loss']:
                if component in loader_stats:
                    print(f"  {component}: {loader_stats[component]/num_batches:.4f}")
            
            # Display task weights if available
            if 'task_weights/mae' in loader_stats:
                print(f"Task weights:")
                print(f"  MAE: {loader_stats['task_weights/mae']/num_batches:.4f}")
                print(f"  Corr: {loader_stats['task_weights/corr']/num_batches:.4f}")
                print(f"  Presence: {loader_stats['task_weights/presence']/num_batches:.4f}")
                print(f"  Sparsity: {loader_stats['task_weights/sparsity']/num_batches:.4f}")

            # Store metrics in loader_stats
            loader_stats['avg_precision'] = overall_precision
            loader_stats['avg_recall'] = overall_recall
            loader_stats['avg_f1'] = overall_f1
            loader_stats['mae'] = overall_mae
            loader_stats['mae_avg'] = mae_avg
            loader_stats['mse'] = mse_avg
            loader_stats['r2'] = r2
            loader_stats['per_cell_r2'] = per_cell_r2

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

            # Normalize all non-dictionary values by num_batches
            for key in loader_stats:
                if key != 'per_cell_r2' and not isinstance(loader_stats[key], dict):
                    loader_stats[key] /= num_batches

            val_stats[val_name] = dict(loader_stats)

            dataset_size = len(val_loader.dataset)
            total_samples += dataset_size

    avg_val_loss = weighted_loss_sum / total_batches if total_batches > 0 else 0.0
    return avg_val_loss, val_stats


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loaders: Tuple[Dict[str, DataLoader], Dict[str, DataLoader]],
    model_path: str,
    cell_types: List[str],
    num_epochs: int = 1000,
    patience: int = 20,
    lr: float = 2e-3,
    weight_decay: float = 1e-3,
    use_wandb: bool = True,
    wandb_project: str = "cfDNA-Deconvolution",
    wandb_entity: str = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    A: np.ndarray = None, 
    k: float = 0.1 
) -> Tuple[nn.Module, float]:
    model = model.to(device)
    
    # Initialise log variance parameters with appropriate initial values
    # Setting higher values for presence and sparsity indicates lower initial importance
    log_vars = {
        'mae': torch.nn.Parameter(torch.tensor(0.0, device=device)),      # exp(-0.0) = 1.0 weight
        'corr': torch.nn.Parameter(torch.tensor(0.0, device=device)),     # exp(-0.0) = 1.0 weight
        'presence': torch.nn.Parameter(torch.tensor(1.0, device=device)), # exp(-1.0) ≈ 0.37 weight
        'sparsity': torch.nn.Parameter(torch.tensor(1.0, device=device))  # exp(-1.0) ≈ 0.37 weight
    }
    
    # Modified optimizer with fixed learning rates
    optimizer = optim.AdamW([
        {'params': list(model.parameters()) + [param for pm in model.presence_models for param in pm.parameters()], 'lr': 1e-4},
        {'params': [log_vars['mae'], log_vars['corr'], log_vars['presence'], log_vars['sparsity']], 'lr': 0.01}
    ], weight_decay=weight_decay)
    
    # Use CosineAnnealingLR instead of CyclicLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    os.makedirs(model_path, exist_ok=True)
    
    if use_wandb:
        config = {
            "model_type": model.__class__.__name__,
            "num_markers": getattr(model, "num_markers", "unknown"),
            "num_cell_types": getattr(model, "num_celltypes", "unknown"),
            "feature_dim": getattr(model, "feature_dim", "unknown"),
            "learning_rate_model": 1e-4,
            "learning_rate_logvars": 0.01,
            "weight_decay": weight_decay,
            "batch_size": train_loader.batch_size if hasattr(train_loader, "batch_size") else "unknown",
            "num_epochs": num_epochs,
            "patience": patience,
            "device": str(device),
            "using_dynamic_weighting": True,
            "scheduler": "CosineAnnealingLR(T_max=10)"
        }
        run = init_wandb(config, project_name=wandb_project, entity=wandb_entity)
        wandb.watch(model, log="all", log_freq=100)
    
    print("\nPresence Model Specificity Thresholds:")
    for ct in range(model.num_celltypes):
        threshold = getattr(model.presence_models[ct], 'specificity_threshold', 0.5)
        print(f"Cell Type {ct}: Specificity Threshold = {threshold:.4f}")
        if use_wandb:
            wandb.run.summary[f"presence_specificity_threshold_ct{ct}"] = threshold
    
    history = defaultdict(list)
    best_val_loss = float('inf')
    best_tcells_f1 = 0.0
    best_tcells_r2 = -float('inf')
    best_oac_r2 = -float('inf')
    best_tier1_r2 = -float('inf')
    best_mae_avg = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    best_threshold = 0.01
    best_threshold_f1 = 0.0
    
    eval_presence_threshold = 0.01
    
    val_loaders_unaugmented, val_loaders_augmented = val_loaders
    for epoch in range(num_epochs):
        print(f"\n🔹 Epoch {epoch + 1}/{num_epochs}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.1e} (model), {optimizer.param_groups[1]['lr']:.1e} (log_vars)")
        
        focal_loss_weight_train = 0.005
        focal_loss_weight_val = 0.01
        
        # Run training epoch with log_vars
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            focal_loss_weight=focal_loss_weight_train,
            presence_threshold=eval_presence_threshold,
            log_vars=log_vars,
            A=A,
            k=k,
        )
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        # Curriculum training: Use unaugmented for first 10 epochs, then switch to augmented
        current_val_loaders = val_loaders_unaugmented if epoch < 10 else val_loaders_augmented
        print(f"Validation with augmentation: {epoch >= 10}")

        # Run validation with log_vars
        avg_val_loss, val_stats = validate(
            model,
            current_val_loaders,
            device,
            cell_types=cell_types,
            presence_threshold=eval_presence_threshold,
            focal_loss_weight=focal_loss_weight_val,
            alpha_threshold=eval_presence_threshold,
            log_vars=log_vars
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
        
        # Get current task weights
        task_weights = {
            'mae': torch.clamp(torch.exp(-log_vars['mae']), max=1.0).item(),
            'corr': torch.clamp(torch.exp(-log_vars['corr']), max=1.0).item(),
            'presence': torch.clamp(torch.exp(-log_vars['presence']), max=1.0).item(),
            'sparsity': torch.clamp(torch.exp(-log_vars['sparsity']), max=1.0).item()
        }
        
        print(f"Task weights: MAE={task_weights['mae']:.4f}, Corr={task_weights['corr']:.4f}, "
              f"Presence={task_weights['presence']:.4f}, Sparsity={task_weights['sparsity']:.4f}")
        
        # Update history
        for key, value in train_stats.items():
            history[key].append(value)
        for val_name, stats in val_stats.items():
            for k, v in stats.items():
                history[f"{val_name}/{k}"].append(v)
        
        # Record task weights in history
        for k, v in task_weights.items():
            history[f"task_weight/{k}"].append(v)
        
        tcells_low_f1 = val_stats.get('t-cells_low', {}).get('avg_f1', 0.0)
        
        tcells_r2_sum = 0.0
        tcells_count = 0
        oac_r2_sum = 0.0
        oac_count = 0
        tier1_r2_sum = 0.0
        tier1_count = 0
        mae_sum = 0.0
        mae_count = 0

        for val_name in val_stats.keys():
            if 't-cells' in val_name:
                if 'per_cell_r2' in val_stats[val_name] and 'T-cells' in val_stats[val_name]['per_cell_r2']:
                    tcells_r2_sum += val_stats[val_name]['per_cell_r2']['T-cells']
                    tcells_count += 1
            elif 'oac' in val_name:
                if 'per_cell_r2' in val_stats[val_name] and 'OAC' in val_stats[val_name]['per_cell_r2']:
                    oac_r2_sum += val_stats[val_name]['per_cell_r2']['OAC']
                    oac_count += 1
            elif 'tier1' in val_name:
                if 'r2' in val_stats[val_name]:
                    tier1_r2_sum += val_stats[val_name]['r2']
                    tier1_count += 1
            
            if 'mae' in val_stats[val_name]:
                mae_sum += val_stats[val_name]['mae']
                mae_count += 1

        tcells_r2_avg = tcells_r2_sum / tcells_count if tcells_count > 0 else 0.0
        oac_r2_avg = oac_r2_sum / oac_count if oac_count > 0 else 0.0
        tier1_r2_avg = tier1_r2_sum / tier1_count if tier1_count > 0 else 0.0
        mae_avg = mae_sum / mae_count if mae_count > 0 else 0.0

        print(f"\n🔹 Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_stats['total_loss']:.8f} | Grad Norm: {train_stats['grad_norm']:.8f}")
        print(f"Average T-cells R² (t-cells_*): {tcells_r2_avg:.4f}")
        print(f"Average OAC R² (oac_*): {oac_r2_avg:.4f}")
        print(f"Average Tier1 R² (tier1_*): {tier1_r2_avg:.4f}")
        print(f"Average Validation MAE: {mae_avg:.4f}")
        
        for val_name, stats in val_stats.items():
            print(f"{val_name} Loss: {stats['loss']:.8f}")
            if 'avg_precision' in stats and 'avg_recall' in stats and 'avg_f1' in stats:
                print(f"{val_name} Detection: P={stats['avg_precision']:.4f}, "
                      f"R={stats['avg_recall']:.4f}, F1: {stats['avg_f1']:.4f}")
            if 'r2' in stats:
                print(f"{val_name} Global R²: {stats['r2']:.4f}")
            if 'per_cell_r2' in stats:
                for cell_type, cell_r2 in stats['per_cell_r2'].items():
                    print(f"{val_name} {cell_type} R²: {cell_r2:.4f}")
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
                "val/tcells_r2_avg": tcells_r2_avg,
                "val/oac_r2_avg": oac_r2_avg,
                "val/tier1_r2_avg": tier1_r2_avg,
                "val/avg_mae": mae_avg,
                "lr/model": optimizer.param_groups[0]['lr'],
                "lr/log_vars": optimizer.param_groups[1]['lr'],
                "focal_loss_weight_train": focal_loss_weight_train,
                "focal_loss_weight_val": focal_loss_weight_val,
                "best_threshold": best_threshold,
                "best_threshold_f1": best_threshold_f1,
                "tcells_low_f1": tcells_low_f1,
                "task_weight/mae": task_weights['mae'],
                "task_weight/corr": task_weights['corr'],
                "task_weight/presence": task_weights['presence'],
                "task_weight/sparsity": task_weights['sparsity'],
                "log_var/mae": log_vars['mae'].item(),
                "log_var/corr": log_vars['corr'].item(),
                "log_var/presence": log_vars['presence'].item(),
                "log_var/sparsity": log_vars['sparsity'].item()
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
                        wandb_logs[f"val/{val_name}/absent_indices"] = v
                    elif k == 'per_cell_r2':
                        for cell_type, cell_r2 in v.items():
                            wandb_logs[f"val/{val_name}/r2_{cell_type}"] = cell_r2
                    else:
                        wandb_logs[f"val/{val_name}/{k}"] = v
            
            wandb_logs["threshold_comparison"] = wandb.Table(
                data=[[t, threshold_f1_scores[t]] for t in thresholds],
                columns=["threshold", "detection_accuracy"]
            )
            wandb.log(wandb_logs)
        
        tcells_r2_degradation = best_tcells_r2 - tcells_r2_avg
        oac_r2_degradation = best_oac_r2 - oac_r2_avg
        tier1_r2_degradation = best_tier1_r2 - tier1_r2_avg
        mae_increase = mae_avg - best_mae_avg

        if (tcells_r2_avg > best_tcells_r2 or oac_r2_avg > best_oac_r2 or 
            tier1_r2_avg > best_tier1_r2 or mae_avg < best_mae_avg or 
            tcells_low_f1 > best_tcells_f1):
            if tcells_r2_avg > best_tcells_r2:
                best_tcells_r2 = tcells_r2_avg
            if oac_r2_avg > best_oac_r2:
                best_oac_r2 = oac_r2_avg
            if tier1_r2_avg > best_tier1_r2:
                best_tier1_r2 = tier1_r2_avg
            if mae_avg < best_mae_avg:
                best_mae_avg = mae_avg
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            if tcells_low_f1 > best_tcells_f1:
                best_tcells_f1 = tcells_low_f1
            patience_counter = 0
            best_epoch = epoch
            print(f"New best model with loss: {best_val_loss:.8f}, "
                  f"T-cells low F1: {best_tcells_f1:.4f}, "
                  f"T-cells R²: {best_tcells_r2:.4f}, "
                  f"OAC R²: {best_oac_r2:.4f}, "
                  f"Tier1 R²: {best_tier1_r2:.4f}, "
                  f"Avg MAE: {best_mae_avg:.4f}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_tcells_f1': best_tcells_f1,
                'best_tcells_r2': best_tcells_r2,
                'best_oac_r2': best_oac_r2,
                'best_tier1_r2': best_tier1_r2,
                'best_mae_avg': best_mae_avg,
                'best_threshold': best_threshold,
                'history': dict(history),
                'log_vars': {k: v.data for k, v in log_vars.items()}  # Save log vars
            }
            torch.save(checkpoint, os.path.join(model_path, "best_model.pt"))
            if use_wandb:
                wandb.save(os.path.join(model_path, "best_model.pt"))
        else:
            if (tcells_r2_degradation > 0.1 or oac_r2_degradation > 0.1 or 
                tier1_r2_degradation > 0.1 or mae_increase > 0.01):
                patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs due to metric degradation")
            break
    
    print("Loading best model (best dataset-specific metrics)")
    checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore log vars from checkpoint if available
    if 'log_vars' in checkpoint:
        for k, v in checkpoint['log_vars'].items():
            log_vars[k].data = v
    
    plot_training_history(dict(history), os.path.join(model_path, "model_training"))
    
    print(f"\nRecommended threshold for inference: {best_threshold}")
    print(f"(Based on best detection accuracy: {best_threshold_f1:.4f})")
    
    # Print final task weights
    final_task_weights = {
        'mae': torch.clamp(torch.exp(-log_vars['mae']), max=1.0).item(),
        'corr': torch.clamp(torch.exp(-log_vars['corr']), max=1.0).item(),
        'presence': torch.clamp(torch.exp(-log_vars['presence']), max=1.0).item(),
        'sparsity': torch.clamp(torch.exp(-log_vars['sparsity']), max=1.0).item()
    }
    print(f"Final task weights: MAE={final_task_weights['mae']:.4f}, Corr={final_task_weights['corr']:.4f}, "
          f"Presence={final_task_weights['presence']:.4f}, Sparsity={final_task_weights['sparsity']:.4f}")
    
    if use_wandb:
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_tcells_f1"] = best_tcells_f1
        wandb.run.summary["best_tcells_r2"] = best_tcells_r2
        wandb.run.summary["best_oac_r2"] = best_oac_r2
        wandb.run.summary["best_tier1_r2"] = best_tier1_r2
        wandb.run.summary["best_mae_avg"] = best_mae_avg
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["total_epochs"] = epoch + 1
        wandb.run.summary["best_threshold"] = best_threshold
        wandb.run.summary["best_threshold_f1"] = best_threshold_f1
        
        # Log final task weights
        for k, v in final_task_weights.items():
            wandb.run.summary[f"final_task_weight_{k}"] = v
        
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
