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
    presence_threshold: float = 0.01
) -> Dict[str, float]:
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
        x_nnls = batch['x_nnls'].to(device)
        y_true = batch['y'].to(device)
        presence_probs = batch['presence_probs'].to(device)
        timing_stats['data_loading'] += time.time() - start_data

        # Forward pass
        start_forward = time.time()
        props, presence_probs, _ = model(fraction, coverage, x_nnls, presence_probs)
        timing_stats['forward_pass'] += time.time() - start_forward

        # Loss computation
        start_loss = time.time()
        loss, _ = loss_fn(
            pred_props=props,
            true_props=y_true,
            presence_probs=presence_probs,
            coverage=coverage,
            x_nnls=x_nnls,
            presence_threshold=presence_threshold,
            device=device
        )
        timing_stats['loss_computation'] += time.time() - start_loss

        # Metrics calculation
        start_metrics = time.time()
        mae = torch.abs(props - y_true).mean()
        mse = F.mse_loss(props, y_true)
        epoch_stats['mae'] += mae.item()
        epoch_stats['mse'] += mse.item()
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
            optimiser.step()
            optimiser.zero_grad()
            timing_stats['total_optimizer'] += time.time() - start_optim
            epoch_stats['grad_norm'] += grad_norm.item()

            # Logging to wandb
            if wandb.run is not None and (batch_idx + 1) % (log_interval // 2) == 0:
                start_wandb = time.time()
                wandb_log = {
                    "batch/loss": loss.item(),
                    "batch/grad_norm": grad_norm.item(),
                    "batch/lr": optimiser.param_groups[0]['lr'],
                    "batch/step": batch_idx
                }
                wandb.log(wandb_log)
                timing_stats['wandb_logging'] += time.time() - start_wandb

        # Printing
        if batch_idx % log_interval == 0:
            start_print = time.time()
            print(f"\nBatch {batch_idx} | Loss: {loss.item():.8f}")
            print(f"Batch {batch_idx} | Proportion Accuracy - MAE: {mae.item():.4f}, MSE: {mse.item():.4f}")
            timing_stats['printing'] += time.time() - start_print

        epoch_stats['total_loss'] += scaled_loss.item() * accumulation_steps
        timing_stats['total_batch'] += time.time() - start_batch
        num_batches += 1

    # Average stats
    for key in epoch_stats:
        epoch_stats[key] /= num_batches
    for key in timing_stats:
        epoch_stats[f'time_{key}'] = timing_stats[key] / num_batches

    print(f"\nEpoch {epoch + 1} | Average Loss: {epoch_stats['total_loss']:.8f}")
    print(f"Epoch {epoch + 1} | Average Proportion Accuracy - MAE: {epoch_stats['mae']:.4f}, MSE: {epoch_stats['mse']:.4f}")

    return dict(epoch_stats)

def validate(
    model: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    cell_types: List[str],
    presence_threshold: float = 0.01,
    alpha_threshold: float = 1e-4
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    model.eval()
    val_stats = {}
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    threshold_results = {t: {} for t in thresholds}
    total_batches = 0
    weighted_loss_sum = 0.0

    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            loader_stats = defaultdict(float)
            num_batches = 0
            all_preds = []
            all_true = []

            for batch in tqdm(val_loader, desc=f'Validating {val_name}'):
                fraction = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                x_nnls = batch['x_nnls'].to(device)
                y_true = batch['y'].to(device)
                presence_probs = batch['presence_probs'].to(device)

                props, presence_probs, _ = model(fraction, coverage, x_nnls, presence_probs)

                loss, details = loss_fn(
                    pred_props=props,
                    true_props=y_true,
                    presence_probs=presence_probs,
                    coverage=coverage,
                    x_nnls=x_nnls,
                    presence_threshold=presence_threshold,
                    device=device
                )

                all_preds.append(props.cpu().numpy())
                all_true.append(y_true.cpu().numpy())

                mae = torch.abs(props - y_true).mean()
                mse = F.mse_loss(props, y_true)
                batch_size = y_true.size(0)
                loader_stats['mae'] += mae.item() * batch_size
                loader_stats['mse'] += mse.item() * batch_size

                for t in thresholds:
                    if t not in threshold_results[t][val_name]:
                        threshold_results[t][val_name] = defaultdict(float)
                    batch_results = threshold_results[t][val_name]

                    thresholded_props = props.clone()
                    thresholded_props[thresholded_props < 1e-4] = 0.0
                    row_sums = thresholded_props.sum(dim=1, keepdim=True)
                    valid_rows = (row_sums > 0).squeeze(-1)
                    if valid_rows.any():
                        thresholded_props[valid_rows] /= row_sums[valid_rows]

                    mse_t = F.mse_loss(thresholded_props, y_true)
                    mae_t = torch.abs(thresholded_props - y_true).mean()
                    pred_present_t = (thresholded_props > 0)
                    true_present_t = (y_true > presence_threshold)
                    detection_accuracy = (pred_present_t == true_present_t).float().mean()

                    batch_results['mse'] += mse_t.item() * batch_size
                    batch_results['mae'] += mae_t.item() * batch_size
                    batch_results['detection_accuracy'] += detection_accuracy.item() * batch_size
                    batch_results['count'] += batch_size

                loader_stats['loss'] += loss.item()
                for key, value in details.items():
                    loader_stats[key] += value * batch_size

                num_batches += 1
                total_batches += 1
                weighted_loss_sum += loss.item()

            # Compute metrics
            all_preds = np.concatenate(all_preds, axis=0)
            all_true = np.concatenate(all_true, axis=0)
            eval_metrics = evaluate_performance(all_true, all_preds, cell_types, alpha_threshold=alpha_threshold)
            r2 = eval_metrics["Overall"].get('Global R² (Flattened)', 0.0)
            overall_mae = eval_metrics["Overall"].get('Overall MAE', 0.0)
            per_cell_r2 = {ct: eval_metrics["Per_Cell_Type"][ct].get("R²", 0.0) for ct in cell_types if ct in eval_metrics["Per_Cell_Type"]}

            # Normalize stats
            total_samples = num_batches * val_loader.batch_size
            for key in loader_stats:
                loader_stats[key] /= total_samples if key in ['mae', 'mse'] else num_batches
            for t in thresholds:
                batch_results = threshold_results[t][val_name]
                if batch_results['count'] > 0:
                    for key in ['mse', 'mae', 'detection_accuracy']:
                        batch_results[key] /= batch_results['count']
                    for key, value in batch_results.items():
                        if key != 'count':
                            loader_stats[f'thresh_{t}_{key}'] = value

            loader_stats['r2'] = r2
            loader_stats['mae'] = overall_mae
            loader_stats['per_cell_r2'] = per_cell_r2
            val_stats[val_name] = dict(loader_stats)

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
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[nn.Module, float]:
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
    patience_counter = 0
    history = defaultdict(list)

    val_loaders_unaugmented, val_loaders_augmented = val_loaders
    for epoch in range(num_epochs):
        print(f"\n🔹 Epoch {epoch + 1}/{num_epochs}")
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            presence_threshold=0.01
        )
        scheduler.step()

        current_val_loaders = val_loaders_unaugmented if epoch < 10 else val_loaders_augmented
        avg_val_loss, val_stats = validate(
            model,
            current_val_loaders,
            device,
            cell_types=cell_types,
            presence_threshold=0.01,
            alpha_threshold=0.01
        )

        for key, value in train_stats.items():
            history[key].append(value)
        for val_name, stats in val_stats.items():
            for k, v in stats.items():
                history[f"{val_name}/{k}"].append(v)

        if use_wandb:
            wandb_logs = {
                "epoch": epoch,
                "train/loss": train_stats['total_loss'],
                "val/avg_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr']
            }
            for val_name, stats in val_stats.items():
                for k, v in stats.items():
                    if k == 'per_cell_r2':
                        for ct, r2 in v.items():
                            wandb_logs[f"val/{val_name}/r2_{ct}"] = r2
                    else:
                        wandb_logs[f"val/{val_name}/{k}"] = v
            wandb.log(wandb_logs)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
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

    checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    if use_wandb:
        wandb.finish()

    return model, 0.01  # Default threshold; adjust if needed


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
