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
from deep_conv.deconvolution.loss import coverage_adaptive_loss
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


def train_epoch(model, train_loader, optimizer, device, log_interval=100, epoch=0):
    """
    Train the model for one epoch.
    
    Args:
        model: Coverage-aware deconvolution model
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        device: Computation device
        log_interval: How often to log batch metrics
        
    Returns:
        avg_loss: Average training loss
        metrics: Dictionary of training metrics    """
    model.train()
    start_time = time.time()
    total_loss = 0.0
    metrics = {
        'loss_props': 0, 
        'recon_loss': 0, 
        'sparsity_penalty': 0,
        'presence_precision': 0, 
        'presence_recall': 0, 
        'presence_f1': 0,
        'gate_usage': [],
        'avg_coverage': 0
    }
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} training")):
        marker_values = batch['X'].to(device)
        coverage = batch['coverage'].to(device)
        true_props = batch['y'].to(device)
        
        # Forward pass
        pred_props, reconstructed, valid_mask, presence_probs, presence_logits = model(
            marker_values, coverage)
        
        # Calculate loss
        loss, details = coverage_adaptive_loss(
            pred_props, true_props, reconstructed, marker_values,
            coverage, valid_mask, presence_probs, presence_logits,
            **model.loss_params
        )
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        metrics['loss_props'] += details['loss_props']
        metrics['recon_loss'] += details['recon_loss']
        metrics['sparsity_penalty'] += details['sparsity_penalty']
        metrics['presence_precision'] += details['presence_metrics']['precision']
        metrics['presence_recall'] += details['presence_metrics']['recall']
        metrics['presence_f1'] += details['presence_metrics']['f1']
        metrics['avg_coverage'] += details['avg_coverage']
        
        # Track gate usage
        with torch.no_grad():
            if hasattr(model, 'coverage_gate') and hasattr(model, 'temp'):
                log_coverage = torch.log1p(coverage.mean(dim=1, keepdim=True))
                gate_logits = model.coverage_gate(log_coverage)
                gate_value = torch.sigmoid(gate_logits / model.temp)
                metrics['gate_usage'].extend(gate_value.cpu().numpy().flatten())
        
        num_batches += 1
        
        # Log intermediate progress
        if batch_idx % log_interval == 0:
            print(f"  [Batch {batch_idx}/{len(train_loader)}] Loss={loss.item():.4f}")
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    for key in ['loss_props', 'recon_loss', 'sparsity_penalty', 'avg_coverage']:
        metrics[key] /= num_batches
    
    # Calculate gate statistics
    if metrics['gate_usage']:
        gate_usage = np.array(metrics['gate_usage'])
        metrics['gate_mean'] = np.mean(gate_usage)
        metrics['gate_std'] = np.std(gate_usage)
    
    # End epoch timing
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s. Avg loss={avg_loss:.4f}")
    
    return avg_loss, metrics

def validate_epoch(model, val_loaders, device):
    """
    Validate the model across all validation sets.
    
    Args:
        model: Coverage-aware deconvolution model
        val_loaders: Dictionary of validation DataLoaders
        device: Computation device
        
    Returns:
        avg_val_loss: Average validation loss across all sets
        val_metrics: Dictionary of validation metrics by set
    """
    model.eval()
    val_metrics = {}
    total_val_loss = 0
    total_sets = len(val_loaders)
    
    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            val_loss = 0
            set_metrics = {
                'loss_props': 0, 
                'recon_loss': 0, 
                'sparsity_penalty': 0,
                'presence_precision': 0, 
                'presence_recall': 0, 
                'presence_f1': 0,
                'gate_usage': [],
                'avg_coverage': 0
            }
            num_batches = 0
            
            for batch in val_loader:
                marker_values = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                true_props = batch['y'].to(device)
                
                # Forward pass
                pred_props, reconstructed, valid_mask, presence_probs, presence_logits = model(
                    marker_values, coverage)
                
                # Calculate loss
                loss, details = coverage_adaptive_loss(
                    pred_props, true_props, reconstructed, marker_values,
                    coverage, valid_mask, presence_probs, presence_logits,
                    **model.loss_params
                )
                
                # Track metrics
                val_loss += loss.item()
                set_metrics['loss_props'] += details['loss_props']
                set_metrics['recon_loss'] += details['recon_loss']
                set_metrics['sparsity_penalty'] += details['sparsity_penalty']
                set_metrics['presence_precision'] += details['presence_metrics']['precision']
                set_metrics['presence_recall'] += details['presence_metrics']['recall']
                set_metrics['presence_f1'] += details['presence_metrics']['f1']
                set_metrics['avg_coverage'] += details['avg_coverage']
                
                # Track gate usage
                log_coverage = torch.log1p(coverage.mean(dim=1, keepdim=True))
                gate_logits = model.coverage_gate(log_coverage)
                gate_value = torch.sigmoid(gate_logits / model.temp)
                set_metrics['gate_usage'].extend(gate_value.cpu().numpy().flatten())
                
                num_batches += 1
            
            # Calculate averages
            avg_set_loss = val_loss / num_batches
            for key in ['loss_props', 'recon_loss', 'sparsity_penalty', 
                        'presence_precision', 'presence_recall', 'presence_f1',
                        'avg_coverage']:
                set_metrics[key] /= num_batches
            
            # Calculate gate statistics
            gate_usage = np.array(set_metrics['gate_usage'])
            set_metrics['gate_mean'] = np.mean(gate_usage)
            set_metrics['gate_std'] = np.std(gate_usage)
            
            # Store set metrics
            val_metrics[val_name] = {
                'loss': avg_set_loss,
                **set_metrics
            }
            
            # Add to total validation loss
            total_val_loss += avg_set_loss
            
            print(f"Validation ({val_name}) - Loss: {avg_set_loss:.4f}, "
                  f"Props: {set_metrics['loss_props']:.4f}, F1: {set_metrics['presence_f1']:.4f}")
    
    # Calculate average validation loss
    avg_val_loss = total_val_loss / total_sets
    
    return avg_val_loss, val_metrics

def analyze_gate_behavior(model, val_loader, device):
    """
    Analyze the gate behavior across different coverage levels.
    
    Args:
        model: Coverage-aware deconvolution model
        val_loader: Validation DataLoader
        device: Computation device
        
    Returns:
        gate_stats: Dictionary of gate statistics by coverage bin
    """
    model.eval()
    coverage_bins = [0, 5, 10, 20, 50, float('inf')]
    gate_stats = {f"{low}-{high}": [] for low, high in zip(
        coverage_bins[:-1], coverage_bins[1:])}
    
    with torch.no_grad():
        for batch in val_loader:
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            
            # Calculate average coverage per sample
            avg_coverage = coverage.mean(dim=1)
            
            # Get gate values
            log_coverage = torch.log1p(avg_coverage.unsqueeze(1))
            gate_logits = model.coverage_gate(log_coverage)
            gate_value = torch.sigmoid(gate_logits / model.temp)
            
            # Group by coverage bin
            for i in range(len(avg_coverage)):
                cov = avg_coverage[i].item()
                gate = gate_value[i].item()
                
                for low, high in zip(coverage_bins[:-1], coverage_bins[1:]):
                    if low <= cov < high:
                        gate_stats[f"{low}-{high}"].append(gate)
                        break
    
    # Calculate statistics for each bin
    bin_statistics = {}
    for bin_name, gates in gate_stats.items():
        if gates:
            bin_statistics[bin_name] = {
                'count': len(gates),
                'mean': np.mean(gates),
                'std': np.std(gates),
                'min': np.min(gates),
                'max': np.max(gates)
            }
    
    return bin_statistics

def train_model(
    model, train_loader, val_loaders, model_path, 
    num_epochs=1000, patience=10, learning_rate=5e-3, 
    weight_decay=1e-5, use_wandb=True):
    """
    Training loop with coverage-aware monitoring and early stopping.
    
    Args:
        model: CoverageAwareDeconvolutionModel instance
        train_loader: DataLoader for training data
        val_loaders: Dict of validation DataLoaders
        model_path: Path to save model checkpoints
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        use_wandb: Whether to log to Weights & Biases
    
    Returns:
        model: Trained model (best checkpoint)
        best_val_metrics: Dictionary of metrics for the best model
    """
    # Setup
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True)
    
    # Create directories
    os.makedirs(model_path, exist_ok=True)
    
    # Initialize gate temperature
    init_temp = model.init_temperature(train_loader)
    print(f"Initialized gate temperature to {init_temp:.4f}")
    
    # Setup tracking
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    # Initialize wandb
    if use_wandb:
        import wandb
        wandb.init(project="methylation-deconv", config={
            "model_type": model.__class__.__name__,
            "num_cell_types": model.num_celltypes,
            "num_markers": model.num_markers,
            "feature_dim": model.feature_dim,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss_params": model.loss_params
        })
        wandb.watch(model, log_freq=100)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        
        # Training phase
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Validation phase
        val_loss, val_metrics = validate_epoch(model, val_loaders, device)
        
        # Analyze gate behavior
        gate_stats = analyze_gate_behavior(model, next(iter(val_loaders.values())), device)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train'].append({
            'epoch': epoch,
            'loss': train_loss,
            **train_metrics
        })
        history['val'].append({
            'epoch': epoch,
            'loss': val_loss,
            'metrics': val_metrics,
            'gate_stats': gate_stats
        })
        
        # Log to wandb
        if use_wandb:
            wandb_log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "gate/temperature": model.temp.item(),
                "gate/mean": train_metrics['gate_mean'],
                "gate/std": train_metrics['gate_std'],
                "lr": optimizer.param_groups[0]['lr']
            }
            
            # Add detailed metrics
            for key, value in train_metrics.items():
                if key not in ['gate_usage', 'gate_mean', 'gate_std']:
                    wandb_log[f"train/{key}"] = value
            
            # Add validation metrics
            for val_name, metrics in val_metrics.items():
                for key, value in metrics.items():
                    if key not in ['gate_usage', 'gate_mean', 'gate_std']:
                        wandb_log[f"val/{val_name}/{key}"] = value
            
            # Add gate statistics
            for bin_name, stats in gate_stats.items():
                for key, value in stats.items():
                    wandb_log[f"gate_bins/{bin_name}/{key}"] = value
            
            wandb.log(wandb_log)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'history': history
            }
            torch.save(checkpoint, os.path.join(model_path, "best_model.pt"))
            print(f"Saved new best model with val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    # Load best model
    print(f"\nLoading best model from epoch {best_epoch+1} with val_loss={best_val_loss:.4f}")
    checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Finish wandb run
    if use_wandb:
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["total_epochs"] = epoch + 1
        wandb.finish()
    
    return model, checkpoint['val_metrics']


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