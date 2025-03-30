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

    Args:
        config (dict): Dictionary containing experiment configurations.
        project_name (str): The wandb project name.
        entity (str, optional): The wandb entity (username or team name).

    Returns:
        run (wandb.run): The wandb run object.
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
    """Train the model for one epoch with zero-focused loss"""
    model.train()
    start_time = time.time()
    total_loss = 0.0
    metrics = {
        'prop_loss': 0, 
        'quality_loss': 0,
        'recon_loss': 0, 
        'sparsity': 0,
        'presence_loss': 0,
        'avg_coverage': 0
    }
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} training")):
        marker_values = batch['X'].to(device)
        coverage = batch['coverage'].to(device)
        true_props = batch['y'].to(device)
        
        # Forward pass with our enhanced model
        pred_props, reconstructed, valid_mask, feature_quality, presence_probs = model(marker_values, coverage)
        
        # Calculate loss with our zero-focused loss
        loss, details = zero_focused_adaptive_loss(
            pred_props, true_props, reconstructed, marker_values,
            coverage, valid_mask, feature_quality, presence_probs,
            alpha=1.0, beta=0.05, gamma=0.03, delta=0.2, zero_weight=5.0
        )
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        metrics['prop_loss'] += details['prop_loss']
        metrics['quality_loss'] += details.get('quality_loss', 0)
        metrics['recon_loss'] += details['recon_loss']
        metrics['sparsity'] += details['sparsity']
        metrics['presence_loss'] += details.get('presence_loss', 0)
        metrics['avg_coverage'] += details['avg_coverage']
        
        num_batches += 1
        
        # Log intermediate progress
        if batch_idx % log_interval == 0:
            print(f"  [Batch {batch_idx}/{len(train_loader)}] Loss={loss.item():.4f}")
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    for key in ['prop_loss', 'quality_loss', 'recon_loss', 'sparsity', 'presence_loss', 'avg_coverage']:
        metrics[key] /= num_batches
    
    # End epoch timing
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s. Avg loss={avg_loss:.4f}")
    
    return avg_loss, metrics


def validate_epoch(model, val_loaders, device):
    """
    Validate the model across all validation sets.
    
    Args:
        model: Clinical deconvolution model
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
                'prop_loss': 0, 
                'quality_loss': 0,
                'recon_loss': 0, 
                'sparsity': 0,
                'avg_coverage': 0,
                'coverage_weight': []
            }
            num_batches = 0
            
            for batch in val_loader:
                marker_values = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                true_props = batch['y'].to(device)
                
                # Forward pass
                pred_props, reconstructed, valid_mask, feature_quality = model(marker_values, coverage)
                
                # Calculate loss
                loss, details = coverage_adaptive_loss(
                    pred_props, true_props, reconstructed, marker_values,
                    coverage, valid_mask, feature_quality
                )
                
                # Track metrics
                val_loss += loss.item()
                set_metrics['prop_loss'] += details['prop_loss']
                set_metrics['quality_loss'] += details.get('quality_loss', 0)
                set_metrics['recon_loss'] += details['recon_loss']
                set_metrics['sparsity'] += details['sparsity']
                set_metrics['avg_coverage'] += details['avg_coverage']
                
                # Track coverage classifier values
                if hasattr(model, 'coverage_classifier'):
                    log_coverage = torch.log1p(coverage.mean(dim=1, keepdim=True))
                    coverage_weight = model.coverage_classifier(log_coverage)
                    set_metrics['coverage_weight'].extend(coverage_weight.cpu().numpy().flatten())
                
                num_batches += 1
            
            # Calculate averages
            avg_set_loss = val_loss / num_batches
            for key in ['prop_loss', 'quality_loss', 'recon_loss', 'sparsity', 'avg_coverage']:
                set_metrics[key] /= num_batches
            
            # Calculate coverage weight statistics
            if set_metrics['coverage_weight']:
                cov_weights = np.array(set_metrics['coverage_weight'])
                set_metrics['coverage_weight_mean'] = np.mean(cov_weights)
                set_metrics['coverage_weight_std'] = np.std(cov_weights)
            
            # Store set metrics
            val_metrics[val_name] = {
                'loss': avg_set_loss,
                **set_metrics
            }
            
            # Add to total validation loss
            total_val_loss += avg_set_loss
            
            print(f"Validation ({val_name}) - Loss: {avg_set_loss:.4f}, "
                  f"Props: {set_metrics['prop_loss']:.4f}, Quality: {set_metrics['quality_loss']:.4f}")
    
    # Calculate average validation loss
    avg_val_loss = total_val_loss / total_sets
    
    return avg_val_loss, val_metrics


def analyze_coverage_behavior(model, val_loader, device):
    """
    Analyze model behavior across different coverage levels.
    
    Args:
        model: Clinical deconvolution model
        val_loader: Validation DataLoader
        device: Computation device
        
    Returns:
        coverage_stats: Dictionary of coverage statistics by coverage bin
    """
    model.eval()
    coverage_bins = [0, 2, 5, 10, 20, 50, float('inf')]
    coverage_stats = {f"{low}-{high}": [] for low, high in zip(
        coverage_bins[:-1], coverage_bins[1:])}
    
    with torch.no_grad():
        for batch in val_loader:
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            
            # Calculate average coverage per sample
            avg_coverage = coverage.mean(dim=1)
            
            # Get coverage weight values
            log_coverage = torch.log1p(avg_coverage.unsqueeze(1))
            coverage_weight = model.coverage_classifier(log_coverage)
            
            # Forward pass to get quality scores
            pred_props, _, _, feature_quality = model(marker_values, coverage)
            
            # Group statistics by coverage bin
            for i in range(len(avg_coverage)):
                cov = avg_coverage[i].item()
                weight = coverage_weight[i].item()
                quality = feature_quality[i].mean().item()
                
                for low, high in zip(coverage_bins[:-1], coverage_bins[1:]):
                    if low <= cov < high:
                        coverage_stats[f"{low}-{high}"].append({
                            'coverage': cov,
                            'weight': weight,
                            'quality': quality,
                            'props': pred_props[i].cpu().numpy()
                        })
                        break
    
    # Calculate statistics for each bin
    bin_statistics = {}
    for bin_name, samples in coverage_stats.items():
        if samples:
            weights = [s['weight'] for s in samples]
            qualities = [s['quality'] for s in samples]
            coverages = [s['coverage'] for s in samples]
            
            # Calculate average number of non-zero predictions
            props_arrays = [s['props'] for s in samples]
            nonzeros = [np.sum(props > 0.01) for props in props_arrays]
            
            bin_statistics[bin_name] = {
                'count': len(samples),
                'weight_mean': np.mean(weights),
                'weight_std': np.std(weights),
                'quality_mean': np.mean(qualities),
                'coverage_mean': np.mean(coverages),
                'nonzero_cells_mean': np.mean(nonzeros),
                'nonzero_cells_std': np.std(nonzeros)
            }
    
    return bin_statistics


def train_model(
    model, train_loader, val_loaders, model_path, 
    num_epochs=1000, patience=10, learning_rate=1e-3, 
    weight_decay=1e-5, use_wandb=True):
    """
    Training loop with coverage-aware monitoring and early stopping.
    
    Args:
        model: Clinical deconvolution model
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True)
    
    # Create directories
    os.makedirs(model_path, exist_ok=True)
    
    # Setup tracking
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    # Initialize wandb
    if use_wandb:
        config = {
            "model_type": model.__class__.__name__,
            "num_cell_types": model.num_celltypes,
            "num_markers": model.num_markers,
            "feature_dim": model.feature_dim,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay
        }
        run = init_wandb(config, project_name="methylation-deconv")
        wandb.watch(model, log_freq=100)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        
        # Training phase
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, epoch=epoch)
        
        # Validation phase
        val_loss, val_metrics = validate_epoch(model, val_loaders, device)
        
        # Analyze coverage behavior
        first_val_loader = next(iter(val_loaders.values()))
        coverage_stats = analyze_coverage_behavior(model, first_val_loader, device)
        
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
            'coverage_stats': coverage_stats
        })
        
        # Log to wandb
        if use_wandb:
            wandb_log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": optimizer.param_groups[0]['lr']
            }
            
            # Add detailed metrics
            for key, value in train_metrics.items():
                if not isinstance(value, list):
                    wandb_log[f"train/{key}"] = value
            
            # Add validation metrics
            for val_name, metrics in val_metrics.items():
                for key, value in metrics.items():
                    if not isinstance(value, list):
                        wandb_log[f"val/{val_name}/{key}"] = value
            
            # Add coverage behavior statistics
            for bin_name, stats in coverage_stats.items():
                for key, value in stats.items():
                    wandb_log[f"coverage_bins/{bin_name}/{key}"] = value
            
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


def plot_training_history(history, save_path):
    """
    Plot training history using Plotly.
    
    Args:
        history: Dictionary of training metrics
        save_path: Path to save the plot
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Extract epochs and metrics
    epochs = list(range(len(history['train'])))
    train_loss = [entry['loss'] for entry in history['train']]
    val_loss = [entry['loss'] for entry in history['val']]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Loss Evolution',
            'Coverage Usage',
            'Proportion Loss',
            'Feature Quality'
        )
    )
    
    # Plot loss evolution
    fig.add_trace(
        go.Scatter(x=epochs, y=train_loss, name='Train Loss'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_loss, name='Val Loss'),
        row=1, col=1
    )
    
    # Plot coverage weight usage if available
    if 'coverage_weight_mean' in history['train'][0]:
        cov_weights = [entry.get('coverage_weight_mean', 0) for entry in history['train']]
        fig.add_trace(
            go.Scatter(x=epochs, y=cov_weights, name='Coverage Weight'),
            row=1, col=2
        )
    
    # Plot proportion loss
    prop_loss = [entry.get('prop_loss', 0) for entry in history['train']]
    fig.add_trace(
        go.Scatter(x=epochs, y=prop_loss, name='Prop Loss'),
        row=2, col=1
    )
    
    # Plot quality metrics if available
    if 'quality_loss' in history['train'][0]:
        quality_loss = [entry.get('quality_loss', 0) for entry in history['train']]
        fig.add_trace(
            go.Scatter(x=epochs, y=quality_loss, name='Quality Loss'),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text='Training History',
        showlegend=True
    )
    
    # Save figure
    if save_path:
        fig.write_html(f"{save_path}.html")
    else:
        fig.show()