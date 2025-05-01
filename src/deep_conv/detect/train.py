import os
import sys
import argparse
import torch
import numpy as np
import json
import logging
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import scipy.stats as stats
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
import math
import torch.optim as optim
from deep_conv.detect.visualise import *
from scipy import stats
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import pickle
from sklearn.metrics import confusion_matrix

from deep_conv.detect.preprocess import prepare_data_for_training, load_train_with_contrastive_data
from deep_conv.detect.model import EnhancedCancerDetectionModel, MarkerImportanceAnalyser

def parse_args():
    parser = argparse.ArgumentParser(description='Train improved cell type concentration model')
    
    parser.add_argument('--name', type=str, default=None, help='Name for this training run (used for output directory)')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing parquet files')
    parser.add_argument('--atlas_path', type=str, required=True, help='Path to atlas file')
    parser.add_argument('--target_cell_type', type=str, required=True, help='Target cell type')
    parser.add_argument('--target_cell_idx', type=int, required=True, help='Target cell index in ground truth')
    parser.add_argument('--excluded_markers', type=str, default='', help='Comma-separated list of marker indices to exclude')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularisation')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--min_reliable_coverage', type=float, default=5.0, 
                   help='Minimum coverage considered reliable for marker values')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimiser')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default="./saved_models", help='Output directory')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    
    parser.add_argument('--control_data_dir', type=str, default=None, 
                   help='Directory containing control data for contrastive learning')
    parser.add_argument('--calibrate', action='store_true', 
                    help='Calibrate confidence intervals')
    parser.add_argument('--calibrate_every', type=int, default=5,
                    help='Calibrate model every N epochs')

    # Clinical evaluation parameters (these were in the main function)
    parser.add_argument('--clinical_eval', action='store_true',
                       help='Enable comprehensive clinical evaluation metrics')
    parser.add_argument('--blank_samples_path', type=str, default=None,
                       help='Path to numpy array of blank sample predictions for LoB/LoD calculation')
    parser.add_argument('--generate_clinical_report', action='store_true',
                       help='Generate clinical interpretation report')
    parser.add_argument('--visualise_clinical', action='store_true',
                       help='Generate clinical performance visualisations')

    args = parser.parse_args()
    
    if args.name:
        dir_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"cell_type_model_{timestamp}"
    
    args.output_dir = os.path.join(args.output_dir, dir_name)

    args.early_stopping = max(args.early_stopping, 20)

    return args

def set_seed(seed):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(output_dir):
    """Set up logging configuration"""
    log_file = os.path.join(output_dir, 'training.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('cell_detection')
    logger.setLevel(logging.INFO)
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_git_info():
    """Retrieve information about the current Git repository state.

    This function extracts the commit hash, branch name, and repository cleanliness status
    using Git commands. It is used to track the codebase version during training for
    reproducibility and debugging.

    Returns:
        dict: Dictionary containing:
            - commit (str): Short hash of the current commit.
            - branch (str): Name of the current branch.
            - clean (bool): True if the repository has no uncommitted changes, False otherwise.
    """
    import subprocess

    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).strip().decode('utf-8')
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).strip().decode('utf-8')
        status = subprocess.check_output(
            ['git', 'status', '--porcelain']
        ).strip().decode('utf-8')
        return {
            'commit': commit_hash,
            'branch': branch,
            'clean': len(status) == 0
        }
    except subprocess.CalledProcessError:
        return {'commit': 'unknown', 'branch': 'unknown', 'clean': False}

def parse_excluded_markers(excluded_markers_str):
    """Parse excluded markers string into a list of indices"""
    if not excluded_markers_str:
        return []
    return [int(idx.strip()) for idx in excluded_markers_str.split(',')]# T-cells

def compute_loss(mu, uncertainty, y_true, control_mask=None):
    """
    Compute concentration-focused loss with enhanced range-specific weighting
    
    Args:
        mu: Predicted concentration values [batch_size, 1]
        uncertainty: Predicted uncertainty values [batch_size, 1]
        y_true: Ground truth concentration values [batch_size, 1]
        control_mask: Optional boolean mask identifying control samples
    """
    # Base MSE loss
    mse_loss = F.mse_loss(mu, y_true, reduction='none')
    
    # Calculate relative error for non-zero targets
    epsilon = 1e-6
    non_zero_mask = (y_true > epsilon)
    
    # Initialise relative error tensor
    rel_error = torch.zeros_like(mse_loss)
    
    # Compute relative error only for non-zero targets
    if non_zero_mask.sum() > 0:
        rel_error[non_zero_mask] = torch.abs(mu[non_zero_mask] - y_true[non_zero_mask]) / y_true[non_zero_mask]
    
    # Apply concentration-aware weighting
    # Higher weight for lower concentrations (on log scale)
    log_weights = 1.0 / torch.log10(y_true * 1000 + 10.0)
    log_weights = torch.clamp(log_weights, 0.5, 2.0)
    
    # Apply range-specific weights
    critical_ranges = [
        (0.0005, 0.001, 2.0),  # 0.05-0.1%
        (0.001, 0.005, 1.8),   # 0.1-0.5%
        (0.005, 0.01, 1.5),    # 0.5-1%
        (0.01, 0.05, 1.2)      # 1-5%
    ]
    
    range_weights = torch.ones_like(log_weights)
    for low, high, weight in critical_ranges:
        range_mask = (y_true >= low) & (y_true < high)
        range_weights[range_mask] = weight
    
    # Combine weights
    combined_weights = log_weights * range_weights
    
    # Weight both MSE and relative error
    weighted_mse = (mse_loss * combined_weights).mean()
    weighted_rel = (rel_error * combined_weights).mean() if non_zero_mask.sum() > 0 else torch.tensor(0.0)
    
    # Add control sample penalty (if provided)
    control_loss = torch.tensor(0.0)
    if control_mask is not None and control_mask.sum() > 0:
        control_loss = 10.0 * mu[control_mask].mean()
    
    # Add uncertainty calibration term
    z_scores = torch.abs(mu - y_true) / (uncertainty + epsilon)
    calibration_loss = F.smooth_l1_loss(z_scores, torch.ones_like(z_scores) * 1.96)
    
    # Combine all components
    total_loss = weighted_mse + 0.7 * weighted_rel + control_loss + 0.2 * calibration_loss
    
    return total_loss

def train_model(model, train_loader, val_loader, control_loader, args, device):
    """
    Train the model with a simplified approach
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        control_loader: DataLoader for control samples (can be None)
        args: Training arguments
        device: Device to run training on
    """
    # Setup logging
    logger = logging.getLogger('cell_detection')
    git_info = get_git_info()
    git_commit = git_info['commit']

    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Initialise tracking variables
    best_val_loss = float('inf')
    best_low_conc_error = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'mae': [],
        'r2_score': [],
        'low_conc_error': []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        # Update epoch counter in model if it exists
        if hasattr(model, 'epoch'):
            model.epoch = epoch
        
        # Training phase
        model.train()
        train_loss = 0
        
        # Progress bar for training
        train_bar = tqdm(enumerate(train_loader), 
                         desc=f"Epoch {epoch+1}/{args.epochs} [Train]", 
                         total=len(train_loader))
        
        for i, batch_data in train_bar:
            # Handle dataset with control_mask
            if len(batch_data) == 4:
                marker_values, coverage, y_true, control_mask = batch_data
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                control_mask = control_mask.to(device)
                
                # Forward pass
                mu, uncertainty, _ = model(marker_values, coverage)
                
                # Calculate loss with control mask
                loss = compute_loss(mu, uncertainty, y_true, control_mask)
                
            else:  # Standard dataset without control_mask
                marker_values, coverage, y_true = batch_data
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                # Forward pass
                mu, uncertainty, _ = model(marker_values, coverage)
                
                # Calculate loss without control mask
                loss = compute_loss(mu, uncertainty, y_true)
            
            # Scale for gradient accumulation
            if args.grad_accum_steps > 1:
                loss = loss / args.grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            train_loss += loss.item() * (args.grad_accum_steps if args.grad_accum_steps > 1 else 1)
            
            # Gradient accumulation and optimizer step
            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            # Update progress bar
            train_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        val_metrics = validate_model(model, val_loader, device)
        val_loss = val_metrics['loss']
        
        # Calculate low concentration error
        # Focus on 0.1-1% range which is often critical
        low_conc_error = 0
        count = 0
        for range_name, metrics in val_metrics['concentration_metrics'].items():
            if range_name in ['0.1-0.5%', '0.5-1%']:  # Critical low concentration range
                low_conc_error += metrics['mae'] * metrics['count']
                count += metrics['count']
        
        if count > 0:
            low_conc_error /= count
        
        # Add to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(val_metrics['mae'])
        history['r2_score'].append(val_metrics['r2'])
        history['low_conc_error'].append(low_conc_error)
        
        # Log validation results
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"MAE: {val_metrics['mae']:.6f}, "
                   f"R²: {val_metrics['r2']:.4f}")
        
        # Log concentration-specific metrics for critical ranges
        for range_name in ['0.1-0.5%', '0.5-1%', '1-5%']:
            if range_name in val_metrics['concentration_metrics']:
                metrics = val_metrics['concentration_metrics'][range_name]
                logger.info(f"  {range_name} (n={metrics['count']}): "
                           f"MAE={metrics['mae']:.6f}, "
                           f"Within 25%={metrics.get('within_25pct', 0):.1f}%")
        
        # Check for improvement
        improvement = False
        improvement_msg = ""
        
        # Check improvement in validation loss
        if val_loss < best_val_loss:
            improvement = True
            improvement_msg = f"New best model (loss)! {best_val_loss:.4f} → {val_loss:.4f}"
            best_val_loss = val_loss
        
        # Check improvement in low concentration error
        if low_conc_error < best_low_conc_error and count > 0:
            if improvement:
                improvement_msg += " and "
            improvement = True
            improvement_msg += f"Low conc. error: {best_low_conc_error:.6f} → {low_conc_error:.6f}"
            best_low_conc_error = low_conc_error
        
        # If there's improvement, save the model
        if improvement:
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'args': vars(args),
                'git_commit': git_commit
            }
            
            # Save best model
            torch.save(best_model_state, os.path.join(args.output_dir, 'best_model.pt'))
            logger.info(f"✓ {improvement_msg}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"× No improvement. Patience: {patience_counter}/{args.early_stopping}")
        
        # Periodic calibration (if enabled)
        if args.calibrate and (epoch % args.calibrate_every == 0 or epoch == args.epochs - 1):
            logger.info("Calibrating model...")
            try:
                calibration_results = model.calibrate(val_loader, device)
                logger.info(f"  Calibration factor: {calibration_results['calibration_factor']:.4f}")
            except Exception as e:
                logger.error(f"× Error during calibration: {str(e)}")
                logger.info("  Skipping calibration for this epoch")
        
        # Early stopping
        if patience_counter >= args.early_stopping:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Checkpoint saving
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'args': vars(args),
                'history': history,
                'git_commit': git_commit
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'val_metrics': val_metrics,
        'args': vars(args),
        'history': history,
        'git_commit': git_commit
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    
    # Create plots of training history
    plot_training_history(history, args.output_dir)
    
    # Load best model for return
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model'])
    
    return model, best_model_state

def validate_model(model, val_loader, device):
    """
    Validate model performance with concentration-focused metrics
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        device: Device to run validation on
        
    Returns:
        Dictionary of validation metrics
    """
    logger = logging.getLogger('cell_detection')
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    all_uncertainties = []
    
    # Track batch count for reporting
    batch_count = 0
    total_batches = len(val_loader)
    logger.info(f"Starting validation on {total_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            try:
                # Log progress periodically
                if batch_idx % 100 == 0:
                    logger.info(f"Validating batch {batch_idx}/{total_batches}...")
                
                # Extract data safely regardless of format
                marker_values = batch_data[0].to(device)
                coverage = batch_data[1].to(device)
                y_true = batch_data[2].to(device)
                
                # Forward pass
                mu, uncertainty, _ = model(marker_values, coverage)
                
                # Compute loss safely
                try:
                    # Use control_mask if available (4th element)
                    if len(batch_data) > 3:
                        control_mask = batch_data[3].to(device)
                        batch_loss = compute_loss(mu, uncertainty, y_true, control_mask)
                    else:
                        batch_loss = compute_loss(mu, uncertainty, y_true)
                except Exception as e:
                    logger.error(f"Error computing loss in validation batch {batch_idx}: {str(e)}")
                    # Use a default loss to continue
                    batch_loss = torch.tensor(1.0)
                
                val_loss += batch_loss.item()
                
                # Store predictions and targets for metrics
                all_preds.append(mu.cpu().numpy())
                all_targets.append(y_true.cpu().numpy())
                all_uncertainties.append(uncertainty.cpu().numpy())
                
                batch_count += 1
            except Exception as e:
                logger.error(f"Error processing validation batch {batch_idx}: {str(e)}")
                # Continue to next batch
                continue
    
    # If no batches were processed successfully, return empty metrics
    if batch_count == 0:
        logger.error("No validation batches were processed successfully!")
        return {
            'loss': float('inf'),
            'r2': 0.0,
            'mae': float('inf'),
            'concentration_metrics': {},
            'uncertainty_metrics': {}
        }
    
    # Calculate average loss
    val_loss /= batch_count
    
    try:
        # Concatenate predictions and targets
        predictions = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        uncertainties = np.concatenate(all_uncertainties)
        
        # Calculate regression metrics
        r2 = r2_score(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        # Calculate concentration-stratified metrics
        concentration_metrics = compute_concentration_metrics(predictions, targets)
        
        # Calculate uncertainty calibration metrics
        uncertainty_metrics = compute_uncertainty_metrics(predictions, targets, uncertainties)
        
        # Log success
        logger.info(f"Validation complete: processed {batch_count}/{total_batches} batches")
        
        # Return all metrics
        return {
            'loss': val_loss,
            'r2': r2,
            'mae': mae,
            'concentration_metrics': concentration_metrics,
            'uncertainty_metrics': uncertainty_metrics
        }
    except Exception as e:
        logger.error(f"Error calculating validation metrics: {str(e)}")
        # Return basic metrics that were calculated
        return {
            'loss': val_loss,
            'r2': 0.0,
            'mae': float('inf'),
            'concentration_metrics': {},
            'uncertainty_metrics': {}
        }

def compute_concentration_metrics(predictions, targets):
    """
    Compute concentration-stratified metrics
    
    Args:
        predictions: Predicted concentrations (numpy array)
        targets: Ground truth concentrations (numpy array)
        
    Returns:
        Dictionary of concentration-stratified metrics
    """
    # Define concentration ranges
    ranges = [
        (0, 0.0005, "0-0.05%"),
        (0.0005, 0.001, "0.05-0.1%"),
        (0.001, 0.005, "0.1-0.5%"),
        (0.005, 0.01, "0.5-1%"),
        (0.01, 0.05, "1-5%"),
        (0.05, 0.1, "5-10%"),
        (0.1, 1.0, ">10%")
    ]
    
    # Initialise results dictionary
    results = {}
    
    # Calculate metrics for each range
    for low, high, name in ranges:
        mask = (targets >= low) & (targets < high)
        range_preds = predictions[mask]
        range_targets = targets[mask]
        
        # Skip ranges with no samples
        if len(range_targets) == 0:
            continue
        
        # Calculate MAE
        mae = np.mean(np.abs(range_preds - range_targets))
        
        # Calculate percentage within error bands
        within_pct = {}
        non_zero_mask = range_targets > 0
        if np.sum(non_zero_mask) > 0:
            rel_errors = np.abs(range_preds[non_zero_mask] - range_targets[non_zero_mask]) / range_targets[non_zero_mask]
            for threshold in [0.1, 0.25, 0.5]:  # 10%, 25%, 50%
                within_pct[f"{int(threshold*100)}pct"] = float(np.mean(rel_errors <= threshold) * 100)
        
        # Store metrics
        results[name] = {
            'count': int(np.sum(mask)),
            'mae': float(mae),
            **within_pct
        }
    
    return results

def compute_uncertainty_metrics(predictions, targets, uncertainties):
    """
    Compute metrics for uncertainty estimates
    
    Args:
        predictions: Predicted concentrations (numpy array)
        targets: Ground truth concentrations (numpy array)
        uncertainties: Predicted uncertainties (numpy array)
        
    Returns:
        Dictionary of uncertainty metrics
    """
    # Calculate standardized errors (z-scores)
    z_scores = np.abs(predictions.flatten() - targets.flatten()) / (uncertainties.flatten() + 1e-6)
    
    # Calculate percentage within standard deviations
    within_1std = np.mean(z_scores <= 1.0) * 100
    within_2std = np.mean(z_scores <= 2.0) * 100
    
    # Calculate correlation between error magnitude and uncertainty
    abs_errors = np.abs(predictions.flatten() - targets.flatten())
    error_uncertainty_corr = np.corrcoef(abs_errors, uncertainties.flatten())[0, 1]
    
    return {
        'within_1std': float(within_1std),
        'within_2std': float(within_2std),
        'error_uncertainty_corr': float(error_uncertainty_corr),
        'mean_uncertainty': float(np.mean(uncertainties)),
        'median_uncertainty': float(np.median(uncertainties))
    }

def plot_training_history(history, output_dir):
    """Create plots of training history with Plotly"""
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get epoch numbers
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Create figure with subplots for main metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training & Validation Loss', 
            'R² Score & Low Concentration Error', 
            'Mean Absolute Error', 
            'Learning Rate'
        )
    )
    
    # 1. Training and validation loss
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['train_loss'], 
            mode='lines+markers', 
            name='Train Loss',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['val_loss'], 
            mode='lines+markers', 
            name='Validation Loss',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # 2. R² score and Low concentration error (dual axis)
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['r2_score'], 
            mode='lines+markers', 
            name='R² Score',
            line=dict(color='purple', width=2),
            yaxis="y2"
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['low_conc_error'], 
            mode='lines+markers', 
            name='Low Conc. Error',
            line=dict(color='orange', width=2),
        ),
        row=1, col=2
    )
    
    # 3. Mean absolute error
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['mae'], 
            mode='lines+markers', 
            name='MAE',
            line=dict(color='green', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Learning rate (if available)
    if 'lr' in history:
        fig.add_trace(
            go.Scatter(
                x=epochs, 
                y=history['lr'], 
                mode='lines+markers', 
                name='Learning Rate',
                line=dict(color='cyan', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Set log scale for learning rate
        fig.update_yaxes(type='log', row=2, col=2)
    
    # Update layout for main figure
    fig.update_layout(
        height=800,
        width=1000,
        title_text='Training History',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Update axes titles
    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)
    fig.update_xaxes(title_text='Epoch', row=2, col=1)
    fig.update_xaxes(title_text='Epoch', row=2, col=2)
    
    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_yaxes(title_text='Low Conc. Error', row=1, col=2)
    fig.update_yaxes(title_text='R²', row=1, col=2, side='right', range=[0, 1], secondary_y=True)
    fig.update_yaxes(title_text='MAE', row=2, col=1)
    fig.update_yaxes(title_text='Learning Rate', row=2, col=2)
    
    # Save main figure
    fig.write_html(os.path.join(plots_dir, 'training_history.html'))
    fig.write_image(os.path.join(plots_dir, 'training_history.png'), scale=2)
    
    # Create a concentration-stratified metrics plot if available
    if 'concentration_metrics' in history and len(history['concentration_metrics']) > 0:
        # Create a new figure for concentration metrics over time
        fig_conc = go.Figure()
        
        # Get the last epoch's concentration metrics
        last_epoch_metrics = history['concentration_metrics'][-1]
        
        # Extract range names and MAE values
        ranges = []
        mae_values = []
        counts = []
        for range_name, metrics in last_epoch_metrics.items():
            ranges.append(range_name)
            mae_values.append(metrics['mae'])
            counts.append(metrics['count'])
        
        # Add bars for MAE
        fig_conc.add_trace(
            go.Bar(
                x=ranges,
                y=mae_values,
                text=[f"n={c}" for c in counts],
                name='MAE',
                marker_color='blue'
            )
        )
        
        # Update layout
        fig_conc.update_layout(
            title='Concentration-Stratified MAE (Final Epoch)',
            xaxis_title='Concentration Range',
            yaxis_title='Mean Absolute Error',
            template='plotly_white',
            width=900,
            height=600
        )
        
        # Save figure
        fig_conc.write_html(os.path.join(plots_dir, 'concentration_stratified_mae.html'))
        fig_conc.write_image(os.path.join(plots_dir, 'concentration_stratified_mae.png'), scale=2)
    
    return True

def evaluate(model, data_loader, args, device, split_name="test"):
    """
    Enhanced evaluation function focused on regression metrics
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        args: Training arguments
        device: Device to run evaluation on
        split_name: Name of data split (e.g., "test")
        
    Returns:
        Dictionary of evaluation results
    """
    logger = logging.getLogger('cell_detection')
    logger.info(f"Starting model evaluation on {split_name} set...")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_uncertainties = []
    all_marker_attentions = []
    
    # Create progress bar for evaluation
    eval_bar = tqdm(data_loader, desc=f"Evaluating {split_name} set", position=0)
    
    with torch.no_grad():
        for batch_data in eval_bar:
            # Handle different data formats
            if len(batch_data) == 4:
                marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask
            else:
                marker_values, coverage, y_true = batch_data
                
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            
            # Forward pass
            mu, uncertainty, attention_weights = model(marker_values, coverage)
            
            # Store results
            all_preds.append(mu.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())
            all_marker_attentions.append(attention_weights.cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    uncertainties = np.concatenate(all_uncertainties)
    marker_attentions = np.concatenate(all_marker_attentions)
    
    # Calculate basic metrics
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    # Calculate confidence intervals
    lower_ci = []
    upper_ci = []
    
    for mu, sigma in zip(predictions.flatten(), uncertainties.flatten()):
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma
        lower_ci.append([max(0, lower)])  # Ensure non-negative
        upper_ci.append([upper])
    
    lower_ci = np.array(lower_ci)
    upper_ci = np.array(upper_ci)
    
    # Calculate concentration metrics
    concentration_metrics = compute_concentration_metrics(predictions, targets)
    
    # Calculate uncertainty metrics
    uncertainty_metrics = compute_uncertainty_metrics(predictions, targets, uncertainties)
    
    # Log basic results
    logger.info(f"\n{split_name.upper()} RESULTS:")
    logger.info(f"Number of {split_name} samples: {len(targets)}")
    logger.info(f"R² Score: {r2:.6f}")
    logger.info(f"Mean Absolute Error: {mae:.6f}")
    
    # Log uncertainty metrics
    logger.info(f"\nUNCERTAINTY METRICS:")
    logger.info(f"  Errors within 1 std: {uncertainty_metrics['within_1std']:.2f}% (ideal: 68%)")
    logger.info(f"  Errors within 2 std: {uncertainty_metrics['within_2std']:.2f}% (ideal: 95%)")
    logger.info(f"  Error-Uncertainty Correlation: {uncertainty_metrics['error_uncertainty_corr']:.4f}")
    
    # Log concentration metrics
    logger.info(f"\nCONCENTRATION METRICS:")
    for range_name, metrics in concentration_metrics.items():
        logger.info(f"  {range_name} (n={metrics['count']}): "
                   f"MAE={metrics['mae']:.6f}, "
                   f"Within 25%={metrics.get('25pct', 0):.1f}%")
    
    # Analyze marker importance (simplified)
    marker_importances = marker_attentions.mean(axis=0)
    top_indices = np.argsort(marker_importances)[::-1][:10]
    top_weights = marker_importances[top_indices]
    
    logger.info(f"\nTOP 10 MARKERS BY IMPORTANCE:")
    for i, (idx, weight) in enumerate(zip(top_indices, top_weights)):
        logger.info(f"  #{i+1}: Marker {idx} (weight: {weight:.4f})")
    
    # Save results
    results = {
        'predictions': predictions.flatten().tolist(),
        'targets': targets.flatten().tolist(),
        'uncertainties': uncertainties.flatten().tolist(),
        'metrics': {
            'r2': float(r2),
            'mae': float(mae)
        },
        'concentration_metrics': concentration_metrics,
        'uncertainty_metrics': uncertainty_metrics,
        'marker_importance': {
            'top_indices': top_indices.tolist(),
            'top_weights': top_weights.tolist()
        }
    }
    
    # Save to file
    results_file = os.path.join(args.output_dir, f'{split_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"{split_name} results saved to {results_file}")
    
    # Create visualisations using visualise_results instead of create_evaluation_visualisations
    vis_dir = os.path.join(args.output_dir, 'visualisations', split_name)
    try:
        visualise_results(
            predictions, 
            targets, 
            vis_dir,
            ci_data=(lower_ci, upper_ci),
            marker_importance=marker_importances,
            prefix=f"{split_name.capitalize()} "
        )
        logger.info(f"✓ {split_name.capitalize()} visualisations created in {vis_dir}")
    except Exception as e:
        logger.error(f"× Error creating visualisations for {split_name}: {str(e)}")
    
    return results

def visualise_results(predictions, ground_truth, output_dir, ci_data=None, marker_importance=None, prefix=""):
    """
    Unified visualisation function for both validation and test results,
    with added clinical assessment metrics
    
    Args:
        predictions: Array of predicted cell type concentrations
        ground_truth: Array of true cell type concentrations
        output_dir: Directory to save visualisations
        ci_data: Optional tuple of (lower_ci, upper_ci) for confidence interval visualisation
        marker_importance: Optional marker importance data
        prefix: Optional prefix for output files
    """
    import os
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from sklearn.metrics import r2_score, mean_absolute_error
    from scipy import stats
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create clinical directory
    clinical_dir = os.path.join(output_dir, 'clinical')
    os.makedirs(clinical_dir, exist_ok=True)
    
    # Convert inputs to numpy arrays
    preds = np.array(predictions).flatten()
    targets = np.array(ground_truth).flatten()
    
    # Calculate basic metrics
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    
    # Create dataframe for analysis
    df = pd.DataFrame({
        'true_value': targets,
        'predicted_value': preds,
        'error': preds - targets,
        'abs_error': np.abs(preds - targets)
    })
    
    # Calculate relative error
    rel_error = np.full_like(targets, np.nan, dtype=float)
    non_zero_mask = targets > 0
    rel_error[non_zero_mask] = np.abs(preds[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask] * 100
    df['rel_error'] = rel_error
    
    # Calculate within-percentage metrics
    within_10pct = np.mean(rel_error[non_zero_mask] <= 10) * 100 if np.any(non_zero_mask) else 0
    within_25pct = np.mean(rel_error[non_zero_mask] <= 25) * 100 if np.any(non_zero_mask) else 0
    
    # Add confidence interval data if provided
    if ci_data is not None:
        lower_ci, upper_ci = ci_data
        df['lower_ci'] = lower_ci.flatten()
        df['upper_ci'] = upper_ci.flatten()
        
        # Calculate percentage of targets within CI
        in_ci = ((targets >= lower_ci.flatten()) & (targets <= upper_ci.flatten())).mean() * 100
        ci_width = (upper_ci.flatten() - lower_ci.flatten()).mean()
        
        print(f"{prefix}Targets within CI: {in_ci:.2f}%")
        print(f"{prefix}Average CI width: {ci_width:.6f}")
    
    # Define concentration ranges
    ranges = [
        (0, 0.0005, "0-0.05%"),
        (0.0005, 0.001, "0.05-0.1%"),
        (0.001, 0.005, "0.1-0.5%"),
        (0.005, 0.01, "0.5-1%"),
        (0.01, 0.05, "1-5%"),
        (0.05, 0.1, "5-10%"),
        (0.1, 1.0, ">10%")
    ]
    
    # Add range column
    df['range'] = 'Unknown'
    for low, high, name in ranges:
        mask = (df['true_value'] >= low) & (df['true_value'] < high)
        df.loc[mask, 'range'] = name
    
    # Log key metrics
    print(f"{prefix}Number of samples: {len(targets)}")
    print(f"{prefix}R² Score: {r2:.4f}")
    print(f"{prefix}MAE: {mae:.6f}")
    print(f"{prefix}Within 10% error: {within_10pct:.1f}%")
    print(f"{prefix}Within 25% error: {within_25pct:.1f}%")
    
    # Create visualisations
    
    # 1. Scatter plot of predictions vs ground truth
    fig_scatter = px.scatter(
        df, 
        x='true_value', 
        y='predicted_value',
        color='range',
        log_x=True, 
        log_y=True,
        opacity=0.7,
        hover_data=['abs_error', 'rel_error']
    )
    
    # Add identity line
    fig_scatter.add_trace(
        go.Scatter(
            x=[1e-6, 1],
            y=[1e-6, 1],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect prediction'
        )
    )
    
    # Add metrics annotation
    fig_scatter.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"R² = {r2:.4f}<br>MAE = {mae:.6f}",
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig_scatter.update_layout(
        title=f'{prefix}Predicted vs True Concentration (Log Scale)',
        xaxis_title='True Concentration',
        yaxis_title='Predicted Concentration',
        template='plotly_white',
        width=900,
        height=700
    )
    
    # Format axes as percentages
    fig_scatter.update_xaxes(tickformat='.2%')
    fig_scatter.update_yaxes(tickformat='.2%')
    
    # Save figure
    fig_scatter.write_html(os.path.join(output_dir, 'scatter_plot_log.html'))
    fig_scatter.write_image(os.path.join(output_dir, 'scatter_plot_log.png'), scale=2)
    
    # 2. Stratified error analysis
    range_stats = []
    for low, high, name in ranges:
        mask = (df['true_value'] >= low) & (df['true_value'] < high)
        range_df = df[mask]
        if len(range_df) > 0:
            range_mae = range_df['abs_error'].mean()
            range_within_25pct = np.mean(range_df['rel_error'] <= 25) * 100 if non_zero_mask.any() else 0
            
            range_stats.append({
                'range': name,
                'mae': range_mae,
                'count': len(range_df),
                'within_25pct': range_within_25pct
            })
    
    # Create dataframe for plotting
    range_df = pd.DataFrame(range_stats)
    
    # Create bar chart
    fig_stratified = go.Figure()
    
    fig_stratified.add_trace(
        go.Bar(
            x=range_df['range'],
            y=range_df['mae'],
            text=[f"n={count}" for count in range_df['count']],
            name='MAE',
            marker_color='blue',
            opacity=0.7
        )
    )
    
    # Update layout
    fig_stratified.update_layout(
        title=f'{prefix}Stratified Mean Absolute Error by Concentration Range',
        xaxis_title='Concentration Range',
        yaxis_title='MAE',
        template='plotly_white',
        width=900,
        height=600
    )
    
    # Save figure
    fig_stratified.write_html(os.path.join(output_dir, 'stratified_mae.html'))
    fig_stratified.write_image(os.path.join(output_dir, 'stratified_mae.png'), scale=2)
    
    # 3. "Within 25%" analysis
    fig_within = go.Figure()
    
    fig_within.add_trace(
        go.Bar(
            x=range_df['range'],
            y=range_df['within_25pct'],
            text=[f"n={count}" for count in range_df['count']],
            name='Within 25%',
            marker_color='green',
            opacity=0.7
        )
    )
    
    # Add reference line at 75%
    fig_within.add_shape(
        type="line",
        x0=-0.5,
        y0=75,
        x1=len(range_df) - 0.5,
        y1=75,
        line=dict(color="red", dash="dash")
    )
    
    # Update layout
    fig_within.update_layout(
        title=f'{prefix}Percentage Within 25% Error by Concentration Range',
        xaxis_title='Concentration Range',
        yaxis_title='Percentage Within 25% Error',
        template='plotly_white',
        width=900,
        height=600,
        yaxis=dict(range=[0, 100])
    )
    
    # Save figure
    fig_within.write_html(os.path.join(output_dir, 'within_25pct.html'))
    fig_within.write_image(os.path.join(output_dir, 'within_25pct.png'), scale=2)
    
    # 4. Plot predictions with CI if CI data is provided
    if ci_data is not None:
        # Sort by ground truth for clearer CI visualisation
        sorted_indices = np.argsort(targets)
        sorted_targets = targets[sorted_indices]
        sorted_preds = preds[sorted_indices]
        sorted_lower = lower_ci.flatten()[sorted_indices]
        sorted_upper = upper_ci.flatten()[sorted_indices]
        
        # Create figure
        fig_ci = go.Figure()
        
        # Add confidence interval
        fig_ci.add_trace(
            go.Scatter(
                x=np.arange(len(sorted_targets)),
                y=sorted_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig_ci.add_trace(
            go.Scatter(
                x=np.arange(len(sorted_targets)),
                y=sorted_lower,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(0, 176, 246, 0.2)',
                fill='tonexty',
                name='95% CI'
            )
        )
        
        # Add predictions
        fig_ci.add_trace(
            go.Scatter(
                x=np.arange(len(sorted_targets)),
                y=sorted_preds,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Predictions'
            )
        )
        
        # Add targets
        fig_ci.add_trace(
            go.Scatter(
                x=np.arange(len(sorted_targets)),
                y=sorted_targets,
                mode='lines',
                line=dict(color='red', width=2),
                name='True Values'
            )
        )
        
        # Update layout
        fig_ci.update_layout(
            title=f'{prefix}Predictions vs Targets with Confidence Intervals (sorted) (In CI: {in_ci:.1f}%)',
            xaxis_title='Sample Index (sorted by true value)',
            yaxis_title='Cell Type Concentration',
            template='plotly_white',
            width=900,
            height=600
        )
        
        # Save figure
        fig_ci.write_html(os.path.join(output_dir, 'predictions_with_ci.html'))
        fig_ci.write_image(os.path.join(output_dir, 'predictions_with_ci.png'), scale=2)
    
    # 5. Plot marker importance if provided
    if marker_importance is not None:
        # Get top markers
        top_k = min(20, len(marker_importance))
        top_indices = np.argsort(marker_importance)[::-1][:top_k]
        top_weights = marker_importance[top_indices]
        
        # Create figure
        fig_markers = go.Figure()
        
        fig_markers.add_trace(
            go.Bar(
                y=[f"Marker {i}" for i in top_indices],
                x=top_weights,
                orientation='h',
                marker_color='blue',
                opacity=0.7
            )
        )
        
        # Update layout
        fig_markers.update_layout(
            title=f'{prefix}Top {top_k} Markers by Importance',
            xaxis_title='Importance Weight',
            yaxis_title='Marker ID',
            template='plotly_white',
            width=900,
            height=600,
            yaxis=dict(autorange="reversed")  # Highest at top
        )
        
        # Save figure
        fig_markers.write_html(os.path.join(output_dir, 'marker_importance.html'))
        fig_markers.write_image(os.path.join(output_dir, 'marker_importance.png'), scale=2)
    
    # CLINICAL ASSESSMENT visualisATIONS
    
    # 1. Error distribution analysis
    fig_err = make_subplots(rows=2, cols=1, 
                          subplot_titles=('Absolute Error Distribution', 'Relative Error Distribution'))
    
    # Add absolute error histogram
    fig_err.add_trace(
        go.Histogram(
            x=df['abs_error'],
            nbinsx=50,
            name='Absolute Error',
            marker_color='blue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add vertical line at median and mean
    median_abs_error = df['abs_error'].median()
    mean_abs_error = df['abs_error'].mean()
    
    fig_err.add_vline(x=median_abs_error, line=dict(color="red", dash="dash"), 
                    annotation_text=f"Median: {median_abs_error:.6f}", row=1, col=1)
    fig_err.add_vline(x=mean_abs_error, line=dict(color="green", dash="dash"), 
                    annotation_text=f"Mean: {mean_abs_error:.6f}", row=1, col=1)
    
    # Add relative error histogram (cap extreme values)
    rel_error_capped = df['rel_error'].clip(-500, 500)  # Cap at ±500%
    rel_error_valid = rel_error_capped.dropna()
    
    fig_err.add_trace(
        go.Histogram(
            x=rel_error_valid,
            nbinsx=50,
            name='Relative Error',
            marker_color='orange',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add vertical line at median and mean
    median_rel_error = rel_error_valid.median()
    mean_rel_error = rel_error_valid.mean()
    
    fig_err.add_vline(x=median_rel_error, line=dict(color="red", dash="dash"), 
                    annotation_text=f"Median: {median_rel_error:.1f}%", row=2, col=1)
    fig_err.add_vline(x=mean_rel_error, line=dict(color="green", dash="dash"), 
                    annotation_text=f"Mean: {mean_rel_error:.1f}%", row=2, col=1)
    
    # Add reference lines at ±25%
    fig_err.add_vline(x=25, line=dict(color="black", dash="dot"), row=2, col=1)
    fig_err.add_vline(x=-25, line=dict(color="black", dash="dot"), row=2, col=1)
    
    # Update layout
    fig_err.update_layout(
        title='Error Distribution Analysis',
        template='plotly_white',
        height=800,
        width=900,
        showlegend=False
    )
    
    # Update x-axis ranges
    fig_err.update_xaxes(title_text="Absolute Error", row=1, col=1)
    fig_err.update_xaxes(title_text="Relative Error (%)", row=2, col=1, range=[-100, 100])
    
    # Save figure
    fig_err.write_html(os.path.join(clinical_dir, 'error_distribution.html'))
    fig_err.write_image(os.path.join(clinical_dir, 'error_distribution.png'), scale=2)
    
    # 2. Log-space analysis (important for clinical assessment)
    # Add small constant to avoid log(0)
    epsilon = 1e-10
    log_targets = np.log10(targets + epsilon)
    log_preds = np.log10(preds + epsilon)
    
    # Linear regression in log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_targets, log_preds)
    log_r2 = r_value ** 2
    
    # Create log-log plot
    fig_log = go.Figure()
    
    # Add scatter points
    fig_log.add_trace(
        go.Scatter(
            x=log_targets,
            y=log_preds,
            mode='markers',
            marker=dict(
                color=df['rel_error'].clip(-50, 50),  # Use capped relative error for color
                colorscale='RdBu_r',
                cmin=-50,
                cmax=50,
                colorbar=dict(title='Relative Error (%)')
            ),
            name='Samples'
        )
    )
    
    # Add regression line
    x_range = np.linspace(min(log_targets), max(log_targets), 100)
    y_pred = slope * x_range + intercept
    
    fig_log.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'Regression Line (slope={slope:.2f})'
        )
    )
    
    # Add identity line (slope=1, intercept=0)
    fig_log.add_trace(
        go.Scatter(
            x=x_range,
            y=x_range,
            mode='lines',
            line=dict(color='black', dash='dot'),
            name='Perfect Prediction'
        )
    )
    
    # Add annotation for log-space metrics
    fig_log.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Log-Space R² = {log_r2:.4f}<br>Slope = {slope:.3f}<br>Intercept = {intercept:.3f}",
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig_log.update_layout(
        title='Log-Space Analysis for Linearity at Low Concentrations',
        xaxis_title='Log10(True Concentration)',
        yaxis_title='Log10(Predicted Concentration)',
        template='plotly_white',
        width=900,
        height=700
    )
    
    # Save figure
    fig_log.write_html(os.path.join(clinical_dir, 'log_space_analysis.html'))
    fig_log.write_image(os.path.join(clinical_dir, 'log_space_analysis.png'), scale=2)
    
    # Return metrics dictionary
    metrics = {
        'r2': float(r2),
        'mae': float(mae),
        'within_10pct': float(within_10pct),
        'within_25pct': float(within_25pct),
        'concentration_metrics': {
            name: {
                'count': int(stats['count']),
                'mae': float(stats['mae']),
                'within_25pct': float(stats['within_25pct'])
            }
            for name, stats in zip(range_df['range'], range_df.to_dict('records'))
        },
        # Add log-space metrics
        'log_space': {
            'r2': float(log_r2),
            'slope': float(slope),
            'intercept': float(intercept)
        }
    }
    
    # Add CI metrics if available
    if ci_data is not None:
        metrics['in_ci_percentage'] = float(in_ci)
        metrics['ci_width'] = float(ci_width)
    
    return metrics

def sensitivity_score(y_true, y_pred):
    """Calculate sensitivity/recall score"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def specificity_score(y_true, y_pred):
    """Calculate specificity score"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def evaluate_clinical_performance(y_true, y_pred, blank_samples=None):
    """
    Comprehensive evaluation of tumor fraction prediction model with clinical perspective.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth tumor fractions (0-1)
    y_pred : array-like
        Predicted tumor fractions (0-1)
    blank_samples : array-like, optional
        Predictions on known blank/negative samples for LoB calculation
        
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    results = {}
    
    # Define clinically relevant ranges
    ranges = {
        'ultra_low': (0, 0.001),
        'very_low': (0.001, 0.01),
        'low': (0.01, 0.1),
        'medium': (0.1, 0.5),
        'high': (0.5, 1.0)
    }
    
    # 1. Performance across clinically relevant ranges
    results['stratified_performance'] = {}
    for range_name, (lower, upper) in ranges.items():
        mask = (y_true >= lower) & (y_true < upper)
        if sum(mask) > 0:
            results['stratified_performance'][range_name] = {
                'count': sum(mask),
                'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                'rmse': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                'median_error': np.median(np.abs(y_true[mask] - y_pred[mask])),
                'relative_error': np.median(np.abs((y_true[mask] - y_pred[mask]) / 
                                           np.maximum(y_true[mask], 1e-10))) * 100
            }
    
    # 2. Limit of Blank and Detection
    if blank_samples is not None:
        results['limit_of_blank'] = np.percentile(blank_samples, 95)
        # LoD = LoB + 1.645 * SD of low concentration samples
        low_samples_mask = y_true < 0.01
        if sum(low_samples_mask) > 0:
            low_sd = np.std(y_pred[low_samples_mask] - y_true[low_samples_mask])
            results['limit_of_detection'] = results['limit_of_blank'] + 1.645 * low_sd
    
    # 3. Calibration plot data (for visualisation)
    results['calibration'] = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    # 4. "Detection accuracy" at various clinical thresholds
    results['detection_performance'] = {}
    clinical_thresholds = [0.001, 0.005, 0.01, 0.05]
    
    for threshold in clinical_thresholds:
        y_true_binary = y_true >= threshold
        y_pred_binary = y_pred >= threshold
        results['detection_performance'][threshold] = {
            'sensitivity': sensitivity_score(y_true_binary, y_pred_binary),
            'specificity': specificity_score(y_true_binary, y_pred_binary),
            'ppv': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'npv': precision_score(~y_true_binary, ~y_pred_binary, zero_division=0)
        }
    
    # 5. Evaluation in log space for low-range accuracy
    # Add small constant to avoid log(0)
    epsilon = 1e-10
    log_y_true = np.log10(y_true + epsilon)
    log_y_pred = np.log10(y_pred + epsilon)
    
    # Linear regression in log space
    slope, intercept, r_value, _, _ = stats.linregress(log_y_true, log_y_pred)
    results['log_space'] = {
        'r_squared': r_value ** 2,
        'slope': slope,
        'intercept': intercept
    }
    
    # 6. Clinical error assessment - weighted more heavily for false negatives
    # and overestimation at low values
    weighted_errors = []
    for true, pred in zip(y_true, y_pred):
        error = pred - true
        # Higher penalty for missing tumor (false negative)
        if error < 0 and true > 0.001:
            weighted_errors.append(abs(error) * 2)
        # Higher penalty for overestimation at very low values
        elif error > 0 and true < 0.01:
            weighted_errors.append(abs(error) * 3)
        else:
            weighted_errors.append(abs(error))
    
    results['clinical_weighted_error'] = np.mean(weighted_errors)
    
    return results

def visualise_clinical_performance(evaluation_results):
    """
    Create comprehensive visualisation of clinical performance metrics using Plotly.
    
    Parameters:
    -----------
    evaluation_results : dict
        Output from evaluate_clinical_performance
        
    Returns:
    --------
    fig : plotly figure
        Figure containing all visualisations
    """
    # Create subplot figure
    fig = sp.make_subplots(
        rows=3, 
        cols=2,
        subplot_titles=(
            'MAE by Tumor Fraction Range', 
            'Calibration Plot (Log Scale)',
            'Detection Performance vs. Threshold', 
            'Log-space Residual Plot',
            'Limit of Blank and Detection', 
            ''
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter", "colspan": 2}, {}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # 1. Error distribution across ranges plot
    if 'stratified_performance' in evaluation_results:
        ranges = list(evaluation_results['stratified_performance'].keys())
        maes = [evaluation_results['stratified_performance'][r]['mae'] for r in ranges]
        
        fig.add_trace(
            go.Bar(x=ranges, y=maes, name='MAE', marker_color='royalblue'),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text='Tumor Fraction Range', row=1, col=1)
        fig.update_yaxes(title_text='Mean Absolute Error', row=1, col=1)
    
    # 2. Calibration plot (with log scale)
    if 'calibration' in evaluation_results:
        true = evaluation_results['calibration']['y_true']
        pred = evaluation_results['calibration']['y_pred']
        
        # Create perfect prediction line
        min_val = min(min(true), min(pred))
        max_val = max(max(true), max(pred))
        perfect_line = np.logspace(np.log10(max(min_val, 1e-10)), np.log10(max_val), 100)
        
        fig.add_trace(
            go.Scatter(
                x=true, 
                y=pred, 
                mode='markers', 
                name='Predictions',
                marker=dict(color='royalblue', size=8, opacity=0.6)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=perfect_line, 
                y=perfect_line, 
                mode='lines', 
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(
            title_text='True Tumor Fraction', 
            type='log',
            row=1, col=2
        )
        fig.update_yaxes(
            title_text='Predicted Tumor Fraction', 
            type='log',
            row=1, col=2
        )
    
    # 3. Detection performance across thresholds
    if 'detection_performance' in evaluation_results:
        thresholds = list(evaluation_results['detection_performance'].keys())
        sens = [evaluation_results['detection_performance'][t]['sensitivity'] for t in thresholds]
        spec = [evaluation_results['detection_performance'][t]['specificity'] for t in thresholds]
        ppv = [evaluation_results['detection_performance'][t]['ppv'] for t in thresholds]
        npv = [evaluation_results['detection_performance'][t]['npv'] for t in thresholds]
        
        fig.add_trace(
            go.Scatter(
                x=thresholds, 
                y=sens, 
                mode='lines+markers', 
                name='Sensitivity',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=thresholds, 
                y=spec, 
                mode='lines+markers', 
                name='Specificity',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=thresholds, 
                y=ppv, 
                mode='lines+markers', 
                name='PPV',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=thresholds, 
                y=npv, 
                mode='lines+markers', 
                name='NPV',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(
            title_text='Tumor Fraction Threshold', 
            type='log',
            row=2, col=1
        )
        fig.update_yaxes(
            title_text='Performance',
            range=[0, 1.05],
            row=2, col=1
        )
    
    # 4. Residual plot in log space
    if 'calibration' in evaluation_results:
        true = evaluation_results['calibration']['y_true']
        pred = evaluation_results['calibration']['y_pred']
        
        epsilon = 1e-10
        log_true = np.log10(true + epsilon)
        log_pred = np.log10(pred + epsilon)
        
        fig.add_trace(
            go.Scatter(
                x=log_true, 
                y=log_pred - log_true, 
                mode='markers', 
                name='Log Residuals',
                marker=dict(color='royalblue', size=8, opacity=0.6)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[min(log_true), max(log_true)], 
                y=[0, 0], 
                mode='lines', 
                name='Zero Error',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text='Log True Tumor Fraction', row=2, col=2)
        fig.update_yaxes(title_text='Log Residual', row=2, col=2)
    
    # 5. Limit of Blank/Detection visualisation if available
    if 'limit_of_blank' in evaluation_results and 'calibration' in evaluation_results:
        lob = evaluation_results['limit_of_blank']
        lod = evaluation_results.get('limit_of_detection', lob * 1.5)
        
        # Create histogram data for low values
        pred = evaluation_results['calibration']['y_pred']
        true = evaluation_results['calibration']['y_true']
        low_mask = true < 0.05
        
        if sum(low_mask) > 10:
            low_preds = pred[low_mask]
            
            fig.add_trace(
                go.Histogram(
                    x=low_preds,
                    name='Low Range Predictions',
                    marker_color='royalblue',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[lob, lob], 
                    y=[0, 30], 
                    mode='lines', 
                    name=f'LoB: {lob:.6f}',
                    line=dict(color='red', dash='dash')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[lod, lod], 
                    y=[0, 30], 
                    mode='lines', 
                    name=f'LoD: {lod:.6f}',
                    line=dict(color='green', dash='dash')
                ),
                row=3, col=1
            )
            
            fig.update_xaxes(
                title_text='Predicted Tumor Fraction', 
                range=[0, min(0.1, lod * 5)],
                row=3, col=1
            )
            fig.update_yaxes(title_text='Count', row=3, col=1)
        else:
            fig.add_annotation(
                text="Not enough low samples for LoB/LoD visualisation",
                xref="x3", yref="y3",
                x=0.5, y=0.5,
                showarrow=False,
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text="Clinical Performance Evaluation",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def generate_clinical_report(evaluation_results):
    """
    Generate a human-readable clinical interpretation of model performance.
    
    Parameters:
    -----------
    evaluation_results : dict
        Output from evaluate_clinical_performance
        
    Returns:
    --------
    str
        Textual report of clinical performance
    """
    report = ["# Clinical Performance Report\n"]
    
    # Overall assessment
    if 'limit_of_detection' in evaluation_results:
        lod = evaluation_results['limit_of_detection']
        report.append(f"## Lower Limit of Detection\nThe model can reliably detect tumor fractions above {lod:.6f} (LoD).")
    
    # Performance by range
    report.append("\n## Performance Across Clinical Ranges")
    for range_name, metrics in evaluation_results['stratified_performance'].items():
        report.append(f"\n### {range_name.replace('_', ' ').title()} Range:")
        report.append(f"- Sample count: {metrics['count']}")
        report.append(f"- Median absolute error: {metrics['median_error']:.6f}")
        report.append(f"- Median relative error: {metrics['relative_error']:.1f}%")
    
    # Detection performance
    report.append("\n## Detection Performance at Clinical Decision Thresholds")
    for threshold, metrics in evaluation_results['detection_performance'].items():
        report.append(f"\n### At {threshold:.4f} threshold:")
        report.append(f"- Sensitivity: {metrics['sensitivity']:.2f}")
        report.append(f"- Specificity: {metrics['specificity']:.2f}")
        report.append(f"- Positive Predictive Value: {metrics['ppv']:.2f}")
        report.append(f"- Negative Predictive Value: {metrics['npv']:.2f}")
    
    # Clinical recommendations
    report.append("\n## Clinical Recommendations")
    
    # Identify poorest performing range
    performance_by_range = {r: m['mae'] for r, m in evaluation_results['stratified_performance'].items()}
    worst_range = max(performance_by_range, key=performance_by_range.get)
    
    report.append(f"- Exercise additional caution when interpreting results in the {worst_range.replace('_', ' ')} range.")
    
    # Add detection recommendation based on LoD
    if 'limit_of_detection' in evaluation_results:
        lod = evaluation_results['limit_of_detection']
        report.append(f"- Results below {lod:.6f} should be reported as 'below reliable detection limit'.")
    
    # Add recommendation based on specificity at low threshold
    lowest_threshold = min(evaluation_results['detection_performance'].keys())
    spec_at_lowest = evaluation_results['detection_performance'][lowest_threshold]['specificity']
    
    if spec_at_lowest < 0.95:
        report.append(f"- Consider confirmatory testing for positive results near the detection threshold due to specificity of {spec_at_lowest:.2f}.")
    
    return "\n".join(report)

# qrsh -b y -l h_vmem=2g -pe smp 32 -V -N train_t -wd /users/zetzioni/sharedscratch/deepconv/src -o ~/sharedscratch/logs/train_t.log 'cd /users/zetzioni/sharedscratch/deepconv/src && python -m deep_conv.detect.train \
# --name optimised_CpGenie_T-cells_1pct \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell \
# --data_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train_single_cell_clinical/T-cells/ \
# --atlas_path /users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed \
# --target_cell_type T-cells \
# --target_cell_idx 11 \
# --snr_profile low \
# --dropout_rate 0.15 \
# --feature_dim 128 \
# --num_heads 8 \
# --num_layers 3 \
# --detection_thresholds "0.001,0.005,0.01,0.02,0.05" \
# --critical_ranges "0.01,0.02,6.0;0.02,0.05,4.0;0.05,0.1,2.5" \
# --detection_loss_weight 0.4 \
# --min_reliable_coverage 3.0 \
# --enable_adaptive_thresholds \
# --early_stopping 30 \
# --grad_accum_steps 4 \
# --batch_size 32 \
# --weight_decay 0.02 \
# --epochs 300 \
# --lr 3.5e-4 \
# --save_interval 10'


# OAC
# qrsh -b y -l h_vmem=2g -pe smp 32 -V -N train_oac -wd /users/zetzioni/sharedscratch/deepconv/src -o ~/sharedscratch/logs/train_oac.log 'cd /users/zetzioni/sharedscratch/deepconv/src && python -m deep_conv.detect.train --name oac_simplified --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell --data_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train_single_cell_clinical/OAC/ --atlas_path /users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed --target_cell_type OAC --target_cell_idx 9 --dropout_rate 0.15 --feature_dim 128 --control_data_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/ --calibrate --calibrate_every 15 --early_stopping 30 --excluded_markers "44,58,111,133,77,95,127,38,108,115" --epochs 200 --weight_decay 0.01 --clinical_eval --generate_clinical_report --visualise_clinical'


def main():
    """
    Main function for training and evaluating the improved cell type concentration model
    """
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting improved cell type concentration model training")
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {args.output_dir}")

    logger.info(f"git info: {get_git_info()}")
    
    # Save arguments
    args_file = os.path.join(args.output_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Arguments saved to {args_file}")
    
    # Process excluded markers
    excluded_markers = parse_excluded_markers(args.excluded_markers)
    if excluded_markers:
        logger.info(f"Excluding {len(excluded_markers)} markers: {excluded_markers}")
    
    # Prepare data
    logger.info(f"Preparing data from {args.data_dir}...")
    try:
        train_loader, val_loader, test_loader, num_markers = prepare_data_for_training(
            data_dir=args.data_dir,
            atlas_path=args.atlas_path,
            target_cell_type=args.target_cell_type,
            target_cell_idx=args.target_cell_idx,
            excluded_markers=excluded_markers
        )
        logger.info(f"✓ Data preparation complete")
    except Exception as e:
        logger.error(f"× Error during data preparation: {str(e)}")
        raise
    
    # Load control data if provided
    control_val_loader = None
    if args.control_data_dir:
        logger.info(f"Loading control data from {args.control_data_dir}...")
        try:
            train_loader, control_val_loader = load_train_with_contrastive_data(
                train_loader, 
                args.control_data_dir, 
                args.atlas_path, 
                args.target_cell_type, 
                args.batch_size, 
                logger,
                excluded_markers=excluded_markers
            )
            logger.info(f"✓ Control data loaded successfully")
        except Exception as e:
            logger.error(f"× Error loading control data: {str(e)}")
            logger.info("  Continuing without control data")
    
    # Initialise improved model
    logger.info(f"Initializing improved model with {num_markers} markers...")
    try:
        model = EnhancedCancerDetectionModel(
            num_markers=num_markers,
            feature_dim=args.feature_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            min_reliable_coverage=args.min_reliable_coverage
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✓ Improved model initialised with {total_params:,} total parameters ({trainable_params:,} trainable)")
    except Exception as e:
        logger.error(f"× Error initializing improved model: {str(e)}")
        raise
    
    # Train model
    logger.info("Starting model training...")
    try:
        model, best_model_state = train_model(
            model, 
            train_loader, 
            val_loader, 
            control_val_loader, 
            args, 
            device
        )
        logger.info(f"✓ Training completed successfully")
    except Exception as e:
        logger.error(f"× Error during training: {str(e)}")
        raise
    
    # Perform final calibration if requested
    if args.calibrate:
        logger.info("Performing final model calibration...")
        try:
            calibration_results = model.calibrate(val_loader, device)
            
            # Update best model state with calibration results
            if best_model_state is not None:
                best_model_state['calibration'] = calibration_results
            
            # Save updated best model
            torch.save(best_model_state, os.path.join(args.output_dir, 'best_model_calibrated.pt'))
            
            logger.info(f"✓ Model calibrated:")
            logger.info(f"  Calibration factor: {calibration_results['calibration_factor']:.4f}")
            
        except Exception as e:
            logger.error(f"× Error during calibration: {str(e)}")
            logger.info("  Continuing without calibration")
    
    # Evaluate on test set
    logger.info("Evaluating final model on test set...")
    try:
        test_results = evaluate(model, test_loader, args, device, split_name="test")
        logger.info(f"✓ Test evaluation complete")
    except Exception as e:
        logger.error(f"× Error during test evaluation: {str(e)}")
        logger.info("  Skipping test evaluation")
    
    # Also collect predictions for clinical evaluation and visualisation
    test_preds = []
    test_targets = []
    test_lower_ci = []
    test_upper_ci = []
    test_attentions = []
    
    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            # Handle both dataset types
            if len(batch_data) == 4:
                marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask
            else:
                marker_values, coverage, y_true = batch_data
                
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            
            # Forward pass
            mu, uncertainty, attention_weights = model(marker_values, coverage)
            estimate, ci, _ = model.get_estimate_and_ci(mu, uncertainty)
            
            # Store results
            test_preds.append(estimate.cpu().numpy())
            test_targets.append(y_true.cpu().numpy())
            test_lower_ci.append(ci[:, 0:1].cpu().numpy())
            test_upper_ci.append(ci[:, 1:2].cpu().numpy())
            test_attentions.append(attention_weights.cpu().numpy())
    
    # Concatenate results
    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_lower_ci = np.concatenate(test_lower_ci)
    test_upper_ci = np.concatenate(test_upper_ci)
    test_attentions = np.concatenate(test_attentions)
    
    # Create visualisation directory
    vis_dir = os.path.join(args.output_dir, 'visualisations', 'test')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Call visualise_results
    try:
        visualise_results(
            test_preds, 
            test_targets, 
            vis_dir,
            ci_data=(test_lower_ci, test_upper_ci),
            marker_importance=test_attentions.mean(axis=0),
            prefix="Test "
        )
        logger.info(f"✓ Test visualisations created in {vis_dir}")
    except Exception as e:
        logger.error(f"× Error creating visualisations: {str(e)}")
    
    # Clinical evaluation
    if args.clinical_eval:
        logger.info("Performing clinical evaluation...")
        clinical_vis_dir = os.path.join(args.output_dir, 'visualisations', 'clinical')
        os.makedirs(clinical_vis_dir, exist_ok=True)
        
        try:
            # Load blank samples if provided
            blank_samples = None
            if args.blank_samples_path:
                logger.info(f"Loading blank samples from {args.blank_samples_path}...")
                try:
                    blank_samples = np.load(args.blank_samples_path)
                    logger.info(f"✓ Loaded {len(blank_samples)} blank samples")
                except Exception as e:
                    logger.error(f"× Error loading blank samples: {str(e)}")
                    logger.info("  Continuing without blank samples")
            
            # Perform clinical evaluation
            logger.info("Calculating clinical performance metrics...")
            clinical_results = evaluate_clinical_performance(
                test_targets.flatten(), 
                test_preds.flatten(),
                blank_samples=blank_samples
            )
            
            # Save clinical results
            clinical_results_file = os.path.join(args.output_dir, 'clinical_results.pkl')
            with open(clinical_results_file, 'wb') as f:
                pickle.dump(clinical_results, f)
            logger.info(f"✓ Clinical results saved to {clinical_results_file}")
            
            # Generate clinical visualisations if requested
            if args.visualise_clinical:
                logger.info("Generating clinical visualisations...")
                fig = visualise_clinical_performance(clinical_results)
                fig_path = os.path.join(clinical_vis_dir, 'clinical_performance.html')
                fig.write_html(fig_path)
                logger.info(f"✓ Clinical visualisations saved to {fig_path}")
            
            # Generate clinical report if requested
            if args.generate_clinical_report:
                logger.info("Generating clinical report...")
                report = generate_clinical_report(clinical_results)
                report_path = os.path.join(args.output_dir, 'clinical_report.md')
                with open(report_path, 'w') as f:
                    f.write(report)
                logger.info(f"✓ Clinical report saved to {report_path}")
            
            # Log key clinical metrics
            if 'limit_of_detection' in clinical_results:
                logger.info(f"  Lower Limit of Detection (LoD): {clinical_results['limit_of_detection']:.6f}")
            
            # Log performance in ultra-low range if available
            if 'stratified_performance' in clinical_results and 'ultra_low' in clinical_results['stratified_performance']:
                ultra_low = clinical_results['stratified_performance']['ultra_low']
                logger.info(f"  Ultra-low range (<0.001) performance:")
                logger.info(f"    Samples: {ultra_low['count']}")
                logger.info(f"    MAE: {ultra_low['mae']:.6f}")
                logger.info(f"    Relative error: {ultra_low['relative_error']:.1f}%")
            
            # Log detection performance at lowest threshold
            if 'detection_performance' in clinical_results:
                lowest_threshold = min(clinical_results['detection_performance'].keys())
                metrics = clinical_results['detection_performance'][lowest_threshold]
                logger.info(f"  Detection at {lowest_threshold:.4f} threshold:")
                logger.info(f"    Sensitivity: {metrics['sensitivity']:.2f}")
                logger.info(f"    Specificity: {metrics['specificity']:.2f}")
                logger.info(f"    PPV: {metrics['ppv']:.2f}")
                
        except Exception as e:
            logger.error(f"× Error during clinical evaluation: {str(e)}")
            logger.info("  Continuing without clinical evaluation")
    
    logger.info("\n" + "="*60)
    logger.info("IMPROVED MODEL TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
