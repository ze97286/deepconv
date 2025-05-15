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
from scipy import stats
import plotly.subplots as sp
from sklearn.metrics import r2_score, mean_absolute_error, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, mean_squared_error, confusion_matrix
import plotly.subplots as sp
import pickle
from sklearn.metrics import confusion_matrix
from plotly.subplots import make_subplots

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
    parser.add_argument('--excluded_markers', type=str, default='0,2,3,5,9,10,11,13,14,16,18,21,22,23,25,27,28,29,30,31,32,34,38,39,40,42,43,44,46,47,48,50,51,52,54,55,56,58,59,61,62,64,65,67,70,74,75,76,77,78,82,83,84,85,86,87,89,91,94,95,97,98,100,101,102,103,104,105,108,109,111,112,113,114,115,116,117,118,120,121,122,124,125,126,127,128,129,133,135', help='Comma-separated list of marker indices to exclude')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=64, help='Feature dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularisation')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--min_reliable_coverage', type=float, default=3.0, 
                   help='Minimum coverage considered reliable for marker values')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimiser')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default="./saved_models", help='Output directory')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    
    parser.add_argument('--control_data_dir', type=str, default=None, 
                   help='Directory containing control data for contrastive learning')
    parser.add_argument('--calibrate_clinical_threshold', action='store_true',
                        help='Calibrate clinical decision threshold in addition to uncertainty')
    parser.add_argument('--target_specificity', type=float, default=0.95,
                        help='Target specificity for clinical threshold calibration')
    parser.add_argument('--calibrate', action='store_true', 
                    help='Calibrate confidence intervals')
    parser.add_argument('--calibrate_every', type=int, default=5,
                    help='Calibrate model every N epochs')

    # Clinical evaluation parameters (these were in the main function)
    parser.add_argument('--clinical_eval', action='store_true',
                       help='Enable comprehensive clinical evaluation metrics')
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

def compute_loss(mu, uncertainty, y_true, control_mask=None, zero_prob=None):
    """
    Simplified loss function focused on relative error
    """
    # Base MSE loss
    mse_loss = F.mse_loss(mu, y_true, reduction='none')
    
    # Calculate relative error for non-zero targets
    epsilon = 1e-6
    non_zero_mask = (y_true > epsilon)
    
    if non_zero_mask.sum() > 0:
        # Relative error
        rel_error = torch.abs(mu[non_zero_mask] - y_true[non_zero_mask]) / (y_true[non_zero_mask] + epsilon)
        rel_loss = rel_error.mean()
    else:
        rel_loss = torch.tensor(0.0, device=mu.device)
    
    control_loss = torch.tensor(0.0, device=mu.device)
    if control_mask is not None and control_mask.sum() > 0:
        control_loss = 100.0 * mu[control_mask].mean()
    
    # Zero probability supervision (if available)
    zero_loss = torch.tensor(0.0, device=mu.device)
    if zero_prob is not None:
        zero_target = (y_true < epsilon).float()
        zero_loss = F.binary_cross_entropy(zero_prob.squeeze(), zero_target.squeeze()) * 5.0
    
    # Uncertainty calibration
    if uncertainty is not None:
        z_scores = torch.abs(mu - y_true) / (uncertainty + epsilon)
        calibration_loss = F.smooth_l1_loss(z_scores, torch.ones_like(z_scores) * 1.96)
    else:
        calibration_loss = torch.tensor(0.0, device=mu.device)
    
    # Combine losses - increase weight on relative error
    total_loss = mse_loss.mean() + 2.0 * rel_loss + control_loss + zero_loss + 0.2 * calibration_loss
    
    return total_loss

def train_model(model, train_loader, val_loader, args, device):
    """
    Train the model with log-space metrics to improve performance at low concentrations
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
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
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    # Initialise tracking variables
    best_composite = float('-inf')
    best_composite_components = {}

    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'mae': [],
        'r2_score': [],
        'low_conc_error': [],
        'log_r2_score': [],
        'log_slope': [],
        'log_intercept': []
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

                # Forward pass with the new model outputs
                mu, uncertainty, _, zero_prob = model(marker_values, coverage)

                # Calculate loss with control mask and zero probability
                loss = compute_loss(mu, uncertainty, y_true, control_mask, zero_prob)

            else:  # Standard dataset without control_mask
                marker_values, coverage, y_true = batch_data
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)

                # Forward pass
                mu, uncertainty, _, zero_prob = model(marker_values, coverage)

                # Calculate loss without control mask
                loss = compute_loss(mu, uncertainty, y_true, None, zero_prob)

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

        # Add log-space metrics to history
        history['log_r2_score'].append(val_metrics.get('log_r2', 0.0))
        history['log_slope'].append(val_metrics.get('log_slope', 0.0))
        history['log_intercept'].append(val_metrics.get('log_intercept', 0.0))

        # Log validation results with added log-space metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"MAE: {val_metrics['mae']:.6f}, "
            f"R²: {val_metrics['r2']:.4f}, "
            f"Log-R²: {val_metrics.get('log_r2', 0.0):.4f}, "
            f"Log-Slope: {val_metrics.get('log_slope', 0.0):.4f}, "
            f"Log-Intercept: {val_metrics.get('log_intercept', 0.0):.4f}, "
            f"Low-Conc error: {low_conc_error:.6f}"
        )

        # Log concentration-specific metrics for critical ranges
        for range_name in ['0.1-0.5%', '0.5-1%', '1-5%']:
            if range_name in val_metrics['concentration_metrics']:
                metrics = val_metrics['concentration_metrics'][range_name]
                within_25 = metrics.get('within_25pct', 0.0) 
                logger.info(f"  {range_name} (n={metrics['count']}): "
                        f"MAE={metrics['mae']:.6f}, "
                        f"Within 25%={within_25:.1f}%")

        # Check for improvement - now including log-space metrics
        improvement = False
        improvement_msg = ""

        log_slope = val_metrics.get('log_slope', 0.0)
        log_intercept = val_metrics.get('log_intercept', 0.0)
        log_r2 = val_metrics.get('log_r2', 0.0)

        composite_score = (
            -0.5 * val_loss +                      # Lower loss is better (negative weight)
            0.2 * log_r2 +                         # Higher log-R² is better
            -1.0 * abs(log_slope - 1.0) +          # Closer to slope=1 is better
            -1.0 * abs(log_intercept) +            # Closer to intercept=0 is better
            -0.3 * low_conc_error                  # Lower error in 0.1-1% range is better
        )

        composite_score_components = {
            "val_loss": val_loss,
            "log_r2": log_r2,   
            "log_slope": log_slope,
            "log_intercept": log_intercept,
            "low_conc_error": low_conc_error
        }

        # Check improvement in validation loss
        if composite_score > best_composite:
            improvement = True
            improvement_msg = (
                f"New best model (composite)!"
                + f"\ncomposite score: {best_composite} -> {composite_score}"
                + f"\nloss: {best_composite_components.get('val_loss',float('inf')):.4f} → {val_loss:.4f}"
                + f"\nlog_r2: {best_composite_components.get('log_r2',0):.4f} → {log_r2:.4f}"
                + f"\nlog_slope: {best_composite_components.get('log_slope',0):.4f} → {log_slope:.4f}"
                + f"\nlog_intercept: {best_composite_components.get('log_intercept',0):.4f} → {log_intercept:.4f}"
                + f"\nlow_conc_error: {best_composite_components.get('low_conc_error',0):.4f} → {low_conc_error:.4f}"
            )

            best_composite  = composite_score
            best_composite_components = composite_score_components

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
            logger.info(
                f"× No improvement (composite score: {composite_score:.4f} best composite score: {best_composite:.4f}). Patience: {patience_counter}/{args.early_stopping}"
            )

        # Periodic calibration (if enabled)
        if args.calibrate and (epoch % args.calibrate_every == 0 or epoch == args.epochs - 1):
            logger.info("Calibrating model uncertainty estimates...")
            try:
                calibration_results = model.calibrate(val_loader, device)
                logger.info(f"  Calibration factor: {calibration_results['calibration_factor']:.4f}")

                # Also calibrate clinical threshold if enabled
                if hasattr(args, 'calibrate_clinical_threshold') and args.calibrate_clinical_threshold:
                    logger.info("Calibrating clinical decision threshold...")
                    clinical_calibration = model.calibrate_clinical_threshold(
                        val_loader,
                        device,
                        target_metric='concentration_aware'
                    )

                    if clinical_calibration:
                        logger.info(f"  Clinical threshold: {clinical_calibration['threshold']:.6f}")
                        logger.info(f"  Specificity: {clinical_calibration['specificity']:.2f}")
                        logger.info(f"  Sensitivity: {clinical_calibration['sensitivity']:.2f}")
                    else:
                        logger.warning("  Failed to calibrate clinical threshold")

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
    and log-space metrics for better low concentration evaluation
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        device: Device to run validation on
        
    Returns:
        Dictionary of validation metrics including log-space metrics
    """
    logger = logging.getLogger('cell_detection')
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    all_uncertainties = []
    all_clinical_detections = []
    all_zero_probs = []
    
    # Track batch count for reporting
    batch_count = 0
    total_batches = len(val_loader)
    logger.info(f"Starting validation on {total_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            try:
                # Extract data safely regardless of format
                marker_values = batch_data[0].to(device)
                coverage = batch_data[1].to(device)
                y_true = batch_data[2].to(device)
                
                # Forward pass with clinical threshold if available
                if hasattr(model, 'predict_with_clinical_threshold'):
                    mu, uncertainty, _, is_detected = model.predict_with_clinical_threshold(
                        marker_values, coverage
                    )
                    all_clinical_detections.append(is_detected.cpu().numpy())
                else:
                    # Use the new model output format
                    mu, uncertainty, _, zero_prob = model(marker_values, coverage)
                    all_zero_probs.append(zero_prob.cpu().numpy())
                
                # Compute loss safely
                try:
                    # Use control_mask if available (4th element)
                    if len(batch_data) > 3:
                        control_mask = batch_data[3].to(device)
                        if hasattr(model, 'predict_with_clinical_threshold'):
                            batch_loss = compute_loss(mu, uncertainty, y_true, control_mask)
                        else:
                            batch_loss = compute_loss(mu, uncertainty, y_true, control_mask, zero_prob)
                    else:
                        if hasattr(model, 'predict_with_clinical_threshold'):
                            batch_loss = compute_loss(mu, uncertainty, y_true)
                        else:
                            batch_loss = compute_loss(mu, uncertainty, y_true, None, zero_prob)
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
            'uncertainty_metrics': {},
            'clinical_metrics': {},
            'log_r2': 0.0,
            'log_slope': 0.0,
            'log_intercept': 0.0
        }
    
    # Calculate average loss
    val_loss /= batch_count
    
    try:
        # Concatenate predictions and targets
        predictions = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        uncertainties = np.concatenate(all_uncertainties)
        
        # Calculate standard regression metrics
        r2 = r2_score(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        # Calculate log-space metrics
        epsilon = 1e-6
        non_zero_mask = (targets > epsilon) & (predictions > epsilon)
        
        # Default values in case there are no valid points
        log_r2 = 0.0
        log_slope = 0.0
        log_intercept = 0.0
        
        log_space_results = calculate_log_space_metrics(predictions[non_zero_mask], targets[non_zero_mask])
        log_r2 = log_space_results['log_r2']
        log_slope = log_space_results['log_slope']
        log_intercept = log_space_results['log_intercept']
        
        # Calculate concentration metrics
        
        # Calculate concentration-stratified metrics
        concentration_metrics = compute_concentration_metrics(predictions, targets)
        
        # Update within 25% calculation for each range to ensure it's consistent
        for range_name, stats in concentration_metrics.items():
            # Extract concentration range bounds
            range_parts = range_name.replace('%', '').split('-')
            if len(range_parts) == 2:
                try:
                    low = float(range_parts[0]) / 100.0
                    high = float(range_parts[1]) / 100.0 if range_parts[1] != '' else float('inf')
                    
                    # Get predictions and targets in this range
                    range_mask = (targets >= low) & (targets < high)
                    range_targets = targets[range_mask]
                    range_preds = predictions[range_mask]
                    
                    # Calculate within 25% metric
                    rel_errors = np.abs(range_preds - range_targets) / (range_targets + epsilon)
                    within_25pct = 100.0 * np.mean(rel_errors <= 0.25)
                    
                    # Ensure the key is consistent
                    concentration_metrics[range_name]['within_25pct'] = within_25pct
                    
                    # Calculate log-space metrics for this range
                    range_non_zero = (range_targets > epsilon) & (range_preds > epsilon)
                    if range_non_zero.sum() > 10:
                        range_log_pred = np.log10(range_preds[range_non_zero])
                        range_log_true = np.log10(range_targets[range_non_zero])
                        
                        range_log_r2 = r2_score(range_log_true, range_log_pred)
                        
                        try:
                            range_poly_coeffs = np.polyfit(range_log_true, range_log_pred, 1)
                            range_log_slope = range_poly_coeffs[0]
                            range_log_intercept = range_poly_coeffs[1]
                            
                            # Add to metrics
                            concentration_metrics[range_name]['log_r2'] = range_log_r2
                            concentration_metrics[range_name]['log_slope'] = range_log_slope
                            concentration_metrics[range_name]['log_intercept'] = range_log_intercept
                        except Exception as e:
                            logger.warning(f"Could not calculate log metrics for range {range_name}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error parsing range {range_name}: {str(e)}")
        
        # Calculate uncertainty calibration metrics
        uncertainty_metrics = compute_uncertainty_metrics(predictions, targets, uncertainties)
        
        # Clinical detection metrics if available
        clinical_metrics = {}
        if len(all_clinical_detections) > 0:
            clinical_detections = np.concatenate(all_clinical_detections)
            true_positives = (clinical_detections & (targets >= 0.001)).sum()
            false_positives = (clinical_detections & (targets < 0.001)).sum()
            true_negatives = ((~clinical_detections) & (targets < 0.001)).sum()
            false_negatives = ((~clinical_detections) & (targets >= 0.001)).sum()
            
            clinical_metrics = {
                'sensitivity': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
                'specificity': true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0,
                'ppv': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
                'npv': true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
            }
        
        # Zero probability metrics if available
        zero_prob_metrics = {}
        if all_zero_probs:
            zero_probs = np.concatenate(all_zero_probs)
            zero_mask = targets < epsilon
            if np.any(zero_mask):
                zero_prob_metrics = {
                    'zero_prob_accuracy': np.mean((zero_probs[zero_mask] > 0.5).astype(float)),
                    'zero_mean_prob': np.mean(zero_probs[zero_mask]),
                    'nonzero_mean_prob': np.mean(zero_probs[~zero_mask])
                }
                # Log zero probability metrics
                logger.info(f"Zero prob metrics - Accuracy: {zero_prob_metrics['zero_prob_accuracy']:.4f}, "
                           f"Mean prob on zeros: {zero_prob_metrics['zero_mean_prob']:.4f}")
        
        # Log success
        logger.info(f"Validation complete: processed {batch_count}/{total_batches} batches")
        
        # Log log-space metrics
        logger.info(f"Log-space metrics - R²: {log_r2:.4f}, Slope: {log_slope:.4f}, Intercept: {log_intercept:.4f}")
        
        # Return all metrics including log-space metrics
        return {
            'loss': val_loss,
            'r2': r2,
            'mae': mae,
            'concentration_metrics': concentration_metrics,
            'uncertainty_metrics': uncertainty_metrics,
            'clinical_metrics': clinical_metrics,
            'zero_prob_metrics': zero_prob_metrics,
            'log_r2': log_r2,
            'log_slope': log_slope,
            'log_intercept': log_intercept
        }
    except Exception as e:
        logger.error(f"Error calculating validation metrics: {str(e)}")
        # Return basic metrics that were calculated
        return {
            'loss': val_loss,
            'r2': 0.0,
            'mae': float('inf'),
            'concentration_metrics': {},
            'uncertainty_metrics': {},
            'clinical_metrics': {},
            'log_r2': 0.0,
            'log_slope': 0.0,
            'log_intercept': 0.0
        }

def compute_concentration_metrics(predictions, targets):
    """
    Compute concentration-stratified metrics
    """
    ranges = [
        (0, 0.0005, "0-0.05%"),
        (0.0005, 0.001, "0.05-0.1%"),
        (0.001, 0.005, "0.1-0.5%"),
        (0.005, 0.01, "0.5-1%"),
        (0.01, 0.05, "1-5%"),
        (0.05, 0.1, "5-10%"),
        (0.1, 1.0, ">10%")
    ]
    
    results = {}
    
    for low, high, name in ranges:
        mask = (targets >= low) & (targets < high)
        range_preds = predictions[mask]
        range_targets = targets[mask]
        
        if len(range_targets) == 0:
            continue
            
        mae = np.mean(np.abs(range_preds - range_targets))
        
        within_pct = {}
        non_zero_mask = range_targets > 0
        if np.sum(non_zero_mask) > 0:
            rel_errors = np.abs(range_preds[non_zero_mask] - range_targets[non_zero_mask]) / range_targets[non_zero_mask]
            
            for threshold in [0.1, 0.25, 0.5]:
                within_pct[f"{int(threshold*100)}pct"] = float(np.mean(rel_errors <= threshold) * 100)
        
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
    Enhanced evaluation function focused on regression metrics with clinical threshold evaluation
    
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
    all_clinical_detections = []
    
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
            
            # Forward pass with clinical threshold if available
            if hasattr(model, 'predict_with_clinical_threshold'):
                mu, uncertainty, attention_weights, is_detected = model.predict_with_clinical_threshold(
                    marker_values, coverage
                )
                all_clinical_detections.append(is_detected.cpu().numpy())
            else:
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
    
    # Clinical detection metrics if available
    clinical_metrics = {}
    if len(all_clinical_detections) > 0:
        clinical_detections = np.concatenate(all_clinical_detections)
        true_positives = (clinical_detections & (targets >= 0.001)).sum()
        false_positives = (clinical_detections & (targets < 0.001)).sum()
        true_negatives = ((~clinical_detections) & (targets < 0.001)).sum()
        false_negatives = ((~clinical_detections) & (targets >= 0.001)).sum()
        
        clinical_metrics = {
            'sensitivity': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
            'specificity': true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0,
            'ppv': true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0,
            'npv': true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        }
        
        logger.info(f"\nCLINICAL DETECTION METRICS:")
        logger.info(f"  Sensitivity: {clinical_metrics['sensitivity']:.2f}")
        logger.info(f"  Specificity: {clinical_metrics['specificity']:.2f}")
        logger.info(f"  PPV: {clinical_metrics['ppv']:.2f}")
        logger.info(f"  NPV: {clinical_metrics['npv']:.2f}")
    
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
        'clinical_metrics': clinical_metrics,
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
    
    # Create visualisations
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

def calculate_log_space_metrics(predictions, targets, epsilon=1e-6):
    """Calculate standardized log-space metrics for both training and plotting.
    
    Args:
        predictions: numpy array of model predictions
        targets: numpy array of true values
        epsilon: small value to avoid log(0)
        
    Returns:
        Dictionary with log_r2, log_slope, log_intercept
    """
    # Filter to ensure we only use valid positive values
    mask = (targets > epsilon) & (predictions > epsilon)
    
    if mask.sum() <= 10:  # Need enough points for reliable regression
        return {
            'log_r2': 0.0,
            'log_slope': 0.0,
            'log_intercept': 0.0
        }
    
    # Transform to log space using base 10
    log_y_true = np.log10(targets[mask])
    log_y_pred = np.log10(predictions[mask])
    
    # Calculate R² in log space
    log_r2 = r2_score(log_y_true, log_y_pred)
    
    # Calculate slope and intercept using numpy polyfit
    coeffs = np.polyfit(log_y_true, log_y_pred, 1)
    log_slope = coeffs[0]
    log_intercept = coeffs[1]
    
    return {
        'log_r2': log_r2,
        'log_slope': log_slope,
        'log_intercept': log_intercept
    }

def visualise_results(predictions, ground_truth, output_dir, ci_data=None, marker_importance=None, prefix=""):
    """
    Enhanced unified visualisation function incorporating clinical metrics and ROC curves
    
    Args:
        predictions: Array of predicted cell type concentrations
        ground_truth: Array of true cell type concentrations
        output_dir: Directory to save visualisations
        ci_data: Optional tuple of (lower_ci, upper_ci) for confidence interval visualisation
        marker_importance: Optional marker importance data
        prefix: Optional prefix for output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create clinical directory
    clinical_dir = os.path.join(output_dir, 'clinical')
    os.makedirs(clinical_dir, exist_ok=True)
    # Flatten arrays
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
    
    # 6. Error distribution analysis
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
    
    # 7. Calculate log-space metrics
    epsilon = 1e-6  
    non_zero_mask = (targets > epsilon) & (preds > epsilon)
    valid_targets = targets[non_zero_mask]
    valid_predictions = preds[non_zero_mask]

    # Transform to log space
    log_targets = np.log10(valid_targets)
    log_preds = np.log10(valid_predictions)

    # Calculate log-space metrics
    log_space_results = calculate_log_space_metrics(targets, preds)
    log_r2 = log_space_results['log_r2']
    slope = log_space_results['log_slope']
    intercept = log_space_results['log_intercept']

    # Calculate relative errors for coloring
    rel_errors = 100 * (valid_predictions - valid_targets) / valid_targets

    # Create log-log plot
    fig_log = go.Figure()

    # Add scatter points
    fig_log.add_trace(
        go.Scatter(
            x=log_targets,
            y=log_preds,
            mode='markers',
            marker=dict(
                color=rel_errors.clip(-50, 50),  # Use capped relative error for color
                colorscale='RdBu_r',
                cmin=-50,
                cmax=50,
                colorbar=dict(
                    title='Relative Error (%)',
                    titleside='right',
                    thickness=20,
                    len=0.75,
                    y=0.5,
                    yanchor='middle'
                )
            ),
            name='Samples',
            showlegend=False  # Hide from legend as it's shown in colorbar
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
            line=dict(color='red', dash='dash', width=2),
            name=f'Regression Line (slope={slope:.2f})'
        )
    )

    # Add identity line (slope=1, intercept=0)
    fig_log.add_trace(
        go.Scatter(
            x=x_range,
            y=x_range,
            mode='lines',
            line=dict(color='black', dash='dot', width=1.5),
            name='Perfect Prediction'
        )
    )

    # Add annotation for log-space metrics in a box
    fig_log.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"<b>Log-Space R² = {log_r2:.4f}</b><br><b>Slope = {slope:.3f}</b><br><b>Intercept = {intercept:.3f}</b>",
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="left",
        opacity=0.8
    )

    # Update layout with better legend positioning
    fig_log.update_layout(
        title='Log-Space Analysis for Linearity at Low Concentrations',
        xaxis_title='Log10(True Concentration)',
        yaxis_title='Log10(Predicted Concentration)',
        template='plotly_white',
        width=900,
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        margin=dict(t=100, r=80, b=80, l=80)  # Add margin to ensure colorbar doesn't get cut off
    )

    # Make axes more proportional
    fig_log.update_layout(
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            range=[min(log_preds)-0.5, max(log_preds)+0.5]
        ),
        xaxis=dict(
            range=[min(log_targets)-0.5, max(log_targets)+0.5]
        )
    )

    # Save figure
    fig_log.write_html(os.path.join(clinical_dir, 'log_space_analysis.html'))
    fig_log.write_image(os.path.join(clinical_dir, 'log_space_analysis.png'), scale=2)
    
    # 8. Add ROC curve analysis for detection at various thresholds
    detection_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    detection_metrics = {}
    
    for thresh in detection_thresholds:
        y_true_binary = (targets >= thresh).astype(int)
        
        # Skip if no positive examples
        if sum(y_true_binary) == 0:
            continue
            
        # ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(y_true_binary, preds)
        roc_auc = auc(fpr, tpr)
        
        # Find sensitivity at 95% specificity (5% FPR)
        idx_95spec = np.argmin(np.abs(fpr - 0.05))
        sens_at_95spec = tpr[idx_95spec]
        
        # Precision-recall curve and average precision
        precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, preds)
        ap = average_precision_score(y_true_binary, preds)
        
        detection_metrics[thresh] = {
            'auc': float(roc_auc),
            'sensitivity_at_95spec': float(sens_at_95spec),
            'average_precision': float(ap),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
    
    # Create ROC curve plot
    create_enhanced_roc_curve(detection_metrics, clinical_dir, title_prefix=prefix)
    
    # 9. Add magnitude-aware metrics
    create_magnitude_aware_metrics(df, detection_thresholds, clinical_dir)
    
    # 10. Add clinical decision metrics
    clinical_metrics = create_clinical_decision_metrics(df, detection_thresholds, clinical_dir)
    
    # 11. Add threshold-specific detailed analysis for key thresholds
    specific_thresholds = [0.001, 0.005, 0.01, 0.05]  # Key clinical thresholds
    create_threshold_specific_analysis(df, specific_thresholds, clinical_dir)
    
    # Return metrics dictionary with clinical additions
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
        'log_space': {
            'r2': float(log_r2),
            'slope': float(slope),
            'intercept': float(intercept)
        },
        'detection_metrics': detection_metrics,
        'clinical_metrics': clinical_metrics
    }
    
    # Add CI metrics if available
    if ci_data is not None:
        metrics['in_ci_percentage'] = float(in_ci)
        metrics['ci_width'] = float(ci_width)
    
    return metrics

def create_enhanced_roc_curve(detection_metrics, output_dir, title_prefix=""):
    """
    Create enhanced ROC curve with consistent metrics from detection_metrics 
    """
    # Create figure
    fig = go.Figure()
    
    # Add diagonal reference line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', dash='dash'),
            showlegend=False
        )
    )
    
    # Add ROC curves for each threshold from the pre-calculated metrics
    for threshold, metrics in sorted(detection_metrics.items()):
        if 'fpr' in metrics and 'tpr' in metrics:
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            auc_value = metrics['auc']
            
            # Add to plot
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'≥{threshold:.1%}, AUC={auc_value:.3f}',
                    line=dict(width=2)
                )
            )
    
    # Create summary text with metrics
    primary_threshold = 0.01  # Default to 1% as primary threshold
    
    # Use the provided metrics if available, otherwise use a default message
    if primary_threshold in detection_metrics:
        metrics = detection_metrics[primary_threshold]
        
        # Create annotation text
        annotation_text = (
            f"<b>ROC Analysis Summary</b><br>"
            f"AUC at {primary_threshold:.1%}: {metrics['auc']:.3f}<br>"
            f"Sensitivity at 95% specificity: {metrics['sensitivity_at_95spec']:.3f}<br>"
        )
        
        # Add metrics for each threshold
        for thresh, thresh_metrics in sorted(detection_metrics.items()):
            annotation_text += (
                f"For ≥{thresh:.1%}: "
                f"AUC = {thresh_metrics['auc']:.3f}, "
                f"Sens@95%Spec = {thresh_metrics['sensitivity_at_95spec']:.3f}<br>"
            )
    else:
        annotation_text = "Metrics not available for standard thresholds"
    
    # Add annotation to figure
    fig.add_annotation(
        x=0.5,
        y=0.1,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )
    
    # Customise layout
    title = "ROC Curve by Concentration Threshold"
    if title_prefix:
        title = f"{title_prefix} - {title}"
        
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        autosize=False,
        width=900,
        height=700,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'enhanced_roc_curve.html'))
    fig.write_image(os.path.join(output_dir, 'enhanced_roc_curve.png'), scale=2)


def create_magnitude_aware_metrics(df, thresholds, output_dir):
    """
    Create metrics and plot for magnitude-aware classification performance.
    This considers both binary detection and the magnitude of errors.
    """
    # Define error tolerance levels (as fraction of true value)
    error_tolerances = [0.1, 0.2, 0.5, 1.0, 2.0]  # 10%, 20%, 50%, 100%, 200%
    
    # Initialize results dictionary
    results = {}
    
    # For each concentration threshold
    for threshold in thresholds:
        tolerance_results = {}
        
        # For each error tolerance
        for tolerance in error_tolerances:
            # True positive: predicted ≥ threshold when true ≥ threshold AND within tolerance
            tp_mask = (df['true_value'] >= threshold) & (df['predicted_value'] >= threshold) & \
                     (df['rel_error'] <= tolerance * 100)
            
            # False positive: predicted ≥ threshold when true < threshold OR exceeds tolerance
            fp_mask = ((df['true_value'] < threshold) & (df['predicted_value'] >= threshold)) | \
                     ((df['true_value'] >= threshold) & (df['predicted_value'] >= threshold) & \
                      (df['rel_error'] > tolerance * 100))
            
            # True negative: predicted < threshold when true < threshold
            tn_mask = (df['true_value'] < threshold) & (df['predicted_value'] < threshold)
            
            # False negative: predicted < threshold when true ≥ threshold
            fn_mask = (df['true_value'] >= threshold) & (df['predicted_value'] < threshold)
            
            # Calculate counts
            tp = tp_mask.sum()
            fp = fp_mask.sum()
            tn = tn_mask.sum()
            fn = fn_mask.sum()
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # Store metrics
            tolerance_results[tolerance] = {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'f1_score': float(f1_score),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
        
        results[threshold] = tolerance_results
    
    # Create plot for key threshold (1%)
    create_magnitude_aware_plot(results, output_dir)
    
    # Save results to file
    magnitude_metrics_file = os.path.join(output_dir, 'magnitude_aware_metrics.json')
    with open(magnitude_metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_magnitude_aware_plot(results, output_dir):
    """
    Create plot for magnitude-aware classification metrics
    """
    # Select a primary threshold for visualization (typically 1%)
    primary_threshold = 0.01
    if primary_threshold not in results:
        # Use the first available threshold
        primary_threshold = list(results.keys())[0]
    
    # Prepare data for plotting
    tolerance_metrics = results[primary_threshold]
    tolerances = sorted(float(t) for t in tolerance_metrics.keys())
    
    sensitivity = [tolerance_metrics[t]['sensitivity'] for t in tolerances]
    specificity = [tolerance_metrics[t]['specificity'] for t in tolerances]
    precision = [tolerance_metrics[t]['precision'] for t in tolerances]
    f1_score = [tolerance_metrics[t]['f1_score'] for t in tolerances]
    
    # Convert tolerances to percentage labels for x-axis
    tolerance_labels = [f"{t*100:.0f}%" for t in tolerances]
    
    # Create figure
    fig = go.Figure()
    
    # Add metric lines
    fig.add_trace(
        go.Scatter(
            x=tolerance_labels,
            y=sensitivity,
            mode='lines+markers',
            name='Sensitivity',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=tolerance_labels,
            y=specificity,
            mode='lines+markers',
            name='Specificity',
            line=dict(color='red', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=tolerance_labels,
            y=precision,
            mode='lines+markers',
            name='Precision',
            line=dict(color='green', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=tolerance_labels,
            y=f1_score,
            mode='lines+markers',
            name='F1 Score',
            line=dict(color='purple', width=2)
        )
    )
    
    # Customize layout
    fig.update_layout(
        title=f'Magnitude-Aware Classification Metrics at {primary_threshold:.1%} Threshold',
        xaxis_title='Error Tolerance',
        yaxis_title='Metric Value',
        template='plotly_white',
        autosize=False,
        width=900,
        height=600,
        yaxis=dict(range=[0, 1]),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'magnitude_aware_metrics.html'))
    fig.write_image(os.path.join(output_dir, 'magnitude_aware_metrics.png'), scale=2)


def create_clinical_decision_metrics(df, thresholds, output_dir):
    """
    Create metrics specifically focused on clinical decision making.
    For each concentration threshold, calculate the PPV, NPV, and likelihood ratios.
    
    Args:
        df: DataFrame with 'true_value' and 'predicted_value' columns
        thresholds: List of concentration thresholds to evaluate
        output_dir: Directory to save visualizations
    
    Returns:
        Dictionary of clinical decision metrics
    """
    # Initialize results dictionary
    clinical_metrics = {}
    
    # For each concentration threshold
    for threshold in thresholds:
        # Create contingency table
        true_positive = ((df['true_value'] >= threshold) & (df['predicted_value'] >= threshold)).sum()
        false_positive = ((df['true_value'] < threshold) & (df['predicted_value'] >= threshold)).sum()
        true_negative = ((df['true_value'] < threshold) & (df['predicted_value'] < threshold)).sum()
        false_negative = ((df['true_value'] >= threshold) & (df['predicted_value'] < threshold)).sum()
        
        # Calculate primary metrics
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        
        # Calculate positive predictive value (PPV) - critical for clinical question
        ppv = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        
        # Calculate negative predictive value (NPV)
        npv = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0
        
        # Calculate likelihood ratios
        positive_lr = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')
        negative_lr = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
        
        # Store metrics
        clinical_metrics[threshold] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),  # This answers "If test is positive, what's the probability it's truly positive?"
            'npv': float(npv),  # This answers "If test is negative, what's the probability it's truly negative?"
            'positive_lr': float(positive_lr),
            'negative_lr': float(negative_lr),
            'true_positive': int(true_positive),
            'false_positive': int(false_positive),
            'true_negative': int(true_negative),
            'false_negative': int(false_negative),
            'prevalence': float((true_positive + false_negative) / len(df))
        }
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'clinical_decision_metrics.json'), 'w') as f:
        json.dump(clinical_metrics, f, indent=2)
    
    # Create visualization for key metrics
    create_clinical_decision_plot(clinical_metrics, output_dir)
    
    return clinical_metrics


def create_clinical_decision_plot(clinical_metrics, output_dir):
    """
    Create a plot showing key clinical decision metrics (PPV, NPV) across thresholds
    """
    # Extract thresholds and metrics
    thresholds = sorted(float(t) for t in clinical_metrics.keys())
    sensitivity = [clinical_metrics[t]['sensitivity'] for t in thresholds]
    specificity = [clinical_metrics[t]['specificity'] for t in thresholds]
    ppv = [clinical_metrics[t]['ppv'] for t in thresholds]  # This is key for clinical question
    npv = [clinical_metrics[t]['npv'] for t in thresholds]
    prevalence = [clinical_metrics[t]['prevalence'] for t in thresholds]
    
    # Create figure
    fig = go.Figure()
    
    # Add metric lines
    fig.add_trace(
        go.Scatter(
            x=[t*100 for t in thresholds],
            y=sensitivity,
            mode='lines+markers',
            name='Sensitivity',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[t*100 for t in thresholds],
            y=specificity,
            mode='lines+markers',
            name='Specificity',
            line=dict(color='red', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[t*100 for t in thresholds],
            y=ppv,
            mode='lines+markers',
            name='PPV (If predicted ≥ threshold, how likely true?)',
            line=dict(color='green', width=3)  # Highlight PPV with thicker line
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[t*100 for t in thresholds],
            y=npv,
            mode='lines+markers',
            name='NPV (If predicted < threshold, how likely true?)',
            line=dict(color='purple', width=2)
        )
    )
    
    # Add prevalence line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=[t*100 for t in thresholds],
            y=prevalence,
            mode='lines+markers',
            name='Prevalence (% of samples ≥ threshold)',
            line=dict(color='gray', width=2, dash='dot'),
            yaxis='y2'
        )
    )
    
    # Create table with key metrics for 1% threshold
    key_threshold = 0.01  # 1%
    if key_threshold in clinical_metrics:
        metrics = clinical_metrics[key_threshold]
        
        table_text = [
            ["Metric", "Value"],
            ["PPV at 1%", f"{metrics['ppv']:.2f}"],
            ["NPV at 1%", f"{metrics['npv']:.2f}"],
            ["Sensitivity at 1%", f"{metrics['sensitivity']:.2f}"],
            ["Specificity at 1%", f"{metrics['specificity']:.2f}"],
            ["Prevalence at 1%", f"{metrics['prevalence']:.2f}"]
        ]
        
        # Add table
        fig.add_trace(
            go.Table(
                domain=dict(x=[0.7, 1.0], y=[0.0, 0.3]),
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"],
                    line_color='darkslategray',
                    fill_color='lightgrey',
                    align='center',
                    font=dict(color='black', size=12)
                ),
                cells=dict(
                    values=list(zip(*table_text))[1:],
                    line_color='darkslategray',
                    fill_color='white',
                    align='left',
                    font=dict(color='black', size=11)
                )
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Clinical Decision Metrics by Concentration Threshold",
        xaxis=dict(
            title="Concentration Threshold (%)",
            type="log"
        ),
        yaxis=dict(
            title="Metric Value",
            range=[0, 1]
        ),
        yaxis2=dict(
            title="Prevalence",
            range=[0, 1],
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template='plotly_white',
        height=700,
        width=1000,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'clinical_decision_metrics.html'))
    fig.write_image(os.path.join(output_dir, 'clinical_decision_metrics.png'), scale=2)


def create_threshold_specific_analysis(df, specific_thresholds, output_dir):
    """
    Create detailed analysis for specific thresholds (0.1%, 0.5%, 1%, 2%, 5%, 10%)
    showing error distributions and confusion matrices
    
    Args:
        df: DataFrame with 'true_value' and 'predicted_value' columns
        specific_thresholds: List of specific concentration thresholds to analyze
        output_dir: Directory to save visualizations
    """
    for threshold in specific_thresholds:
        # Create a subdirectory for each threshold
        threshold_dir = os.path.join(output_dir, f"threshold_{threshold:.4f}")
        os.makedirs(threshold_dir, exist_ok=True)
        
        # Binary classification based on this threshold
        y_true_binary = (df['true_value'] >= threshold).astype(int)
        y_pred_binary = (df['predicted_value'] >= threshold).astype(int)
        
        # Calculate confusion matrix
        true_positive = ((df['true_value'] >= threshold) & (df['predicted_value'] >= threshold)).sum()
        false_positive = ((df['true_value'] < threshold) & (df['predicted_value'] >= threshold)).sum()
        true_negative = ((df['true_value'] < threshold) & (df['predicted_value'] < threshold)).sum()
        false_negative = ((df['true_value'] >= threshold) & (df['predicted_value'] < threshold)).sum()
        
        # Calculate metrics
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        ppv = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        npv = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0
        prevalence = (true_positive + false_negative) / len(df)
        accuracy = (true_positive + true_negative) / len(df)
        f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative) if (2 * true_positive + false_positive + false_negative) > 0 else 0
        
        # Create confusion matrix visualization
        fig_cm = go.Figure(data=go.Heatmap(
            z=[[true_negative, false_positive], 
               [false_negative, true_positive]],
            x=['Predicted < ' + str(threshold*100) + '%', 'Predicted ≥ ' + str(threshold*100) + '%'],
            y=['True < ' + str(threshold*100) + '%', 'True ≥ ' + str(threshold*100) + '%'],
            hoverongaps = False,
            colorscale='Blues',
            showscale=False,
            text=[[true_negative, false_positive], 
                  [false_negative, true_positive]],
            texttemplate="%{text}",
            textfont={"size":20}
        ))
        
        # Add title and annotations
        fig_cm.update_layout(
            title=f"Confusion Matrix for {threshold*100:.1f}% Threshold",
            xaxis_title="Predicted",
            yaxis_title="True",
            height=600,
            width=700
        )
        
        # Add metrics annotations
        annotation_text = (
            f"<b>Key Metrics:</b><br>"
            f"Sensitivity: {sensitivity:.3f}<br>"
            f"Specificity: {specificity:.3f}<br>"
            f"PPV: {ppv:.3f}<br>"
            f"NPV: {npv:.3f}<br>"
            f"Accuracy: {accuracy:.3f}<br>"
            f"F1 Score: {f1:.3f}<br>"
            f"Prevalence: {prevalence:.3f}"
        )
        
        fig_cm.add_annotation(
            x=1.2,
            y=0.5,
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=14),
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        # Save confusion matrix
        fig_cm.write_html(os.path.join(threshold_dir, 'confusion_matrix.html'))
        fig_cm.write_image(os.path.join(threshold_dir, 'confusion_matrix.png'), scale=2)
        
        # Create error distribution analysis for this threshold
        # Separate samples by classification result
        tp_df = df[(df['true_value'] >= threshold) & (df['predicted_value'] >= threshold)]
        fp_df = df[(df['true_value'] < threshold) & (df['predicted_value'] >= threshold)]
        tn_df = df[(df['true_value'] < threshold) & (df['predicted_value'] < threshold)]
        fn_df = df[(df['true_value'] >= threshold) & (df['predicted_value'] < threshold)]
        
        # Create figure for error distributions
        fig_err = make_subplots(rows=2, cols=2, 
                               subplot_titles=('True Positives Error Distribution', 
                                              'False Positives Error Distribution',
                                              'False Negatives Error Distribution',
                                              'True Negatives Error Distribution'))
        
        # Add TP error histogram
        if len(tp_df) > 0:
            fig_err.add_trace(
                go.Histogram(
                    x=tp_df['rel_error'],
                    nbinsx=20,
                    name='TP Rel. Error',
                    marker_color='green',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Add vertical lines at median and mean
            median_tp_err = tp_df['rel_error'].median()
            mean_tp_err = tp_df['rel_error'].mean()
            
            fig_err.add_vline(x=median_tp_err, line=dict(color="red", dash="dash"), 
                           annotation_text=f"Median: {median_tp_err:.1f}%", row=1, col=1)
            fig_err.add_vline(x=mean_tp_err, line=dict(color="blue", dash="dash"), 
                           annotation_text=f"Mean: {mean_tp_err:.1f}%", row=1, col=1)
        
        # Add FP error histogram
        if len(fp_df) > 0:
            # For false positives, we're showing how much they exceed the threshold
            fp_df['threshold_error'] = (fp_df['predicted_value'] - threshold) / threshold * 100
            
            fig_err.add_trace(
                go.Histogram(
                    x=fp_df['threshold_error'],
                    nbinsx=20,
                    name='FP Threshold Error',
                    marker_color='red',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # Add vertical lines at median and mean
            median_fp_err = fp_df['threshold_error'].median()
            mean_fp_err = fp_df['threshold_error'].mean()
            
            fig_err.add_vline(x=median_fp_err, line=dict(color="red", dash="dash"), 
                           annotation_text=f"Median: {median_fp_err:.1f}%", row=1, col=2)
            fig_err.add_vline(x=mean_fp_err, line=dict(color="blue", dash="dash"), 
                           annotation_text=f"Mean: {mean_fp_err:.1f}%", row=1, col=2)
        
        # Add FN error histogram
        if len(fn_df) > 0:
            # For false negatives, we're showing how far they are below the threshold
            fn_df['threshold_error'] = (fn_df['predicted_value'] - threshold) / threshold * 100
            
            fig_err.add_trace(
                go.Histogram(
                    x=fn_df['threshold_error'],
                    nbinsx=20,
                    name='FN Threshold Error',
                    marker_color='orange',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Add vertical lines at median and mean
            median_fn_err = fn_df['threshold_error'].median()
            mean_fn_err = fn_df['threshold_error'].mean()
            
            fig_err.add_vline(x=median_fn_err, line=dict(color="red", dash="dash"), 
                           annotation_text=f"Median: {median_fn_err:.1f}%", row=2, col=1)
            fig_err.add_vline(x=mean_fn_err, line=dict(color="blue", dash="dash"), 
                           annotation_text=f"Mean: {mean_fn_err:.1f}%", row=2, col=1)
        
        # Add TN error histogram
        if len(tn_df) > 0:
            fig_err.add_trace(
                go.Histogram(
                    x=tn_df['rel_error'],
                    nbinsx=20,
                    name='TN Rel. Error',
                    marker_color='blue',
                    opacity=0.7
                ),
                row=2, col=2
            )
            
            # Add vertical lines at median and mean
            median_tn_err = tn_df['rel_error'].median()
            mean_tn_err = tn_df['rel_error'].mean()
            
            fig_err.add_vline(x=median_tn_err, line=dict(color="red", dash="dash"), 
                           annotation_text=f"Median: {median_tn_err:.1f}%", row=2, col=2)
            fig_err.add_vline(x=mean_tn_err, line=dict(color="blue", dash="dash"), 
                           annotation_text=f"Mean: {mean_tn_err:.1f}%", row=2, col=2)
        
        # Update layout
        fig_err.update_layout(
            title=f"Error Distributions by Classification Category at {threshold*100:.1f}% Threshold",
            template='plotly_white',
            height=800,
            width=1000,
            showlegend=False
        )
        
        # Update axis titles
        fig_err.update_xaxes(title_text="Relative Error (%)", row=1, col=1, range=[-100, 100])
        fig_err.update_xaxes(title_text="Error Above Threshold (%)", row=1, col=2)
        fig_err.update_xaxes(title_text="Error Below Threshold (%)", row=2, col=1)
        fig_err.update_xaxes(title_text="Relative Error (%)", row=2, col=2, range=[-100, 100])
        
        # Update y-axis titles
        fig_err.update_yaxes(title_text="Count", row=1, col=1)
        fig_err.update_yaxes(title_text="Count", row=1, col=2)
        fig_err.update_yaxes(title_text="Count", row=2, col=1)
        fig_err.update_yaxes(title_text="Count", row=2, col=2)
        
        # Save error distribution figure
        fig_err.write_html(os.path.join(threshold_dir, 'error_distributions.html'))
        fig_err.write_image(os.path.join(threshold_dir, 'error_distributions.png'), scale=2)
        
        # Create a detailed scatter plot focused just on this threshold
        fig_scatter = go.Figure()
        
        # Add scatter plot with different colors based on classification
        fig_scatter.add_trace(
            go.Scatter(
                x=tp_df['true_value'],
                y=tp_df['predicted_value'],
                mode='markers',
                name='True Positive',
                marker=dict(color='green', size=8),
                opacity=0.7
            )
        )
        
        fig_scatter.add_trace(
            go.Scatter(
                x=fp_df['true_value'],
                y=fp_df['predicted_value'],
                mode='markers',
                name='False Positive',
                marker=dict(color='red', size=8),
                opacity=0.7
            )
        )
        
        fig_scatter.add_trace(
            go.Scatter(
                x=fn_df['true_value'],
                y=fn_df['predicted_value'],
                mode='markers',
                name='False Negative',
                marker=dict(color='orange', size=8),
                opacity=0.7
            )
        )
        
        fig_scatter.add_trace(
            go.Scatter(
                x=tn_df['true_value'],
                y=tn_df['predicted_value'],
                mode='markers',
                name='True Negative',
                marker=dict(color='blue', size=8),
                opacity=0.7
            )
        )
        
        # Add identity line
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, max(df['true_value'].max(), df['predicted_value'].max())],
                y=[0, max(df['true_value'].max(), df['predicted_value'].max())],
                mode='lines',
                name='Identity Line',
                line=dict(color='gray', dash='dash')
            )
        )
        
        # Add threshold lines
        fig_scatter.add_hline(y=threshold, line=dict(color='red', dash='dot'), 
                            annotation_text=f"Threshold = {threshold*100:.1f}%")
        fig_scatter.add_vline(x=threshold, line=dict(color='red', dash='dot'))
        
        # Update layout
        fig_scatter.update_layout(
            title=f"Prediction Analysis at {threshold*100:.1f}% Threshold",
            xaxis=dict(
                title="True Concentration (%)",
                type="log"
            ),
            yaxis=dict(
                title="Predicted Concentration (%)",
                type="log"
            ),
            template='plotly_white',
            height=800,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Save scatter plot
        fig_scatter.write_html(os.path.join(threshold_dir, 'threshold_scatter.html'))
        fig_scatter.write_image(os.path.join(threshold_dir, 'threshold_scatter.png'), scale=2)

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
    # Linear regression in log space
    log_space_results = calculate_log_space_metrics(y_true, y_pred)
    results['log_space'] = {
        'r_squared': log_space_results['log_r2'],
        'slope': log_space_results['log_slope'],
        'intercept': log_space_results['log_intercept']
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
        perfect_line = np.logspace(np.log10(max(min_val, 1e-6)), np.log10(max_val), 100)
        
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
        
        epsilon = 1e-6
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
# qrsh -b y -l h_vmem=2g -pe smp 32 -V -N train_oac -wd /users/zetzioni/sharedscratch/deepconv/src -o ~/sharedscratch/logs/train_oac.log 'cd /users/zetzioni/sharedscratch/deepconv/src && python -m deep_conv.detect.train --name oac_detector_with_pon_feature_dim16 --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell --data_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train_single_cell_clinical/OAC/ --atlas_path /users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed --target_cell_type OAC --target_cell_idx 9 --dropout_rate 0.15 --feature_dim 16 --control_data_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/ --calibrate --calibrate_every 15 --early_stopping 30 --epochs 200 --weight_decay 0.01 --clinical_eval --generate_clinical_report --visualise_clinical --calibrate_clinical_threshold'


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
            # Calibrate uncertainty estimates
            calibration_results = model.calibrate(val_loader, device)
            
            # Calibrate clinical threshold
            clinical_calibration = None
            if hasattr(args, 'calibrate_clinical_threshold') and args.calibrate_clinical_threshold:
                clinical_calibration = model.calibrate_clinical_threshold(
                    val_loader,
                    device,
                    target_metric='concentration_aware'
                )
            
            # Update best model state with calibration results
            if best_model_state is not None:
                best_model_state['calibration'] = calibration_results
                if clinical_calibration:
                    best_model_state['clinical_calibration'] = clinical_calibration
            
            # Save updated best model
            torch.save(best_model_state, os.path.join(args.output_dir, 'best_model_calibrated.pt'))
            
            logger.info(f"✓ Model calibrated:")
            logger.info(f"  Uncertainty calibration factor: {calibration_results['calibration_factor']:.4f}")
            
            if clinical_calibration:
                logger.info(f"  Clinical threshold: {clinical_calibration['threshold']:.6f}")
                logger.info(f"  Specificity: {clinical_calibration['specificity']:.2f}")
                logger.info(f"  Sensitivity: {clinical_calibration['sensitivity']:.2f}")
            
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
            mu, uncertainty, attention_weights, _ = model(marker_values, coverage)
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
            if args.control_data_dir:
                logger.info(f"Loading blank samples from {args.control_data_dir}...")
                try:
                    blank_samples = np.load(args.control_data_dir)
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
