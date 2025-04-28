import os
import sys
import argparse
import torch
import numpy as np
import json
import logging
import scipy.stats as stats
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from deep_conv.detect.preprocess import prepare_data_for_training, load_train_with_contrastive_data
from deep_conv.detect.model import EnhancedCancerDetectionModel, MarkerImportanceAnalyser


def parse_args():
    parser = argparse.ArgumentParser(description='Train enhanced cfDNA methylation cancer detection model')
    
    parser.add_argument('--name', type=str, default=None, help='Name for this training run (used for output directory)')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing parquet files')
    parser.add_argument('--atlas_path', type=str, required=True, help='Path to atlas file')
    parser.add_argument('--target_cell_type', type=str, required=True, help='Target cell type')
    parser.add_argument('--target_cell_idx', type=int, required=True, help='Target cell index in ground truth')
    parser.add_argument('--excluded_markers', type=str, default='', help='Comma-separated list of marker indices to exclude')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=96, help='Feature dimension')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularization')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    
    parser.add_argument('--cell_profile', type=str, default=None, 
                       choices=['default', 'high_snr', 'low_snr', 'ultra_low_snr'],
                       help='Predefined optimisation profile for different cell types')
    parser.add_argument('--detection_loss_weight', type=float, default=0.2, 
                       help='Weight of detection loss relative to concentration loss')
    parser.add_argument('--focal_weight_factor', type=float, default=50,
                       help='Factor for focal weighting of low concentration samples')
    parser.add_argument('--low_concentration_threshold', type=float, default=0.01,
                       help='Threshold defining low concentration samples for special handling')
    parser.add_argument('--l2_weight', type=float, default=0.02,
                       help='Weight for L2 regularization in loss calculation')
    parser.add_argument('--marker_specific_bg', action='store_true',
                       help='Use marker-specific background correction')
    parser.add_argument('--min_reliable_coverage', type=float, default=5.0,
                       help='Minimum coverage to consider a marker reliable')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimiser')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default="./saved_models", help='Output directory')
 
    # Evaluation parameters
    parser.add_argument('--detection_thresholds', type=str, default="0.001,0.005,0.01,0.05", 
                       help='Comma-separated detection thresholds')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    parser.add_argument('--save_interval', type=int, default=50, help='Save checkpoint every N epochs')
    
    parser.add_argument('--control_data_dir', type=str, default=None, 
                   help='Directory containing control data for contrastive learning')
    parser.add_argument('--calibrate', action='store_true', 
                    help='Calibrate confidence intervals and background correction')

    args = parser.parse_args()
    
    if args.name:
        dir_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"enhanced_model_{timestamp}"
    
    args.output_dir = os.path.join(args.output_dir, dir_name)

    # Process detection thresholds
    args.detection_thresholds = [float(x) for x in args.detection_thresholds.split(',')]
    
    if args.cell_profile:
        apply_cell_profile(args)

    return args


def apply_cell_profile(args):
    """Apply predefined parameter sets optimised for different cell types"""
    profiles = {
        'default': {
            'detection_loss_weight': 0.2,
            'focal_weight_factor': 50,
            'low_concentration_threshold': 0.01
        },
        'high_snr': {  # For cells like OAC with good SNR
            'detection_loss_weight': 0.2,
            'focal_weight_factor': 50,  
            'low_concentration_threshold': 0.01
        },
        'low_snr': {  # For cells with moderate SNR issues
            'detection_loss_weight': 0.4,
            'focal_weight_factor': 100,
            'low_concentration_threshold': 0.02
        },
        'ultra_low_snr': {  # For T-cells and other very low SNR cases
            'detection_loss_weight': 0.5,
            'focal_weight_factor': 120,
            'low_concentration_threshold': 0.03
        }
    }
    profile = profiles[args.cell_profile]
    
    # Only override if not explicitly provided in command line
    if args.detection_loss_weight is None:
        args.detection_loss_weight = profile['detection_loss_weight']
    if args.focal_weight_factor is None:
        args.focal_weight_factor = profile['focal_weight_factor']
    if args.low_concentration_threshold is None:
        args.low_concentration_threshold = profile['low_concentration_threshold']


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
    logger = logging.getLogger('cancer_detection')
    logger.setLevel(logging.INFO)
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def calculate_loss(model, mu, uncertainty, detection_probs, y_true, args, control_mask=None):
    """Calculate combined loss with concentration and detection components"""
    # Get concentration loss
    concentration_loss = model.compute_loss(mu, uncertainty, y_true, control_mask)
    
    # Calculate detection losses
    detection_losses = []
    for i, threshold in enumerate(args.detection_thresholds):
        binary_y = (y_true >= threshold).float()
        det_loss = F.binary_cross_entropy(detection_probs[i], binary_y)
        detection_losses.append(det_loss)
    
    # Use configurable detection loss weight
    detection_loss_weight = args.detection_loss_weight
    combined_detection_loss = sum(detection_losses) / len(detection_losses)
    
    # Calculate total loss
    total_loss = concentration_loss + detection_loss_weight * combined_detection_loss
    
    return total_loss, concentration_loss, combined_detection_loss

def train_model(model, train_loader, val_loader, control_loader, args, device):
    """
    Train the model with improved concentration-focused approach while retaining original architecture
    
    Args:
        model: The cancer detection model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        control_loader: DataLoader for control samples (can be None)
        args: Training arguments
        device: Device to run training on
        
    Returns:
        Trained model and best model state
    """
    logger = logging.getLogger('cancer_detection')
    
    git_info = get_git_info()
    git_commit = git_info['commit']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'concentration_loss': [],
        'detection_loss': [],
        'r2_score': [],
        'mean_absolute_error': [],
        'concentration_metrics': [],
        'clinical_metrics': [],
        'lr': []
    }
    
    # Start training loop
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        concentration_loss = 0
        detection_loss = 0
        
        # Progress bar for training
        train_bar = tqdm(enumerate(train_loader), 
                         desc=f"Epoch {epoch+1}/{args.epochs} [Train]", 
                         total=len(train_loader))
        
        for i, batch_data in train_bar:
            # Handle both dataset types (with or without control_mask)
            if len(batch_data) == 4:  # Dataset includes control_mask
                marker_values, coverage, y_true, control_mask = batch_data
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                control_mask = control_mask.to(device)
                
                # Get model predictions
                with autocast():
                    mu, uncertainty, detection_probs, _ = model(marker_values, coverage)
                    
                    # Calculate combined loss
                    loss, conc_loss, det_loss = calculate_loss(
                        model, mu, uncertainty, detection_probs, y_true, args, control_mask
                    )
                    # Scale for gradient accumulation
                    loss = loss / args.grad_accum_steps
                    
            else:  # Standard dataset without control_mask
                marker_values, coverage, y_true = batch_data
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                with autocast():
                    # Get model predictions
                    mu, uncertainty, detection_probs, _ = model(marker_values, coverage)
                    
                    # Calculate combined loss
                    loss, conc_loss, det_loss = calculate_loss(
                        model, mu, uncertainty, detection_probs, y_true, args
                    )
                    # Scale for gradient accumulation
                    loss = loss / args.grad_accum_steps
            
            # Backpropagation with mixed precision
            scaler.scale(loss).backward()
            
            # Update metrics
            train_loss += loss.item() * args.grad_accum_steps
            concentration_loss += conc_loss.item()
            detection_loss += det_loss.item()
            
            # Gradient accumulation and optimizer step
            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step with mixed precision
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # Update progress bar
            train_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        concentration_loss /= len(train_loader)
        detection_loss /= len(train_loader)
        
        # Validation phase with both standard and concentration-focused metrics
        val_loss, val_metrics = validate_model(model, val_loader, device, args)
        conc_metrics = compute_concentration_metrics(model, val_loader, device)
        
        # Periodic calibration
        if args.calibrate and (epoch % 5 == 0 or epoch == args.epochs - 1):
            logger.info(f"Calibrating model...")
            calibration_results = calibrate_model(model, val_loader, control_loader, device)
            logger.info(f"  Calibration factor: {calibration_results['calibration_factor']:.4f}")
            if 'global_bg_level' in calibration_results:
                logger.info(f"  Background level: {calibration_results['global_bg_level']:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['concentration_loss'].append(concentration_loss)
        history['detection_loss'].append(detection_loss)
        history['r2_score'].append(val_metrics['r2'])
        history['mean_absolute_error'].append(val_metrics['mae'])
        history['clinical_metrics'].append(val_metrics['clinical_metrics'])
        history['concentration_metrics'].append(conc_metrics)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # Log validation results
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.6f}, "
                   f"Val Loss: {val_loss:.6f}, "
                   f"R²: {val_metrics['r2']:.4f}, "
                   f"MAE: {val_metrics['mae']:.6f}")
        
        # Log clinical metrics
        if 0.01 in val_metrics['clinical_metrics']:
            metrics_1pct = val_metrics['clinical_metrics'][0.01]
            logger.info(f"  At 1% threshold - "
                       f"Sensitivity: {metrics_1pct['sensitivity']:.4f}, "
                       f"Specificity: {metrics_1pct['specificity']:.4f}")
        
        # Log concentration metrics for key ranges
        for range_name in ['0.1-1%', '1-5%']:
            if range_name in conc_metrics['stratified_metrics']:
                range_metrics = conc_metrics['stratified_metrics'][range_name]
                logger.info(f"  {range_name} (n={range_metrics['count']}): "
                           f"MAE={range_metrics['mae']:.6f}, "
                           f"Within 25%={range_metrics.get('within_25pct', 0):.1f}%")
        
        # Check for improvement
        if val_loss < best_val_loss:
            improvement = "inf" if best_val_loss == float('inf') else f"{(best_val_loss - val_loss) / best_val_loss * 100:.2f}%"
            best_val_loss = val_loss
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'concentration_metrics': conc_metrics,
                'args': vars(args),
                'git_commit': git_commit,
            }
            
            # Save best model
            torch.save(best_model_state, os.path.join(args.output_dir, 'best_model.pt'))
            logger.info(f"✓ New best model saved! Improvement: {improvement}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"× No improvement. Patience: {patience_counter}/{args.early_stopping}")
        
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
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'args': vars(args),
                'history': history,
                'git_comit': git_commit,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'val_metrics': val_metrics,
        'concentration_metrics': conc_metrics,
        'args': vars(args),
        'history': history,
        'git_commit': git_commit,
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert history values to native types for JSON serialization
        serializable_history = {}
        for key, values in history.items():
            if key not in ['clinical_metrics', 'concentration_metrics']:
                serializable_history[key] = [float(v) for v in values]
            elif key == 'clinical_metrics':
                # Handle nested clinical metrics
                serializable_metrics = []
                for epoch_metrics in values:
                    serializable_epoch = {}
                    for threshold, metrics in epoch_metrics.items():
                        serializable_epoch[str(threshold)] = {k: float(v) for k, v in metrics.items() if v is not None}
                    serializable_metrics.append(serializable_epoch)
                serializable_history[key] = serializable_metrics
            elif key == 'concentration_metrics':
                # Handle concentration metrics
                serializable_metrics = []
                for epoch_metrics in values:
                    serializable_epoch = {}
                    for range_name, metrics in epoch_metrics.get('stratified_metrics', {}).items():
                        serializable_epoch[range_name] = {k: float(v) if v is not None else None for k, v in metrics.items()}
                    serializable_metrics.append(serializable_epoch)
                serializable_history[key] = serializable_metrics
        
        json.dump(serializable_history, f, indent=2)
    
    # Create plots of training history
    plot_training_history(history, args.output_dir)
    
    # Load best model for return
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model'])
    
    return model, best_model_state


def compute_concentration_metrics(model, data_loader, device):
    """
    Compute concentration-focused metrics without changing the validation flow
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary of concentration-focused metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            # Handle both dataset types
            if len(batch_data) == 4:
                marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask
            else:
                marker_values, coverage, y_true = batch_data
            
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            
            # Forward pass
            mu, _, _, _ = model(marker_values, coverage)
            
            # Store predictions and targets
            all_preds.append(mu.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate concentration-aware metrics
    concentration_metrics = compute_concentration_aware_metrics(all_preds, all_targets)
    
    return concentration_metrics


def compute_concentration_aware_metrics(predictions, targets):
    """
    Compute concentration-aware metrics that evaluate how well the model estimates
    across different concentration ranges.
    
    Args:
        predictions: Predicted concentrations (numpy array)
        targets: Ground truth concentrations (numpy array)
        
    Returns:
        Dictionary of various concentration-aware metrics
    """
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # Ensure arrays are flattened
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Define concentration ranges
    ranges = [
        (0, 0.001, "0-0.1%"),
        (0.001, 0.01, "0.1-1%"),
        (0.01, 0.05, "1-5%"),
        (0.05, 0.1, "5-10%"),
        (0.1, 1.0, ">10%")
    ]
    
    # Initialize results dict
    results = {
        'stratified_metrics': {},
        'band_accuracy': {}
    }
    
    # Calculate stratified metrics for each concentration range
    for low, high, name in ranges:
        mask = (targets >= low) & (targets < high)
        range_predictions = predictions[mask]
        range_targets = targets[mask]
        
        if len(range_targets) > 0:
            # Calculate range-specific metrics
            range_mae = mean_absolute_error(range_targets, range_predictions)
            
            # Calculate percentage of predictions within percentage bands of true value
            within_10pct = 0
            within_25pct = 0
            within_50pct = 0
            
            if low > 0:  # Only for non-zero ranges
                # Calculate relative errors
                rel_errors = np.abs(range_predictions - range_targets) / np.maximum(range_targets, 1e-6)
                within_10pct = np.mean(rel_errors <= 0.1) * 100
                within_25pct = np.mean(rel_errors <= 0.25) * 100
                within_50pct = np.mean(rel_errors <= 0.5) * 100
            
            results['stratified_metrics'][name] = {
                'count': int(np.sum(mask)),
                'mae': float(range_mae),
                'within_10pct': float(within_10pct),
                'within_25pct': float(within_25pct),
                'within_50pct': float(within_50pct)
            }
    
    # Calculate concentration band accuracy
    # This measures how well the model classifies samples into concentration bands
    bands = [(0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.0)]
    band_names = ["0-0.1%", "0.1-1%", "1-10%", ">10%"]
    
    for i, (low, high) in enumerate(bands):
        mask = (targets >= low) & (targets < high)
        band_preds = predictions[mask]
        correct_band = np.sum((band_preds >= low) & (band_preds < high))
        total_in_band = np.sum(mask)
        
        if total_in_band > 0:
            accuracy = float(correct_band) / float(total_in_band) * 100
            results['band_accuracy'][band_names[i]] = {
                'accuracy': accuracy,
                'count': int(total_in_band)
            }
    
    # Calculate ordering accuracy
    # This measures how well the model preserves the relative ordering of samples
    # For example, if sample A has higher concentration than sample B, does the model predict this correctly?
    n_samples = len(targets)
    n_correct_orders = 0
    n_total_comparisons = 0
    
    # Limit to a manageable number of comparisons for large datasets
    max_comparisons = 100000
    if n_samples > 1000:
        # Sample random pairs for large datasets
        import random
        indices = list(range(n_samples))
        random.shuffle(indices)
        pairs = []
        for _ in range(min(max_comparisons, n_samples * (n_samples - 1) // 2)):
            i, j = random.sample(indices, 2)
            if targets[i] != targets[j]:  # Only compare different concentration samples
                pairs.append((i, j))
    else:
        # Use all pairs for smaller datasets
        pairs = [(i, j) for i in range(n_samples) for j in range(i+1, n_samples) 
                if targets[i] != targets[j]]
    
    for i, j in pairs:
        n_total_comparisons += 1
        
        # Check if ordering is preserved
        if (targets[i] < targets[j] and predictions[i] < predictions[j]) or \
           (targets[i] > targets[j] and predictions[i] > predictions[j]):
            n_correct_orders += 1
    
    if n_total_comparisons > 0:
        ordering_accuracy = float(n_correct_orders) / float(n_total_comparisons) * 100
        results['ordering_accuracy'] = {
            'accuracy': ordering_accuracy,
            'comparisons': n_total_comparisons
        }
    
    return results


def validate_model(model, val_loader, device, args):
    """
    Validate model performance on validation set
    
    Args:
        model: The cancer detection model to validate
        val_loader: DataLoader for validation data
        device: Device to run validation on
        args: Training arguments with detection thresholds
        
    Returns:
        Tuple of (validation loss, metrics dictionary)
    """
    model.eval()
    val_loss = 0
    concentration_loss = 0
    detection_loss = 0
    calibration_error = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            # Handle both dataset types
            if len(batch_data) == 4:
                marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask for validation
            else:
                marker_values, coverage, y_true = batch_data
            
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            
            # Forward pass
            mu, uncertainty, detection_probs, _ = model(marker_values, coverage)
            
            # Compute concentration loss
            batch_conc_loss = model.compute_loss(mu, uncertainty, y_true)
            concentration_loss += batch_conc_loss.item()
            
            # Compute detection loss
            batch_det_loss = 0
            for i, threshold in enumerate(args.detection_thresholds):
                binary_y = (y_true >= threshold).float()
                det_loss = F.binary_cross_entropy(detection_probs[i], binary_y)
                batch_det_loss += det_loss.item()
            
            batch_det_loss /= len(args.detection_thresholds)
            detection_loss += batch_det_loss
            
            # Compute total loss
            batch_loss = batch_conc_loss + args.detection_loss_weight * batch_det_loss
            val_loss += batch_loss.item()
            
            # Calculate calibration error
            estimate, ci, _ = model.get_estimate_and_ci(mu, uncertainty)
            in_ci = (y_true >= ci[:, 0:1]) & (y_true <= ci[:, 1:2])
            calibration_error += (1.0 - in_ci.float().mean()).item()
            
            # Store predictions and targets for metrics
            all_preds.append(mu.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
    
    # Calculate average losses
    val_loss /= len(val_loader)
    concentration_loss /= len(val_loader)
    detection_loss /= len(val_loader)
    calibration_error /= len(val_loader)
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate regression metrics
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Calculate clinical metrics
    clinical_metrics = clinical_performance_metrics(all_preds, all_targets, args.detection_thresholds)
    
    # Calculate concentration metrics
    concentration_metrics = compute_concentration_aware_metrics(all_preds, all_targets)
    
    # Return validation loss and metrics
    metrics = {
        'concentration_loss': concentration_loss,
        'detection_loss': detection_loss,
        'calibration_error': calibration_error,
        'r2': r2,
        'mae': mae,
        'clinical_metrics': clinical_metrics,
        'concentration_metrics': concentration_metrics
    }
    
    return val_loss, metrics

def clinical_performance_metrics(predictions, ground_truth, thresholds=[0.001, 0.01, 0.05]):
    """
    Calculate clinically relevant metrics for cancer detection
    
    Args:
        predictions: Predicted concentrations (numpy array)
        ground_truth: True concentrations (numpy array)
        thresholds: List of concentration thresholds to evaluate
        
    Returns:
        Dictionary of clinical metrics at each threshold
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    
    # Ensure arrays are flattened
    predictions = predictions.flatten()
    ground_truth = ground_truth.flatten()
    
    # Calculate metrics for each threshold
    results = {}
    
    for threshold in thresholds:
        # Convert to binary classification
        y_pred = (predictions >= threshold).astype(float)
        y_true = (ground_truth >= threshold).astype(float)
        
        # Calculate basic metrics
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        
        # Calculate rates
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        
        # Calculate ROC curve and AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, predictions)
            roc_auc = auc(fpr, tpr)
            
            # Find sensitivity at 95% specificity (5% FPR)
            idx_95spec = np.argmin(np.abs(fpr - 0.05))
            sens_at_95spec = tpr[idx_95spec] if idx_95spec < len(tpr) else 0
            
            # Calculate PR curve and average precision
            precision, recall, _ = precision_recall_curve(y_true, predictions)
            avg_precision = average_precision_score(y_true, predictions)
        except:
            # Handle cases with only one class
            roc_auc = 0
            sens_at_95spec = 0
            avg_precision = 0
        
        # Calculate magnitude-aware metrics
        # How close are the predictions to the true values?
        if np.sum(y_true) > 0:
            positive_samples = predictions[y_true == 1]
            positive_targets = ground_truth[y_true == 1]
            
            # Mean absolute percentage error for positive samples
            mape = np.mean(np.abs(positive_samples - positive_targets) / np.maximum(positive_targets, 1e-6)) \
                   if len(positive_samples) > 0 else np.nan
                   
            # Percentage of positive samples with error < 20%
            within_20pct = np.mean(np.abs(positive_samples - positive_targets) <= 0.2 * np.maximum(positive_targets, 1e-6)) * 100 \
                           if len(positive_samples) > 0 else np.nan
        else:
            mape = np.nan
            within_20pct = np.nan
        
        # Store all metrics
        results[threshold] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'roc_auc': float(roc_auc),
            'sens_at_95spec': float(sens_at_95spec),
            'avg_precision': float(avg_precision),
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN),
            'mape': float(mape) if not np.isnan(mape) else None,
            'within_20pct': float(within_20pct) if not np.isnan(within_20pct) else None
        }
    
    return results

def plot_training_history(history, output_dir):
    """
    Create enhanced plots of training history metrics using Plotly, including clinical metrics
    like sensitivity, specificity, and false positive rate (FPR) at the 1% threshold.
    
    Args:
        history: Dictionary containing training history metrics
        output_dir: Directory to save the plots
    """
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
        rows=3, cols=2,
        subplot_titles=(
            'Training & Validation Loss', 
            'Calibration Error', 
            'R² Score', 
            'Mean Absolute Error', 
            'Learning Rate', 
            'R² vs Validation Loss'
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
    
    # 2. Calibration error
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['calibration_error'], 
            mode='lines+markers', 
            name='Calibration Error',
            line=dict(color='green', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. R² score
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['r2_score'], 
            mode='lines+markers', 
            name='R² Score',
            line=dict(color='purple', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Mean absolute error
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['mean_absolute_error'], 
            mode='lines+markers', 
            name='MAE',
            line=dict(color='orange', width=2),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 5. Learning rate
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
            row=3, col=1
        )
        
        # Set log scale for learning rate
        fig.update_yaxes(type='log', row=3, col=1)
    
    # 6. R² vs Validation Loss
    fig.add_trace(
        go.Scatter(
            x=history['val_loss'],
            y=history['r2_score'],
            mode='markers',
            marker=dict(
                size=8,
                color=epochs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Epoch')
            ),
            showlegend=False
        ),
        row=3, col=2
    )
    
    # Label selected epochs on the scatter plot
    for i, epoch in enumerate(epochs):
        if i % 5 == 0 or i == len(epochs) - 1:  # Label every 5th epoch and the last one
            fig.add_annotation(
                x=history['val_loss'][i],
                y=history['r2_score'][i],
                text=str(epoch),
                showarrow=False,
                font=dict(size=8),
                row=3, col=2
            )
    
    # Update layout for main figure
    fig.update_layout(
        height=1000,
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
    fig.update_xaxes(title_text='Epoch', row=3, col=1)
    fig.update_xaxes(title_text='Validation Loss', row=3, col=2)
    
    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_yaxes(title_text='Error', row=1, col=2)
    fig.update_yaxes(title_text='R²', row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text='MAE', row=2, col=2)
    fig.update_yaxes(title_text='Learning Rate', row=3, col=1)
    fig.update_yaxes(title_text='R² Score', row=3, col=2, range=[0, 1])
    
    # Save main figure
    fig.write_html(os.path.join(plots_dir, 'training_history.html'))
    fig.write_image(os.path.join(plots_dir, 'training_history.png'), scale=2)
    
    # Create clinical metrics plot if available
    if 'clinical_metrics' in history and len(history['clinical_metrics']) > 0:
        # Extract metrics for 1% threshold (typically most relevant)
        sensitivity_1pct = []
        specificity_1pct = []
        fpr_1pct = []
        
        for epoch_metrics in history['clinical_metrics']:
            if 0.01 in epoch_metrics or '0.01' in epoch_metrics:
                metrics = epoch_metrics.get(0.01, epoch_metrics.get('0.01', {}))
                sensitivity_1pct.append(metrics.get('sensitivity', None))
                specificity_1pct.append(metrics.get('specificity', None))
                # Calculate FPR = FP / (FP + TN)
                fp = metrics.get('FP', 0)
                tn = metrics.get('TN', 0)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fpr_1pct.append(fpr)
            else:
                sensitivity_1pct.append(None)
                specificity_1pct.append(None)
                fpr_1pct.append(None)
        
        # Filter out None values and create valid data for plotting
        valid_epochs = []
        valid_sens = []
        valid_spec = []
        valid_fpr = []
        
        for i, (sens, spec, fpr) in enumerate(zip(sensitivity_1pct, specificity_1pct, fpr_1pct)):
            if sens is not None and spec is not None and fpr is not None:
                valid_epochs.append(epochs[i])
                valid_sens.append(sens)
                valid_spec.append(spec)
                valid_fpr.append(fpr)
        
        # Create clinical metrics figure if there is valid data
        if valid_epochs:
            clinical_fig = go.Figure()
            
            # Sensitivity at 1% threshold
            clinical_fig.add_trace(
                go.Scatter(
                    x=valid_epochs,
                    y=valid_sens,
                    mode='lines+markers',
                    name='Sensitivity (1% threshold)',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Specificity at 1% threshold
            clinical_fig.add_trace(
                go.Scatter(
                    x=valid_epochs,
                    y=valid_spec,
                    mode='lines+markers',
                    name='Specificity (1% threshold)',
                    line=dict(color='red', width=2)
                )
            )
            
            # False Positive Rate (FPR) at 1% threshold
            clinical_fig.add_trace(
                go.Scatter(
                    x=valid_epochs,
                    y=valid_fpr,
                    mode='lines+markers',
                    name='FPR (1% threshold)',
                    line=dict(color='orange', width=2)
                )
            )
            
            # Update layout
            clinical_fig.update_layout(
                title='Clinical Metrics at 1% Threshold',
                xaxis_title='Epoch',
                yaxis_title='Value',
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                width=900,
                height=500,
                yaxis=dict(range=[0, 1])
            )
            
            # Add reference line at 0.95 for sensitivity and specificity
            clinical_fig.add_shape(
                type="line",
                x0=min(valid_epochs),
                y0=0.95,
                x1=max(valid_epochs),
                y1=0.95,
                line=dict(color="green", dash="dash"),
                name="95% Reference"
            )
            
            # Add reference line at 0.05 for FPR (target for 95% specificity)
            clinical_fig.add_shape(
                type="line",
                x0=min(valid_epochs),
                y0=0.05,
                x1=max(valid_epochs),
                y1=0.05,
                line=dict(color="purple", dash="dash"),
                name="5% FPR Target"
            )
            
            # Save clinical metrics figure
            clinical_fig.write_html(os.path.join(plots_dir, 'clinical_metrics.html'))
            clinical_fig.write_image(os.path.join(plots_dir, 'clinical_metrics.png'), scale=2)
    
    # Create individual plots for better detail
    metrics = [
        ('loss', ['train_loss', 'val_loss'], ['Train Loss', 'Validation Loss'], ['blue', 'red']),
        ('r2_score', ['r2_score'], ['R² Score'], ['purple']),
        ('mae', ['mean_absolute_error'], ['Mean Absolute Error'], ['orange']),
        ('calibration', ['calibration_error'], ['Calibration Error'], ['green'])
    ]
    
    for name, keys, labels, colors in metrics:
        detail_fig = go.Figure()
        
        for key, label, color in zip(keys, labels, colors):
            detail_fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history[key],
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=2)
                )
            )
        
        # Add reference lines for R² and calibration error
        if name == 'r2_score':
            detail_fig.add_shape(
                type="line",
                x0=min(epochs),
                y0=0.9,
                x1=max(epochs),
                y1=0.9,
                line=dict(color="green", dash="dash"),
                name="0.9 R² Reference"
            )
        elif name == 'calibration':
            detail_fig.add_shape(
                type="line",
                x0=min(epochs),
                y0=0.05,
                x1=max(epochs),
                y1=0.05,
                line=dict(color="purple", dash="dash"),
                name="5% Calibration Error Target"
            )
        
        detail_fig.update_layout(
            title=f'{labels[0]}' if len(labels) == 1 else 'Loss Curves',
            xaxis_title='Epoch',
            yaxis_title=name.replace('_', ' ').title(),
            template='plotly_white',
            width=900,
            height=600,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Set y-axis range for R² and calibration error plots
        if name == 'r2_score':
            detail_fig.update_yaxes(range=[0, 1])
        elif name == 'calibration':
            detail_fig.update_yaxes(range=[0, max(history[key]) * 1.1])
        
        # Save individual figure
        detail_fig.write_html(os.path.join(plots_dir, f'{name}_history.html'))
        detail_fig.write_image(os.path.join(plots_dir, f'{name}_history.png'), scale=2)

def visualise_results(predictions, ground_truth, output_subdir, ci_data=None, marker_importance=None, prefix=""):
    """
    Unified visualisation function for both validation and test results
    
    Args:
        predictions: Array of predicted cell type concentrations
        ground_truth: Array of true cell type concentrations
        output_subdir: Directory to save visualisations (relative to output_dir)
        ci_data: Optional tuple of (lower_ci, upper_ci) for confidence interval visualisation
        marker_importance: Optional marker importance data
        prefix: Optional prefix for output files
    """
    logger = logging.getLogger('cancer_detection')
    
    # Create visualisation directory
    os.makedirs(output_subdir, exist_ok=True)
    
    # Try to import visualisation module
    try:
        from deep_conv.detect.visualise import create_visualisations
        
        logger.info(f"Creating visualisations for {prefix}data...")
        
        # Generate visualisations
        metrics = create_visualisations(
            predictions=predictions.flatten(), 
            ground_truth=ground_truth.flatten(),
            output_dir=output_subdir
        )
        
        # Log key metrics
        logger.info(f"{prefix}R² Score: {metrics['r2']:.4f}")
        logger.info(f"{prefix}MAE: {metrics['mae']:.6f}")
        logger.info(f"{prefix}% Within 10% error: {metrics['within_10pct']:.2f}%")
        
        # If confidence interval data is provided, create additional plots
        if ci_data is not None:
            lower_ci, upper_ci = ci_data
            # Additional CI plots would be created here using plot_predictions function
            plot_predictions(predictions, ground_truth, lower_ci, upper_ci, output_subdir)
            
            # Calculate percentage of targets within CI
            in_ci = ((ground_truth >= lower_ci) & (ground_truth <= upper_ci)).mean()
            ci_width = (upper_ci - lower_ci).mean()
            logger.info(f"{prefix}Targets within CI: {in_ci * 100:.2f}%")
            logger.info(f"{prefix}Average CI Width: {ci_width:.6f}")
        
        # If marker importance data is provided, visualise it
        if marker_importance is not None:
            # Plot marker importance
            plot_marker_importance(marker_importance, output_subdir)
        
        logger.info(f"Visualisations saved to {output_subdir}")
        return metrics
        
    except ImportError:
        logger.warning("Plotly visualisation module not found. Skipping advanced visualisations.")
        return None
    except Exception as e:
        logger.error(f"Error creating visualisations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def plot_predictions(predictions, targets, lower_ci, upper_ci, output_dir):
    """
    Plot predictions vs targets with confidence intervals using Plotly
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Sort by targets for clearer visualisation
    sorted_indices = np.argsort(targets.flatten())
    sorted_targets = targets.flatten()[sorted_indices]
    sorted_preds = predictions.flatten()[sorted_indices]
    sorted_lower = lower_ci.flatten()[sorted_indices]
    sorted_upper = upper_ci.flatten()[sorted_indices]
    
    # 1. Plot predictions with CI (sorted by true value)
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sorted_targets)),
            y=sorted_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sorted_targets)),
            y=sorted_lower,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0, 176, 246, 0.2)',
            fill='tonexty',
            showlegend=True,
            name='95% CI'
        )
    )
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sorted_targets)),
            y=sorted_preds,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Predictions'
        )
    )
    
    # Add targets
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sorted_targets)),
            y=sorted_targets,
            mode='lines',
            line=dict(color='red', width=2),
            name='True Values'
        )
    )
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    # Update layout
    fig.update_layout(
        title=f'Predictions vs Targets (sorted) (R² = {r2:.4f}, MAE = {mae:.4f})',
        xaxis_title='Sample Index (sorted by true value)',
        yaxis_title='Cell type Concentration',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        width=900,
        height=600
    )
    
    # Save figure
    fig.write_html(os.path.join(plots_dir, 'predictions_vs_targets.html'))
    fig.write_image(os.path.join(plots_dir, 'predictions_vs_targets.png'), scale=2)

def plot_marker_importance(marker_importance, output_dir):
    """
    Visualise marker importance
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Sort markers by importance
    sorted_indices = np.argsort(marker_importance)[::-1]  # Descending order
    sorted_importance = marker_importance[sorted_indices]
    
    # Get top 20 markers
    top_n = min(20, len(sorted_indices))
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            y=[f"Marker {i+1}" for i in sorted_indices[:top_n]],
            x=sorted_importance[:top_n],
            orientation='h',
            marker_color='blue',
            opacity=0.7
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Top Markers by Importance',
        xaxis_title='Importance Score',
        yaxis_title='Marker ID',
        template='plotly_white',
        width=900,
        height=600,
        yaxis=dict(autorange="reversed")  # Descending order
    )
    
    # Save figure
    fig.write_html(os.path.join(plots_dir, 'marker_importance.html'))
    fig.write_image(os.path.join(plots_dir, 'marker_importance.png'), scale=2)

def evaluate(model, data_loader, args, device, split_name="test"):
    """Evaluation function for test sets with dynamic background support"""
    logger = logging.getLogger('cancer_detection')
    logger.info(f"Starting model evaluation on {split_name} set...")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_lower_ci = []
    all_upper_ci = []
    all_marker_attentions = []
    all_detection_probs = []
    all_bg_levels = []
    
    # Create progress bar for evaluation
    eval_bar = tqdm(data_loader, desc=f"Evaluating {split_name} set", position=0)
    
    with torch.no_grad():
        for batch_data in eval_bar:
            # Updated: Handle 4-element return from dataset
            if len(batch_data) == 4:
                marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask
            else:
                marker_values, coverage, y_true = batch_data
                
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            mu, uncertainty, det_probs, attention_weights = model(marker_values, coverage)
            estimate, ci, _ = model.get_estimate_and_ci(mu, uncertainty)
        
            # Get background levels
            bg_levels = model.get_background_levels(data_loader, device)
            
            all_preds.append(estimate.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
            all_lower_ci.append(ci[:, 0:1].cpu().numpy())
            all_upper_ci.append(ci[:, 1:2].cpu().numpy())
            all_marker_attentions.append(attention_weights.cpu().numpy())
            all_detection_probs.append([dp.cpu().numpy() for dp in det_probs])
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_lower_ci = np.concatenate(all_lower_ci)
    all_upper_ci = np.concatenate(all_upper_ci)
    all_marker_attentions = np.concatenate(all_marker_attentions)
    all_detection_probs = [np.concatenate([batch[i] for batch in all_detection_probs]) for i in range(len(args.detection_thresholds))]
    
    # Calculate metrics
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Calculate percentage of targets within CI
    in_ci = ((all_targets >= all_lower_ci) & (all_targets <= all_upper_ci)).mean()
    
    # Calculate average CI width
    ci_width = (all_upper_ci - all_lower_ci).mean()
    
    # Calculate concentration metrics
    concentration_metrics = compute_concentration_aware_metrics(all_preds, all_targets)
    
    # Log basic results
    logger.info(f"\n{split_name.upper()} RESULTS:")
    logger.info(f"Number of {split_name} samples: {len(all_targets)}")
    logger.info(f"R² Score: {r2:.6f}")
    logger.info(f"Mean Absolute Error: {mae:.6f}")
    logger.info(f"Targets within CI: {in_ci * 100:.2f}%")
    logger.info(f"Average CI Width: {ci_width:.6f}")
    
    # Log concentration metrics
    logger.info(f"\nCONCENTRATION METRICS:")
    for range_name, metrics in concentration_metrics['stratified_metrics'].items():
        logger.info(f"  {range_name} (n={metrics['count']}): "
                   f"MAE={metrics['mae']:.6f}, "
                   f"Within 25%={metrics.get('within_25pct', 0):.1f}%")
    
    # Log band accuracy
    if 'band_accuracy' in concentration_metrics:
        logger.info("\nCONCENTRATION BAND ACCURACY:")
        for band, acc in concentration_metrics['band_accuracy'].items():
            logger.info(f"  {band}: {acc['accuracy']:.2f}% (n={acc['count']})")
    
    # Log detection metrics
    analyser = MarkerImportanceAnalyser(model)
    detection_metrics = analyser.analyse_detection_performance(data_loader, thresholds=args.detection_thresholds)
    
    logger.info(f"\n{split_name.upper()} DETECTION METRICS:")
    for threshold, metrics in detection_metrics.items():
        logger.info(f"At {threshold:.3%} threshold:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Sensitivity at 95% specificity: {metrics['sensitivity_at_95spec']:.4f}")
        logger.info(f"  Average precision: {metrics['average_precision']:.4f}")
    
    # Log background statistics
    if bg_levels is not None:
        logger.info(f"\nBACKGROUND CORRECTION STATISTICS:")
        logger.info(f"  Mean Background: {np.mean(bg_levels):.6f}")
        logger.info(f"  Median Background: {np.median(bg_levels):.6f}")
        logger.info(f"  Min Background: {np.min(bg_levels):.6f}")
        logger.info(f"  Max Background: {np.max(bg_levels):.6f}")
        logger.info(f"  Background Range: {self.bg_correction.min_bg:.6f} - {self.bg_correction.max_bg:.6f}")
    
    # Save results
    results = {
        'predictions': all_preds.flatten().tolist(),
        'targets': all_targets.flatten().tolist(),
        'lower_ci': all_lower_ci.flatten().tolist(),
        'upper_ci': all_upper_ci.flatten().tolist(),
        'metrics': {
            'r2': float(r2),
            'mae': float(mae),
            'in_ci_percentage': float(in_ci * 100),
            'ci_width': float(ci_width)
        },
        'concentration_metrics': concentration_metrics,
        'detection_metrics': {
            str(float(k)): {
                'auc': float(v['auc']), 
                'sensitivity_at_95spec': float(v['sensitivity_at_95spec']),
                'average_precision': float(v['average_precision'])
            } for k, v in detection_metrics.items()
        },
        'background_stats': {
            'mean': float(np.mean(bg_levels)),
            'median': float(np.median(bg_levels)),
            'min': float(np.min(bg_levels)),
            'max': float(np.max(bg_levels)),
            'range_min': float(model.bg_correction.min_bg),
            'range_max': float(model.bg_correction.max_bg)
        } if bg_levels is not None else {}
    }
    
    # Save to file
    results_file = os.path.join(args.output_dir, f'{split_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"{split_name} results saved to {results_file}")
    
    # Create visualisations
    vis_dir = os.path.join(args.output_dir, 'visualisations')
    split_vis_dir = os.path.join(vis_dir, split_name)
    
    # Create standard visualizations
    try:
        from deep_conv.detect.visualise import create_visualisations
        
        os.makedirs(split_vis_dir, exist_ok=True)
        
        logger.info(f"Creating visualisations for {split_name} data...")
        
        # Generate visualisations
        metrics = create_visualisations(
            predictions=all_preds.flatten(), 
            ground_truth=all_targets.flatten(),
            output_dir=split_vis_dir
        )
        
        # Log key metrics
        logger.info(f"{split_name} visualization metrics - "
                   f"R²: {metrics['r2']:.4f}, "
                   f"MAE: {metrics['mae']:.6f}, "
                   f"Within 10%: {metrics['within_10pct']:.2f}%")
        
    except ImportError:
        logger.warning("Visualisation module not found. Skipping visualisations.")
    except Exception as e:
        logger.error(f"Error creating visualisations: {str(e)}")
    
    # Analyse marker importance
    logger.info(f"Analysing marker importance for {split_name} set...")
    top_indices, top_weights = analyser.get_marker_importance(data_loader)
    marker_importance_file = os.path.join(args.output_dir, f'{split_name}_marker_importance.npy')
    np.save(marker_importance_file, all_marker_attentions.mean(axis=0))
    logger.info(f"Marker importance saved to {marker_importance_file}")
    
    # Print top markers
    logger.info(f"Top 5 markers by importance:")
    for i, (idx, weight) in enumerate(zip(top_indices[:5], top_weights[:5])):
        logger.info(f"  #{i+1}: Marker {idx} (weight: {weight:.4f})")
        
    return results

def calibrate_model(model, val_loader, control_loader=None, device='cpu'):
    """
    Calibrate model confidence intervals and background level
    
    Args:
        model: The cancer detection model to calibrate
        val_loader: DataLoader for validation data
        control_loader: Optional DataLoader for control samples
        device: Device to run calibration on
        
    Returns:
        dict: Dictionary containing calibration parameters
    """
    from scipy import stats
    import numpy as np
    
    model.eval()
    
    # 1. Calibrate confidence intervals - use fewer test factors for speed
    best_factor = 1.0
    best_error = float('inf')
    best_low_factor = 1.0
    best_low_error = float('inf')
    
    with torch.no_grad():
        # Test different calibration factors - simplified options
        for factor in [0.5, 1.0, 2.0]:
            coverage_error = 0
            n_batches = 0
            
            for batch_data in val_loader:
                # Handle both 3-element and 4-element returns
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask
                else:
                    marker_values, coverage, y_true = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                mu, uncertainty, _, _ = model(marker_values, coverage)
                
                # Apply test calibration factor
                uncertainty_calibrated = uncertainty * factor
                
                # Calculate CI
                z_score = 1.96  # for 95% CI
                lower = torch.clamp(mu - z_score * uncertainty_calibrated, min=0.0)
                upper = torch.clamp(mu + z_score * uncertainty_calibrated, max=1.0)
                
                # Calculate CI coverage
                in_ci = (y_true >= lower) & (y_true <= upper)
                ci_coverage = in_ci.float().mean().item()
                
                # Error relative to target 95%
                error = abs(ci_coverage - 0.95)
                coverage_error += error
                n_batches += 1
            
            avg_error = coverage_error / n_batches
            if avg_error < best_error:
                best_error = avg_error
                best_factor = factor
        
        # Use the same factor for low concentrations to simplify
        best_low_factor = best_factor
    
    # 2. Calibrate background level using controls - more aggressive
    background_results = {}
    if control_loader is not None:
        all_preds = []
        
        with torch.no_grad():
            for batch_data in control_loader:
                if len(batch_data) == 4:
                    marker_values, coverage, _, _ = batch_data
                else:
                    marker_values, coverage, _ = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                
                mu, _, _, _ = model(marker_values, coverage)
                all_preds.append(mu.cpu().numpy())
        
        # Use 95th percentile instead of median for more conservative background correction
        all_preds = np.concatenate(all_preds)
        global_bg_level = float(np.percentile(all_preds, 95))
        
        # Make sure background level is at least 0.05
        global_bg_level = max(global_bg_level, 0.05)
        
        # Update background level parameter
        with torch.no_grad():
            model.background_level.fill_(global_bg_level)
                
        background_results = {
            'global_bg_level': global_bg_level
        }
    
    # Apply calibration factors to model
    with torch.no_grad():
        # Update model parameters if they exist
        model.calibration.copy_(torch.tensor([best_factor]))
    
    # Return calibration parameters
    calibration_results = {
        'calibration_factor': best_factor,
        'low_calibration_factor': best_low_factor,
        'coverage_error': float(best_error),
    }
    
    # Merge with background results if available
    if background_results:
        calibration_results.update(background_results)
    
    return calibration_results

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


# qrsh -b y -l h_vmem=2g -pe smp 32 -V -N train_t -wd /users/zetzioni/sharedscratch/deepconv/src -o ~/sharedscratch/logs/train_t.log "cd /users/zetzioni/sharedscratch/deepconv/src && python -m deep_conv.detect.train \
# --name CpGenie_T-cells \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell \
# --data_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train_single_cell_clinical/T-cells/ \
# --atlas_path /users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed \
# --target_cell_type T-cells \
# --target_cell_idx 11 \
# --grad_accum_steps 8 \
# --dropout_rate 0.2 \
# --l2_weight 0.05 \
# --feature_dim 96 \
# --num_heads 6 \
# --num_layers 2 \
# --cell_profile ultra_low_snr \
# --calibrate \
# --detection_thresholds "0.001,0.005,0.01,0.05" \
# --min_reliable_coverage 5.0 \
# --epochs 100"



# OAC
# qrsh -b y -l h_vmem=2g -pe smp 32 -V -N train_oac -wd /users/zetzioni/sharedscratch/deepconv/src -o ~/sharedscratch/logs/train_oac.log "cd /users/zetzioni/sharedscratch/deepconv/src && python -m deep_conv.detect.train \
# --name CpGenie_OAC \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell \
# --data_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train_single_cell_clinical/OAC/ \
# --atlas_path /users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed \
# --target_cell_type OAC \
# --target_cell_idx 9 \
# --cell_profile high_snr \
# --dropout_rate 0.2 \
# --l2_weight 0.05 \
# --feature_dim 96 \
# --num_heads 6 \
# --num_layers 2 \
# --detection_thresholds "0.001,0.005,0.01,0.05" \
# --min_reliable_coverage 5.0 \
# --control_data_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/ \
# --calibrate \
# --excluded_markers "44,58,111,133,77,95,127,38,108,115" \
# --epochs 100 

def main():
    """
    Main function with enhanced approach to training and calibration
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
    logger.info(f"Starting enhanced cancer detection training pipeline")
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {args.output_dir}")
    
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
    
    # Initialize enhanced model
    logger.info(f"Initializing enhanced model with {num_markers} markers...")
    try:
        model = EnhancedCancerDetectionModel(
            num_markers=num_markers,
            feature_dim=args.feature_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            detection_thresholds=args.detection_thresholds,
            min_reliable_coverage=args.min_reliable_coverage
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✓ Enhanced model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    except Exception as e:
        logger.error(f"× Error initializing enhanced model: {str(e)}")
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
            calibration_results = model.calibrate(
                val_loader, 
                control_val_loader, 
                device
            )
            # Update best model state with calibration results
            best_model_state['calibration'] = calibration_results
            # Save updated best model
            torch.save(best_model_state, os.path.join(args.output_dir, 'best_model_calibrated.pt'))
            
            logger.info(f"✓ Model calibrated:")
            logger.info(f"  Calibration factor: {calibration_results['calibration_factor']:.4f}")
            if 'min_bg' in calibration_results:
                logger.info(f"  Background range: {calibration_results['min_bg']:.6f} - {calibration_results['max_bg']:.6f}")
            
        except Exception as e:
            logger.error(f"× Error during calibration: {str(e)}")
            logger.info("  Continuing without calibration")
    
    # Evaluate on test set
    logger.info("Evaluating final model on test set...")
    try:
        # Use the evaluate function for full test evaluation
        test_results = evaluate(model, test_loader, args, device, split_name="test")
        logger.info(f"Test evaluation completed successfully and saved to {args.output_dir}/test_results.json")
        
    except Exception as e:
        logger.error(f"× Error during test evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("\n" + "="*60)
    logger.info("ENHANCED MODEL TRAINING COMPLETED SUCCESSFULLY")
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