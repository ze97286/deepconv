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
from deep_conv.detect.visualise import *

from deep_conv.detect.preprocess import prepare_data_for_training, load_train_with_contrastive_data
from deep_conv.detect.model import EnhancedCancerDetectionModel, MarkerImportanceAnalyser


def parse_args():
    parser = argparse.ArgumentParser(description='Train improved cfDNA methylation cancer detection model')
    
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
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularization')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    
    parser.add_argument('--snr_profile', type=str, default='medium', 
                       choices=['high', 'medium', 'low'],
                       help='Signal-to-noise profile for the cell type')
    parser.add_argument('--detection_loss_weight', type=float, default=0.3, 
                       help='Weight of detection loss relative to concentration loss')
    parser.add_argument('--marker_specific_bg', action='store_true',
                       help='Use marker-specific background correction')
    parser.add_argument('--min_reliable_coverage', type=float, default=5.0,
                       help='Minimum coverage to consider a marker reliable')
    parser.add_argument('--enable_adaptive_thresholds', action='store_true',
                       help='Enable adaptive detection thresholds')
    parser.add_argument('--critical_ranges', type=str, default=None,
                       help='Comma-separated triplets of low,high,weight for critical ranges')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimiser')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default="./saved_models", help='Output directory')
 
    # Evaluation parameters
    parser.add_argument('--detection_thresholds', type=str, default="0.0005,0.001,0.005,0.01,0.05", 
                       help='Comma-separated detection thresholds')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    
    parser.add_argument('--control_data_dir', type=str, default=None, 
                   help='Directory containing control data for contrastive learning')
    parser.add_argument('--calibrate', action='store_true', 
                    help='Calibrate confidence intervals and background correction')
    parser.add_argument('--calibrate_every', type=int, default=5,
                    help='Calibrate model every N epochs')

    args = parser.parse_args()
    
    if args.name:
        dir_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"enhanced_model_{timestamp}"
    
    args.output_dir = os.path.join(args.output_dir, dir_name)

    # Process detection thresholds
    args.detection_thresholds = [float(x) for x in args.detection_thresholds.split(',')]
    
    # Process critical ranges if provided
    if args.critical_ranges:
        # Parse triplets: low,high,weight;low,high,weight;...
        ranges = []
        triplets = args.critical_ranges.split(';')
        for triplet in triplets:
            low, high, weight = map(float, triplet.split(','))
            ranges.append((low, high, weight))
        args.critical_ranges = ranges
    else:
        # Use default ranges based on SNR profile
        if args.snr_profile == "high":
            args.critical_ranges = [(0.0005, 0.001, 2.0), (0.001, 0.005, 3.0), (0.005, 0.01, 1.5), (0.01, 0.05, 1.2)]
        elif args.snr_profile == "medium":
            args.critical_ranges = [(0.001, 0.005, 3.0), (0.005, 0.02, 1.5), (0.02, 0.1, 1.2)]
        else:  # "low"
            args.critical_ranges = [(0.001, 0.01, 3.0), (0.01, 0.05, 1.8), (0.05, 0.2, 1.5)]

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
    logger = logging.getLogger('cancer_detection')
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

def calculate_loss(model, mu, uncertainty, detection_probs, y_true, args, control_mask=None):
    """Calculate combined loss with specialized handling for T-cells and OAC"""
    # Check if we're dealing with T-cells or OAC
    is_tcells = hasattr(args, 'target_cell_type') and args.target_cell_type.lower() == 't-cells'
    is_oac = hasattr(args, 'target_cell_type') and args.target_cell_type.lower() == 'oac'
    
    # Get concentration loss using model's compute_loss method
    concentration_loss = model.compute_loss(mu, uncertainty, y_true, control_mask)
    
    # Calculate detection losses
    detection_losses = []
    sens_spec_reg_losses = []  # Sensitivity-specificity regularization losses
    
    for i, threshold in enumerate(args.detection_thresholds):
        binary_y = (y_true >= threshold).float()
        
        # Apply focal loss weighting for imbalanced detection
        pos_weight = torch.sum(1 - binary_y) / torch.sum(binary_y) if torch.sum(binary_y) > 0 else torch.tensor(1.0, device=mu.device)
        
        # Different positive weighting based on cell type
        if is_tcells:
            pos_weight = torch.clamp(pos_weight * 1.5, 2.0, 15.0)  # Higher weight range for T-cells
        elif is_oac:
            # Lower weight range for OAC to prevent extreme values
            pos_weight = torch.clamp(pos_weight * 0.7, 1.0, 3.0)
        else:
            pos_weight = torch.clamp(pos_weight, 1.0, 10.0)
        
        # Focal loss component
        gamma = 2.0
        if is_oac:
            gamma = 1.0
        elif is_tcells:
            gamma = 2.5
            
        pt = binary_y * detection_probs[i] + (1 - binary_y) * (1 - detection_probs[i])
        focal_weight = torch.pow(1 - pt, gamma)
        
        # Weighted BCE loss
        bce_loss = F.binary_cross_entropy(detection_probs[i], binary_y, reduction='none')
        weighted_loss = bce_loss * focal_weight
        
        # Higher weight to positive samples at the decision boundary
        if threshold < 0.01:  # Ultra-low concentration detection
            # For OAC, use milder threshold weighting
            if is_oac:
                near_threshold_mask = (y_true >= threshold * 0.8) & (y_true <= threshold * 1.2)
                threshold_weight = torch.ones_like(weighted_loss, device=weighted_loss.device)
                threshold_weight = torch.where(near_threshold_mask, 
                                            torch.ones_like(threshold_weight, device=threshold_weight.device) * 1.5,
                                            threshold_weight)
            else:
                # Original logic for other cell types
                near_threshold_mask = (y_true >= threshold * 0.7) & (y_true <= threshold * 1.3)
                threshold_weight = torch.ones_like(weighted_loss, device=weighted_loss.device)
                near_threshold_multiplier = 3.0 if is_tcells else 2.0
                threshold_weight = torch.where(near_threshold_mask, 
                                            torch.ones_like(threshold_weight, device=threshold_weight.device) * near_threshold_multiplier,
                                            threshold_weight)
                
            weighted_loss = weighted_loss * threshold_weight
            
        det_loss = weighted_loss.mean()
        detection_losses.append(det_loss)
        
        # Sensitivity-specificity regularization
        if torch.sum(binary_y) > 0 and torch.sum(1 - binary_y) > 0:
            # Calculate batch-level sensitivity and specificity
            true_pos = torch.sum(detection_probs[i] * binary_y)
            true_neg = torch.sum((1 - detection_probs[i]) * (1 - binary_y))
            
            total_pos = torch.sum(binary_y)
            total_neg = torch.sum(1 - binary_y)
            
            sensitivity = true_pos / total_pos
            specificity = true_neg / total_neg
            
            # Set targets based on cell type and threshold
            if is_tcells:
                if threshold <= 0.001:
                    target_sensitivity = 0.70
                    target_specificity = 0.98
                    sens_weight = 2.5
                elif threshold <= 0.01:
                    target_sensitivity = 0.75
                    target_specificity = 0.95
                    sens_weight = 3.0
                else:
                    target_sensitivity = 0.80
                    target_specificity = 0.93
                    sens_weight = 2.0
            elif is_oac:  # Special targets for OAC
                # More balanced targets for OAC's high SNR
                if threshold <= 0.001:
                    target_sensitivity = 0.75
                    target_specificity = 0.85
                    sens_weight = 0.8
                elif threshold <= 0.01:
                    target_sensitivity = 0.70
                    target_specificity = 0.85
                    sens_weight = 0.7
                else:
                    target_sensitivity = 0.65
                    target_specificity = 0.85
                    sens_weight = 0.6
            else:  # Standard targets for all other cell types
                if threshold <= 0.001:
                    target_sensitivity = 0.80
                    target_specificity = 0.90
                    sens_weight = 1.0
                else:
                    target_sensitivity = 0.75
                    target_specificity = 0.90
                    sens_weight = 1.0
            
            # Convert targets to tensors
            target_sens = torch.tensor(target_sensitivity, device=sensitivity.device)
            target_spec = torch.tensor(target_specificity, device=specificity.device)
            
            # L1 loss for sensitivity and specificity
            sens_loss = F.l1_loss(sensitivity, target_sens)
            spec_loss = F.l1_loss(specificity, target_spec)
            
            # Handle penalties based on cell type
            if is_oac:
                # For OAC, use symmetric penalties to prevent extremes in both directions
                min_sens_threshold = torch.tensor(0.50, device=sensitivity.device)
                max_sens_threshold = torch.tensor(0.85, device=sensitivity.device)
                min_spec_threshold = torch.tensor(0.70, device=specificity.device)
                max_spec_threshold = torch.tensor(0.95, device=specificity.device)
                
                # Penalties that prevent both too low and too high values
                sens_low_penalty = torch.pow(torch.clamp(min_sens_threshold - sensitivity, min=0.0), 2) * 3.0
                sens_high_penalty = torch.pow(torch.clamp(sensitivity - max_sens_threshold, min=0.0), 2) * 3.0
                spec_low_penalty = torch.pow(torch.clamp(min_spec_threshold - specificity, min=0.0), 2) * 3.0
                spec_high_penalty = torch.pow(torch.clamp(specificity - max_spec_threshold, min=0.0), 2) * 1.0
                
                # Combined penalty
                extreme_penalty = sens_low_penalty + sens_high_penalty + spec_low_penalty + spec_high_penalty
                
                # Combined loss with symmetric penalties
                balance_loss = sens_weight * sens_loss + spec_loss + extreme_penalty
                
            elif is_tcells:
                # Special asymmetric loss for T-cells
                sens_diff = target_sens - sensitivity
                sens_loss = torch.where(sens_diff > 0, 
                                      sens_diff * sens_diff * 3.0,
                                      sens_diff * sens_diff * 0.5)
                
                spec_loss = F.smooth_l1_loss(specificity, target_spec)
                
                # Minimal extreme penalties
                min_sens_threshold = torch.tensor(0.4 if is_tcells else 0.5, device=sensitivity.device)
                min_spec_threshold = torch.tensor(0.7, device=specificity.device)
                
                extreme_sens_penalty = torch.pow(torch.clamp(min_sens_threshold - sensitivity, min=0.0), 2) * 5.0
                extreme_spec_penalty = torch.pow(torch.clamp(min_spec_threshold - specificity, min=0.0), 2) * 5.0
                
                # Combined loss with simplified weights
                balance_loss = sens_weight * sens_loss + spec_loss + extreme_sens_penalty + extreme_spec_penalty
            else:
                # Simple L1 loss for everything else
                sens_loss = F.l1_loss(sensitivity, target_sens)
                spec_loss = F.l1_loss(specificity, target_spec)
                
                # Minimal extreme penalties
                min_sens_threshold = torch.tensor(0.5, device=sensitivity.device)
                min_spec_threshold = torch.tensor(0.7, device=specificity.device)
                
                extreme_sens_penalty = torch.pow(torch.clamp(min_sens_threshold - sensitivity, min=0.0), 2) * 5.0
                extreme_spec_penalty = torch.pow(torch.clamp(min_spec_threshold - specificity, min=0.0), 2) * 5.0
                
                # Combined loss with default weights
                balance_loss = sens_weight * sens_loss + spec_loss + extreme_sens_penalty + extreme_spec_penalty
            
            sens_spec_reg_losses.append(balance_loss)
    
    # Use moderate detection loss weight, reduced for OAC
    detection_loss_weight = args.detection_loss_weight
    if is_oac:
        detection_loss_weight *= 0.6  # 40% reduction for OAC
    
    combined_detection_loss = sum(detection_losses) / len(detection_losses)
    
    # Add sensitivity-specificity regularization with reduced weight for OAC
    sens_spec_reg_weight = detection_loss_weight * (0.9 if is_tcells else 0.3 if is_oac else 0.5)
    sens_spec_reg_loss = sum(sens_spec_reg_losses) / len(sens_spec_reg_losses) if sens_spec_reg_losses else torch.tensor(0.0, device=mu.device)
    
    # Calculate total loss
    total_loss = concentration_loss + detection_loss_weight * combined_detection_loss + sens_spec_reg_weight * sens_spec_reg_loss
    
    return total_loss, concentration_loss, combined_detection_loss

def train_model(model, train_loader, val_loader, control_loader, args, device):
    """
    Enhanced training with focused optimization for low concentrations, 
    sensitivity-specificity balance, and detection stability monitoring
    
    Args:
        model: The improved cancer detection model to train
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
    
    # Add epoch tracking to model
    model.epoch = 0
    
    # Setup optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with one-cycle policy
    # Faster warmup and cosine decay
    total_steps = len(train_loader) * args.epochs
    pct_start = 0.1  # 10% warmup
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
        div_factor=25.0,  # Initial LR = max_lr / 25
        final_div_factor=1000.0  # Final LR = max_lr / 1000
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_val_low_conc_error = float('inf') 
    best_balance_score = 0.0  
    best_stability_score = 0.0  
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'concentration_loss': [],
        'detection_loss': [],
        'r2_score': [],
        'mean_absolute_error': [],
        'low_conc_error': [],  
        'balance_score': [],   
        'stability_score': [], 
        'sensitivity': [],     
        'specificity': [],     
        'concentration_metrics': [],
        'clinical_metrics': [],
        'lr': []
    }
    
    # Get key low concentration threshold for monitoring
    # This is the key threshold we care most about improving
    if hasattr(args, 'target_cell_type') and args.target_cell_type.lower() == 't-cells':
        # For T-cells, use 1% (0.01) as the key threshold, NOT 0.1%
        key_low_threshold = min(t for t in args.detection_thresholds if t >= 0.01)
        logger.info(f"Using {key_low_threshold} as key threshold for T-cells evaluation")
    else:
        # For other cell types like OAC, use 0.1% (0.001) as before
        key_low_threshold = min(t for t in args.detection_thresholds if t >= 0.001)
    
    # Detection stability tracking
    previous_sens = None
    previous_spec = None
    sens_history = []
    spec_history = []
    stability_window = 5  # Track stability over this many epochs
    
    # Start training loop
    for epoch in range(args.epochs):
        # Update epoch counter in model
        model.epoch = epoch
        
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
                    print(mu)
                    print(uncertainty)
                    print(detection_probs)
                    print(marker_values.iloc[9])
                    
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
        
        # Extract error for low concentration range
        low_conc_error = 0.0
        total_low_conc_samples = 0
        
        for range_name, metrics in conc_metrics['stratified_metrics'].items():
            # Focus on the ranges we care most about for OAC
            if args.snr_profile == "high" and range_name in ['0.05-0.1%', '0.1-0.5%', '0.5-1%']:
                low_conc_error += metrics.get('mae', 0) * metrics.get('count', 0)
                total_low_conc_samples += metrics.get('count', 0)
            elif args.snr_profile == "medium" and range_name in ['0.1-0.5%', '0.5-1%', '1-5%']:
                low_conc_error += metrics.get('mae', 0) * metrics.get('count', 0)
                total_low_conc_samples += metrics.get('count', 0)
            elif args.snr_profile == "low" and range_name in ['0.5-1%', '1-5%', '5-10%']:
                low_conc_error += metrics.get('mae', 0) * metrics.get('count', 0)
                total_low_conc_samples += metrics.get('count', 0)
                
        # Calculate weighted error across low concentration ranges
        if total_low_conc_samples > 0:
            low_conc_error /= total_low_conc_samples
        
        # Calculate sensitivity-specificity balance score for key threshold
        balance_score = 0.0
        stability_score = 0.0
        current_sens = 0.0
        current_spec = 0.0
        
        if key_low_threshold in val_metrics['clinical_metrics']:
            metrics_key = val_metrics['clinical_metrics'][key_low_threshold]
            cell_type_str = "T-cells" if (hasattr(args, 'target_cell_type') and args.target_cell_type.lower() == 't-cells') else "cell"
            logger.info(f"  At {key_low_threshold*100:.3f}% threshold ({cell_type_str} detection) - "
                    f"Sensitivity: {metrics_key['sensitivity']:.4f}, "
                    f"Specificity: {metrics_key['specificity']:.4f}, "
                    f"AUC: {metrics_key.get('roc_auc', 0):.4f}")
            current_sens = metrics_key['sensitivity']
            current_spec = metrics_key['specificity']
            
            # Track sensitivity and specificity history
            sens_history.append(current_sens)
            spec_history.append(current_spec)
            if len(sens_history) > stability_window:
                sens_history.pop(0)
                spec_history.pop(0)
            
            # Use F-beta score with beta=2 to give more weight to sensitivity while maintaining specificity
            beta = 2  # Adjust beta value to control sensitivity-specificity tradeoff
            beta_squared = beta ** 2
            if current_sens > 0 and current_spec > 0:
                balance_score = (1 + beta_squared) * (current_sens * current_spec) / (beta_squared * current_sens + current_spec)
            
            # NEW: Calculate stability score
            # This rewards models that maintain consistent sensitivity/specificity
            if previous_sens is not None and previous_spec is not None:
                # Calculate stability as inverse of change in sens/spec
                sens_change = abs(current_sens - previous_sens)
                spec_change = abs(current_spec - previous_spec)
                
                # Calculate longer-term stability using history
                if len(sens_history) >= 3:
                    sens_std = np.std(sens_history)
                    spec_std = np.std(spec_history)
                    
                    # Exponentially penalize high variability
                    long_term_penalty = np.exp(10 * (sens_std + spec_std)) - 1
                    
                    # Stability score is higher when changes are smaller
                    # and when history shows less variability
                    stability_score = 1.0 / (1.0 + sens_change + spec_change + long_term_penalty)
                else:
                    # Simpler stability score for early epochs
                    stability_score = 1.0 / (1.0 + sens_change + spec_change)
            
            # Update previous values for next epoch
            previous_sens = current_sens
            previous_spec = current_spec
            
            # Log the balance and stability scores
            logger.info(f"  Sensitivity-Specificity Balance (F{beta}): {balance_score:.4f}")
            logger.info(f"  Stability Score: {stability_score:.4f}")
        
        # Periodic calibration
        if args.calibrate and (epoch % args.calibrate_every == 0 or epoch == args.epochs - 1):
            logger.info(f"Calibrating model...")
            calibration_results = model.calibrate(val_loader, control_loader, device)
            logger.info(f"  Calibration factor: {calibration_results['calibration_factor']:.4f}")
            if 'global_bg_level' in calibration_results:
                logger.info(f"  Background level: {calibration_results['global_bg_level']:.6f}")
                logger.info(f"  Background range: {calibration_results['min_bg']:.6f}-{calibration_results['max_bg']:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['concentration_loss'].append(concentration_loss)
        history['detection_loss'].append(detection_loss)
        history['r2_score'].append(val_metrics['r2'])
        history['mean_absolute_error'].append(val_metrics['mae'])
        history['low_conc_error'].append(low_conc_error)
        history['balance_score'].append(balance_score)
        history['stability_score'].append(stability_score)
        history['sensitivity'].append(current_sens)
        history['specificity'].append(current_spec)
        history['clinical_metrics'].append(val_metrics['clinical_metrics'])
        history['concentration_metrics'].append(conc_metrics)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # Log validation results
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.6f}, "
                   f"Val Loss: {val_loss:.6f}, "
                   f"R²: {val_metrics['r2']:.4f}, "
                   f"MAE: {val_metrics['mae']:.6f}, "
                   f"Low Conc. Error: {low_conc_error:.6f}")
        
        # Log clinical metrics
        # Focus on metrics at key_low_threshold (likely 0.001)
        if key_low_threshold in val_metrics['clinical_metrics']:
            metrics_key = val_metrics['clinical_metrics'][key_low_threshold]
            logger.info(f"  At {key_low_threshold*100:.3f}% threshold - "
                       f"Sensitivity: {metrics_key['sensitivity']:.4f}, "
                       f"Specificity: {metrics_key['specificity']:.4f}, "
                       f"AUC: {metrics_key.get('roc_auc', 0):.4f}")
        
        # Log concentration metrics for key ranges
        # Adjust ranges based on SNR profile
        if args.snr_profile == "high":
            key_ranges = [
                '0.05-0.1%',   # 0.0005-0.001
                '0.1-0.5%',    # 0.001-0.005
                '0.5-1%',      # 0.005-0.01
                '1-2.5%',      # 0.01-0.025 (new)
                '2.5-5%',      # 0.025-0.05 (new)
                '5-10%'        # 0.05-0.1 (new)
            ]
        elif args.snr_profile == "medium":
            key_ranges = [
                '0.1-0.5%',    # 0.001-0.005
                '0.5-1%',      # 0.005-0.01
                '1-2.5%',      # 0.01-0.025 (new)
                '2.5-5%',      # 0.025-0.05 (new)
                '5-10%'        # 0.05-0.1 (new)
            ]
        else:  # "low"
            key_ranges = [
                '0.5-1%',      # 0.005-0.01
                '1-2.5%',      # 0.01-0.025 (new)
                '2.5-5%',      # 0.025-0.05 (new)
                '5-10%',       # 0.05-0.1 (new)
                '10-20%'       # 0.1-0.2 (new)
            ]
            
        for range_name in key_ranges:
            if range_name in conc_metrics['stratified_metrics']:
                range_metrics = conc_metrics['stratified_metrics'][range_name]
                logger.info(f"  {range_name} (n={range_metrics['count']}): "
                           f"MAE={range_metrics['mae']:.6f}, "
                           f"Within 25%={range_metrics.get('within_25pct', 0):.1f}%")
        
        # Check for improvement with combined metrics including stability
        improvement = False
        improvement_msg = ""
        
        # Calculate combined score that values stability more as training progresses
        # Early in training focus on metrics, later focus more on stability
        if epoch < args.epochs * 0.3:  # First 30% of training: focus on metrics
            stability_weight = 0.2
        elif epoch < args.epochs * 0.7:  # Middle 40% of training: balanced approach
            stability_weight = 0.5
        else:  # Last 30% of training: prioritize stability
            stability_weight = 0.8
            
        # First check if we have a new best model based on validation loss
        if val_loss < best_val_loss:
            improvement = True
            improvement_msg = f"New best model (loss)! Improvement: {(best_val_loss - val_loss) / best_val_loss * 100:.2f}%"
            best_val_loss = val_loss
            
        # Also check if we have a new best model based on low concentration error
        if low_conc_error < best_val_low_conc_error:
            improvement = True
            if improvement_msg:
                improvement_msg += " and "
            improvement_msg += f"Low conc. error improved: {(best_val_low_conc_error - low_conc_error) / best_val_low_conc_error * 100:.2f}%"
            best_val_low_conc_error = low_conc_error
        
        # Check for improvement in sensitivity-specificity balance with a minimum threshold
        # Only consider balance if it's at least 0.7 (to avoid saving models with bad balance)
        balance_improvement_threshold = 0.03  # 3% improvement required
        min_balance_score = 0.4 if hasattr(args, 'target_cell_type') and args.target_cell_type == 'T-cells' else 0.7
        if balance_score > min_balance_score and balance_score > best_balance_score * (1 + balance_improvement_threshold):
            improvement = True
            if improvement_msg:
                improvement_msg += " and "
                
            # Calculate improvement percentage safely
            if best_balance_score > 0:
                improvement_pct = (balance_score - best_balance_score) / best_balance_score * 100
            else:
                improvement_pct = 100.0  # First non-zero balance score
                
            improvement_msg += f"Sens/Spec balance improved: {improvement_pct:.2f}%"
            best_balance_score = balance_score
                
        # Check for stability improvement (more important later in training)
        if epoch >= stability_window and stability_score > 0:
            # Add increasing stability weight as training progresses
            if stability_score > best_stability_score * (1 + 0.05):  # 5% improvement required
                # Apply stability weight - easier to improve earlier, harder later
                if stability_weight * stability_score > stability_weight * best_stability_score * (1 + 0.05):
                    improvement = True
                    if improvement_msg:
                        improvement_msg += " and "
                        
                    # Calculate improvement percentage
                    improvement_pct = (stability_score - best_stability_score) / max(best_stability_score, 0.01) * 100
                    improvement_msg += f"Stability improved: {improvement_pct:.2f}%"
                    best_stability_score = stability_score
        
        # Additional validation for extreme values - don't save models with extreme sens/spec
        is_tcells = hasattr(args, 'target_cell_type') and args.target_cell_type.lower() == 't-cells'
        is_oac = hasattr(args, 'target_cell_type') and args.target_cell_type.lower() == 'oac'

        if is_tcells:
            # More lenient criteria for T-cells due to low SNR
            if current_sens < 0.3 or current_spec < 0.7:
                logger.info(f"  ⚠️ Rejecting model save due to extreme sensitivity ({current_sens:.4f}) or specificity ({current_spec:.4f})")
                improvement = False
                improvement_msg = ""
        elif is_oac:
            # Moderate criteria for OAC
            if (current_sens < 0.4 and current_spec < 0.95) or current_sens < 0.3:
                logger.info(f"  ⚠️ Rejecting model save due to extreme sensitivity ({current_sens:.4f}) or specificity ({current_spec:.4f})")
                improvement = False
                improvement_msg = ""
        else:
            # Standard criteria for all other cell types (including OAC)
            if current_sens < 0.5 or current_sens > 0.95 or current_spec < 0.5 or current_spec > 0.95:
                logger.info(f"  ⚠️ Rejecting model save due to extreme sensitivity ({current_sens:.4f}) or specificity ({current_spec:.4f})")
                improvement = False
                improvement_msg = ""
                    
        # If there's improvement in any metric, save the model
        if improvement:
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'low_conc_error': low_conc_error,
                'balance_score': balance_score,
                'stability_score': stability_score,
                'concentration_metrics': conc_metrics,
                'args': vars(args),
                'git_commit': git_commit,
            }
            
            # Save best model
            torch.save(best_model_state, os.path.join(args.output_dir, 'best_model.pt'))
            logger.info(f"✓ {improvement_msg}")
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
                'git_commit': git_commit,
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
                        serializable_metrics_dict = {}
                        for k, v in metrics.items():
                            if v is not None:
                                if isinstance(v, dict):
                                    # Handle nested dictionaries
                                    serializable_metrics_dict[k] = {
                                        sk: float(sv) if isinstance(sv, (int, float)) else sv 
                                        for sk, sv in v.items()
                                    }
                                else:
                                    serializable_metrics_dict[k] = float(v)
                        serializable_epoch[str(threshold)] = serializable_metrics_dict
                    serializable_metrics.append(serializable_epoch)
                serializable_history[key] = serializable_metrics
            elif key == 'concentration_metrics':
                # Handle concentration metrics
                serializable_metrics = []
                for epoch_metrics in values:
                    serializable_epoch = {}
                    for range_name, metrics in epoch_metrics.get('stratified_metrics', {}).items():
                        serializable_epoch[range_name] = {
                            k: float(v) if v is not None and not isinstance(v, dict) else None 
                            for k, v in metrics.items()
                        }
                    serializable_metrics.append(serializable_epoch)
                serializable_history[key] = serializable_metrics
        
        json.dump(serializable_history, f, indent=2)
    
    # Create plots of training history
    plot_training_history(history, args.output_dir)
    
    # Load best model for return
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model'])
    
     # Evaluate on validation set to get predictions
    val_preds = []
    val_targets = []
    val_lower_ci = []
    val_upper_ci = []
    val_attentions = []
    
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            # Handle both dataset types
            if len(batch_data) == 4:
                marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask
            else:
                marker_values, coverage, y_true = batch_data
                
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            
            # Forward pass
            mu, uncertainty, _, attention_weights = model(marker_values, coverage)
            estimate, ci, _ = model.get_estimate_and_ci(mu, uncertainty)
            
            # Store results
            val_preds.append(estimate.cpu().numpy())
            val_targets.append(y_true.cpu().numpy())
            val_lower_ci.append(ci[:, 0:1].cpu().numpy())
            val_upper_ci.append(ci[:, 1:2].cpu().numpy())
            val_attentions.append(attention_weights.cpu().numpy())
    
    # Concatenate results
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_lower_ci = np.concatenate(val_lower_ci)
    val_upper_ci = np.concatenate(val_upper_ci)
    val_attentions = np.concatenate(val_attentions)
    
    # Create visualization directory
    vis_dir = os.path.join(args.output_dir, 'visualizations', 'val')
    
    # Call visualise_results to create visualizations
    logger.info("Creating validation set visualizations...")
    visualise_results(
        val_preds, 
        val_targets, 
        vis_dir,
        ci_data=(val_lower_ci, val_upper_ci),
        marker_importance=val_attentions.mean(axis=0),
        prefix="Validation "
    )
    
    # Return model and best model state
    return model, best_model_state

def compute_concentration_metrics(model, data_loader, device):
    """
    Compute concentration-focused metrics with emphasis on accuracy at low concentrations
    
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
    Compute enhanced concentration-aware metrics with finer-grained ranges and better error measures
    
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
    
    # Define finer-grained concentration ranges focused on lower concentrations
    ranges = [
        (0, 0.0005, "0-0.05%"),
        (0.0005, 0.001, "0.05-0.1%"),
        (0.001, 0.005, "0.1-0.5%"),
        (0.005, 0.01, "0.5-1%"),
        (0.01, 0.025, "1-2.5%"),  
        (0.025, 0.05, "2.5-5%"),  
        (0.05, 0.1, "5-10%"),
        (0.1, 0.2, "10-20%"),     
        (0.2, 1.0, ">20%")        
    ]
    
    # Initialize results dict
    results = {
        'stratified_metrics': {},
        'band_accuracy': {},
        'overall': {}
    }
    
    # Calculate overall metrics
    results['overall']['r2'] = r2_score(targets, predictions)
    results['overall']['mae'] = mean_absolute_error(targets, predictions)
    
    # Calculate within percentage bands of true value (overall)
    non_zero_mask = (targets > 1e-6)
    if np.sum(non_zero_mask) > 0:
        rel_errors = np.abs(predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask]
        results['overall']['within_10pct'] = np.mean(rel_errors <= 0.1) * 100
        results['overall']['within_25pct'] = np.mean(rel_errors <= 0.25) * 100
        results['overall']['within_50pct'] = np.mean(rel_errors <= 0.5) * 100
    
    # Calculate stratified metrics for each concentration range
    for low, high, name in ranges:
        mask = (targets >= low) & (targets < high)
        range_predictions = predictions[mask]
        range_targets = targets[mask]
        
        if len(range_targets) > 0:
            # Calculate range-specific metrics
            range_mae = mean_absolute_error(range_targets, range_predictions)
            range_mse = np.mean((range_predictions - range_targets) ** 2)
            range_rmse = np.sqrt(range_mse)
            
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
            
            # Calculate directional bias (over/under-estimation)
            mean_error = np.mean(range_predictions - range_targets)
            mean_abs_error = np.mean(np.abs(range_predictions - range_targets))
            bias_ratio = mean_error / mean_abs_error if mean_abs_error > 0 else 0
            
            results['stratified_metrics'][name] = {
                'count': int(np.sum(mask)),
                'mae': float(range_mae),
                'rmse': float(range_rmse),
                'within_10pct': float(within_10pct),
                'within_25pct': float(within_25pct),
                'within_50pct': float(within_50pct),
                'bias_ratio': float(bias_ratio)  # Positive = overestimation, negative = underestimation
            }
    
    # Calculate concentration band accuracy - how well the model categorizes samples
    # This measures if samples are placed in the correct concentration range
    bands = [
        (0, 0.001),        # 0-0.1%
        (0.001, 0.01),     # 0.1-1%
        (0.01, 0.05),      # 1-5%
        (0.05, 0.1),       # 5-10%
        (0.1, 1.0)         # >10%
    ]
    band_names = ["0-0.1%", "0.1-1%", "1-5%", "5-10%", ">10%"]
    
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
    
    # Calculate ordering accuracy - preserving the relative order of samples
    n_samples = len(targets)
    if n_samples > 1:
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
    Validate model performance with enhanced metrics and detection evaluation
    
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
    all_detection_probs = []
    all_uncertainties = []
    
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
            all_detection_probs.append([dp.cpu().numpy() for dp in detection_probs])
            all_uncertainties.append(uncertainty.cpu().numpy())
    
    # Calculate average losses
    val_loss /= len(val_loader)
    concentration_loss /= len(val_loader)
    detection_loss /= len(val_loader)
    calibration_error /= len(val_loader)
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_detection_probs = [np.concatenate([batch[i] for batch in all_detection_probs]) 
                           for i in range(len(args.detection_thresholds))]
    all_uncertainties = np.concatenate(all_uncertainties)
    
    # Calculate regression metrics
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Calculate clinical metrics
    clinical_metrics = clinical_performance_metrics(
        all_preds, all_targets, all_detection_probs, args.detection_thresholds
    )
    
    # Calculate concentration metrics
    concentration_metrics = compute_concentration_aware_metrics(all_preds, all_targets)
    
    # Calculate uncertainty quality metrics
    uncertainty_metrics = evaluate_uncertainty_quality(all_preds, all_targets, all_uncertainties)
    
    # Return validation loss and metrics
    metrics = {
        'concentration_loss': concentration_loss,
        'detection_loss': detection_loss,
        'calibration_error': calibration_error,
        'r2': r2,
        'mae': mae,
        'clinical_metrics': clinical_metrics,
        'concentration_metrics': concentration_metrics,
        'uncertainty_metrics': uncertainty_metrics
    }
    
    return val_loss, metrics
            
def evaluate_uncertainty_quality(predictions, targets, uncertainties):
    """
    Evaluate the quality of uncertainty estimates
    
    Args:
        predictions: Predicted concentrations
        targets: Ground truth concentrations
        uncertainties: Predicted uncertainties
        
    Returns:
        Dictionary of uncertainty quality metrics
    """
    # Calculate standardized errors (z-scores)
    z_scores = np.abs(predictions.flatten() - targets.flatten()) / (uncertainties.flatten() + 1e-6)
    
    # For a well-calibrated model, ~68% of errors should be within 1 std dev
    within_1std = np.mean(z_scores <= 1.0) * 100
    
    # For a well-calibrated model, ~95% of errors should be within 2 std dev
    within_2std = np.mean(z_scores <= 2.0) * 100
    
    # Calculate correlation between error magnitude and uncertainty
    abs_errors = np.abs(predictions.flatten() - targets.flatten())
    error_uncertainty_corr = np.corrcoef(abs_errors, uncertainties.flatten())[0, 1]
    
    # Calculate mean uncertainty at different concentration levels
    uncertainty_by_range = {}
    ranges = [
        (0, 0.001, "0-0.1%"),
        (0.001, 0.01, "0.1-1%"),
        (0.01, 0.05, "1-5%"),
        (0.05, 0.1, "5-10%"),
        (0.1, 1.0, ">10%")
    ]
    
    for low, high, name in ranges:
        mask = (targets >= low) & (targets < high)
        if np.sum(mask) > 0:
            range_mean_uncertainty = np.mean(uncertainties[mask])
            range_median_uncertainty = np.median(uncertainties[mask])
            uncertainty_by_range[name] = {
                'mean': float(range_mean_uncertainty),
                'median': float(range_median_uncertainty),
                'count': int(np.sum(mask))
            }
    
    return {
        'within_1std': float(within_1std),
        'within_2std': float(within_2std),
        'error_uncertainty_corr': float(error_uncertainty_corr),
        'mean_uncertainty': float(np.mean(uncertainties)),
        'median_uncertainty': float(np.median(uncertainties)),
        'uncertainty_by_range': uncertainty_by_range
    }

def clinical_performance_metrics(predictions, ground_truth, detection_probs, thresholds):
    """
    Calculate enhanced clinical metrics with detection probability integration
    
    Args:
        predictions: Predicted concentrations (numpy array)
        ground_truth: True concentrations (numpy array)
        detection_probs: Detection probabilities from model's detection heads
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
    
    for i, threshold in enumerate(thresholds):
        # Convert to binary classification
        y_pred_binary = (predictions >= threshold).astype(float)
        y_true_binary = (ground_truth >= threshold).astype(float)
        
        # Add both prediction and detection probability
        # The detection head might be better calibrated for binary decisions
        det_probs = detection_probs[i].flatten()
        
        # Calculate basic metrics using prediction values
        TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        TN = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
        FP = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        
        # Calculate rates
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        
        # Calculate ROC curve and AUC using detection probability (more granular)
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true_binary, det_probs)
            roc_auc = auc(fpr, tpr)
            
            # Find sensitivity at various specificity levels
            spec_levels = [0.95, 0.98, 0.99]
            sens_at_spec = {}
            
            for spec_level in spec_levels:
                fpr_target = 1.0 - spec_level
                idx = np.argmin(np.abs(fpr - fpr_target))
                sens_at_spec[f"{spec_level:.2f}"] = float(tpr[idx])
            
            # Calculate PR curve and average precision
            precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, det_probs)
            avg_precision = average_precision_score(y_true_binary, det_probs)
        except:
            # Handle cases with only one class
            roc_auc = 0
            sens_at_spec = {f"{spec:.2f}": 0 for spec in spec_levels}
            avg_precision = 0
        
        # Calculate magnitude-aware metrics
        # How close are the predictions to the true values?
        if np.sum(y_true_binary) > 0:
            positive_samples = predictions[y_true_binary == 1]
            positive_targets = ground_truth[y_true_binary == 1]
            
            # Mean absolute percentage error for positive samples
            mape = np.mean(np.abs(positive_samples - positive_targets) / np.maximum(positive_targets, 1e-6)) \
                   if len(positive_samples) > 0 else np.nan
                   
            # Percentage of positive samples with error < 25%
            within_25pct = np.mean(np.abs(positive_samples - positive_targets) <= 0.25 * np.maximum(positive_targets, 1e-6)) * 100 \
                           if len(positive_samples) > 0 else np.nan
        else:
            mape = np.nan
            within_25pct = np.nan
        
        # Store all metrics
        results[threshold] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'roc_auc': float(roc_auc),
            'sensitivity_at_specificity': sens_at_spec,
            'avg_precision': float(avg_precision),
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN),
            'mape': float(mape) if not np.isnan(mape) else None,
            'within_25pct': float(within_25pct) if not np.isnan(within_25pct) else None
        }
    
    return results

def plot_training_history(history, output_dir):
    """Create plots of training history with enhanced visualization"""
    import os
    try:
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
                y=history['mean_absolute_error'], 
                mode='lines+markers', 
                name='MAE',
                line=dict(color='green', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Learning rate
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
        
        # Create clinical metrics plot if available
        if 'clinical_metrics' in history and len(history['clinical_metrics']) > 0:
            # Extract metrics for key thresholds
            # Prioritize lower thresholds for ultra-low concentration detection
            key_thresholds = [0.0005, 0.001, 0.005, 0.01]
            threshold_metrics = {}
            
            for threshold in key_thresholds:
                sensitivity_values = []
                specificity_values = []
                
                for epoch_metrics in history['clinical_metrics']:
                    if threshold in epoch_metrics or str(threshold) in epoch_metrics:
                        metrics = epoch_metrics.get(threshold, epoch_metrics.get(str(threshold), {}))
                        sensitivity_values.append(metrics.get('sensitivity', None))
                        specificity_values.append(metrics.get('specificity', None))
                    else:
                        sensitivity_values.append(None)
                        specificity_values.append(None)
                
                # Only include thresholds with enough valid data
                valid_count = sum(1 for v in sensitivity_values if v is not None)
                if valid_count >= len(epochs) // 2:  # At least half of epochs have data
                    threshold_metrics[threshold] = {
                        'sensitivity': sensitivity_values,
                        'specificity': specificity_values
                    }
            
            # Create clinical metrics figure if we have valid data
            if threshold_metrics:
                clinical_fig = go.Figure()
                
                # Add traces for each threshold (sensitivity)
                for threshold, values in threshold_metrics.items():
                    # Get valid epochs and values
                    valid_epochs = [epochs[i] for i in range(len(epochs)) if values['sensitivity'][i] is not None]
                    valid_sens = [values['sensitivity'][i] for i in range(len(epochs)) if values['sensitivity'][i] is not None]
                    
                    if valid_epochs:
                        clinical_fig.add_trace(
                            go.Scatter(
                                x=valid_epochs,
                                y=valid_sens,
                                mode='lines+markers',
                                name=f'Sensitivity ({threshold*100:.3f}%)',
                                line=dict(width=2)
                            )
                        )
                
                # Add reference line at 0.95
                clinical_fig.add_shape(
                    type="line",
                    x0=min(epochs),
                    y0=0.95,
                    x1=max(epochs),
                    y1=0.95,
                    line=dict(color="green", dash="dash"),
                    name="95% Target"
                )
                
                # Update layout
                clinical_fig.update_layout(
                    title='Sensitivity by Threshold',
                    xaxis_title='Epoch',
                    yaxis_title='Sensitivity',
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    width=900,
                    height=500,
                    yaxis=dict(range=[0, 1])
                )
                
                # Save clinical metrics figure
                clinical_fig.write_html(os.path.join(plots_dir, 'sensitivity_by_threshold.html'))
                clinical_fig.write_image(os.path.join(plots_dir, 'sensitivity_by_threshold.png'), scale=2)
                
                # Create specificity plot
                specificity_fig = go.Figure()
                
                # Add traces for each threshold (specificity)
                for threshold, values in threshold_metrics.items():
                    # Get valid epochs and values
                    valid_epochs = [epochs[i] for i in range(len(epochs)) if values['specificity'][i] is not None]
                    valid_spec = [values['specificity'][i] for i in range(len(epochs)) if values['specificity'][i] is not None]
                    
                    if valid_epochs:
                        specificity_fig.add_trace(
                            go.Scatter(
                                x=valid_epochs,
                                y=valid_spec,
                                mode='lines+markers',
                                name=f'Specificity ({threshold*100:.3f}%)',
                                line=dict(width=2)
                            )
                        )
                
                # Add reference line at 0.95 and 0.99
                for spec_target in [0.95, 0.99]:
                    specificity_fig.add_shape(
                        type="line",
                        x0=min(epochs),
                        y0=spec_target,
                        x1=max(epochs),
                        y1=spec_target,
                        line=dict(color="green" if spec_target == 0.95 else "blue", dash="dash"),
                        name=f"{spec_target*100:.0f}% Target"
                    )
                
                # Update layout
                specificity_fig.update_layout(
                    title='Specificity by Threshold',
                    xaxis_title='Epoch',
                    yaxis_title='Specificity',
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    width=900,
                    height=500,
                    yaxis=dict(range=[0.9, 1.0])  # Focus on high specificity region
                )
                
                # Save specificity figure
                specificity_fig.write_html(os.path.join(plots_dir, 'specificity_by_threshold.html'))
                specificity_fig.write_image(os.path.join(plots_dir, 'specificity_by_threshold.png'), scale=2)
        
        # Plot accuracy by concentration range if available
        if 'concentration_metrics' in history and len(history['concentration_metrics']) > 0:
            # Extract metrics for key ranges - adapt for SNR profile
            key_ranges = ['0.05-0.1%', '0.1-0.5%', '0.5-1%', '1-5%']
            range_metrics = {}
            
            # See if we have within_25pct metrics
            for range_name in key_ranges:
                accuracy_values = []
                
                for epoch_metrics in history['concentration_metrics']:
                    if epoch_metrics and 'stratified_metrics' in epoch_metrics:
                        if range_name in epoch_metrics['stratified_metrics']:
                            metrics = epoch_metrics['stratified_metrics'][range_name]
                            accuracy_values.append(metrics.get('within_25pct', None))
                        else:
                            accuracy_values.append(None)
                    else:
                        accuracy_values.append(None)
                
                # Only include ranges with enough valid data
                valid_count = sum(1 for v in accuracy_values if v is not None)
                if valid_count >= len(epochs) // 4:  # At least quarter of epochs have data
                    range_metrics[range_name] = accuracy_values
            
            # Create figure if we have valid data
            if range_metrics:
                accuracy_fig = go.Figure()
                
                # Add traces for each range
                for range_name, values in range_metrics.items():
                    # Get valid epochs and values
                    valid_epochs = [epochs[i] for i in range(len(epochs)) if values[i] is not None]
                    valid_acc = [values[i] for i in range(len(epochs)) if values[i] is not None]
                    
                    if valid_epochs:
                        accuracy_fig.add_trace(
                            go.Scatter(
                                x=valid_epochs,
                                y=valid_acc,
                                mode='lines+markers',
                                name=f'{range_name}',
                                line=dict(width=2)
                            )
                        )
                
                # Add reference lines for target accuracy levels (50% and 70%)
                for target in [50, 70]:
                    accuracy_fig.add_shape(
                        type="line",
                        x0=min(valid_epochs),
                        y0=target,
                        x1=max(valid_epochs),
                        y1=target,
                        line=dict(
                            color="orange" if target == 50 else "green", 
                            dash="dash"
                        ),
                        name=f"{target}% Target"
                    )
                
                # Update layout
                accuracy_fig.update_layout(
                    title='Within 25% Accuracy by Concentration Range',
                    xaxis_title='Epoch',
                    yaxis_title='Percentage within 25% error',
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    width=900,
                    height=500,
                    yaxis=dict(range=[0, 100])
                )
                
                # Save accuracy figure
                accuracy_fig.write_html(os.path.join(plots_dir, 'accuracy_by_range.html'))
                accuracy_fig.write_image(os.path.join(plots_dir, 'accuracy_by_range.png'), scale=2)
    except Exception as e:
        print(f"Error creating plots: {e}. Continuing without plots.")

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
    import pandas as pd
    import numpy as np
    logger = logging.getLogger('cancer_detection')
    
    # Create visualisation directory
    os.makedirs(output_subdir, exist_ok=True)
    
    # Try to import visualisation module
    try:
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
        
        df = pd.DataFrame({
            'true_value': np.array(ground_truth).flatten(),
            'predicted_value': np.array(predictions).flatten()
        })

        # Calculate errors
        df['error'] = df['predicted_value'] - df['true_value']
        df['abs_error'] = np.abs(df['error'])
        rel_error = np.full_like(df['true_value'], np.nan, dtype=float)
        non_zero_mask = df['true_value'] > 0
        rel_error[non_zero_mask] = np.abs(df['predicted_value'][non_zero_mask] - df['true_value'][non_zero_mask]) / df['true_value'][non_zero_mask] * 100
        df['rel_error'] = rel_error

        # Define thresholds
        clinical_thresholds = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        specific_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

        # Add clinical metrics
        clinical_metrics = create_clinical_decision_metrics(df, clinical_thresholds, output_subdir)
        create_threshold_specific_analysis(df, specific_thresholds, output_subdir)

        # Update returned metrics
        metrics['clinical_decision_metrics'] = clinical_metrics

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
    """
    Enhanced evaluation function with detailed analysis for low concentrations
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        args: Training arguments
        device: Device to run evaluation on
        split_name: Name of data split (e.g., "test")
        
    Returns:
        Dictionary of evaluation results
    """
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
    all_uncertainties = []
    all_bg_levels = []
    
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
            mu, uncertainty, det_probs, attention_weights = model(marker_values, coverage)
            estimate, ci, _ = model.get_estimate_and_ci(mu, uncertainty)
            
            # Store results
            all_preds.append(estimate.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
            all_lower_ci.append(ci[:, 0:1].cpu().numpy())
            all_upper_ci.append(ci[:, 1:2].cpu().numpy())
            all_marker_attentions.append(attention_weights.cpu().numpy())
            all_detection_probs.append([dp.cpu().numpy() for dp in det_probs])
            all_uncertainties.append(uncertainty.cpu().numpy())
            
            # Get background levels for a subset of batches (to save time)
            if len(all_bg_levels) == 0:
                try:
                    bg_levels = model.get_background_levels(torch.cat([marker_values[:1], marker_values[-1:]], dim=0), 
                                                           torch.cat([coverage[:1], coverage[-1:]], dim=0), 
                                                           device)
                    all_bg_levels.append(bg_levels)
                except Exception as e:
                    logger.warning(f"Could not get background levels: {e}")
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_lower_ci = np.concatenate(all_lower_ci)
    all_upper_ci = np.concatenate(all_upper_ci)
    all_marker_attentions = np.concatenate(all_marker_attentions)
    all_detection_probs = [np.concatenate([batch[i] for batch in all_detection_probs]) 
                          for i in range(len(args.detection_thresholds))]
    all_uncertainties = np.concatenate(all_uncertainties)
    
    # Calculate metrics
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Calculate percentage of targets within CI
    in_ci = ((all_targets >= all_lower_ci) & (all_targets <= all_upper_ci)).mean()
    
    # Calculate average CI width
    ci_width = (all_upper_ci - all_lower_ci).mean()
    
    # Calculate concentration metrics
    concentration_metrics = compute_concentration_aware_metrics(all_preds, all_targets)
    
    # Calculate uncertainty quality metrics
    uncertainty_metrics = evaluate_uncertainty_quality(all_preds, all_targets, all_uncertainties)
    
    # Log basic results
    logger.info(f"\n{split_name.upper()} RESULTS:")
    logger.info(f"Number of {split_name} samples: {len(all_targets)}")
    logger.info(f"R² Score: {r2:.6f}")
    logger.info(f"Mean Absolute Error: {mae:.6f}")
    logger.info(f"Targets within 95% CI: {in_ci * 100:.2f}%")
    logger.info(f"Average CI Width: {ci_width:.6f}")
    
    # Log uncertainty metrics
    logger.info(f"\nUNCERTAINTY METRICS:")
    logger.info(f"  Errors within 1 std: {uncertainty_metrics['within_1std']:.2f}% (ideal: 68%)")
    logger.info(f"  Errors within 2 std: {uncertainty_metrics['within_2std']:.2f}% (ideal: 95%)")
    logger.info(f"  Error-Uncertainty Correlation: {uncertainty_metrics['error_uncertainty_corr']:.4f}")
    
    # Log concentration metrics
    logger.info(f"\nCONCENTRATION METRICS:")
    for range_name, metrics in concentration_metrics['stratified_metrics'].items():
        logger.info(f"  {range_name} (n={metrics['count']}): "
                   f"MAE={metrics['mae']:.6f}, "
                   f"Within 25%={metrics.get('within_25pct', 0):.1f}%, "
                   f"Bias Ratio={metrics.get('bias_ratio', 0):.2f}")
    
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
        
        # Print sensitivity at different specificity levels
        for spec_level, sens in metrics['sensitivity_at_specificity'].items():
            if isinstance(sens, dict):
                sens_val = sens.get('sensitivity', 0)
            else:
                sens_val = sens
            logger.info(f"  Sensitivity at {spec_level} specificity: {sens_val:.4f}")
        
        logger.info(f"  Average precision: {metrics['average_precision']:.4f}")
    
    # Log background statistics
    if all_bg_levels:
        bg_levels = np.concatenate(all_bg_levels)
        logger.info(f"\nBACKGROUND CORRECTION STATISTICS:")
        logger.info(f"  Mean Background: {np.mean(bg_levels):.6f}")
        logger.info(f"  Median Background: {np.median(bg_levels):.6f}")
        logger.info(f"  Min Background: {np.min(bg_levels):.6f}")
        logger.info(f"  Max Background: {np.max(bg_levels):.6f}")
        logger.info(f"  Background Range: {model.bg_correction.min_bg:.6f} - {model.bg_correction.max_bg:.6f}")
    
    # Analyze marker importance
    logger.info(f"\nANALYZING MARKER IMPORTANCE:")
    top_indices, top_weights, stratified_importance = analyser.get_marker_importance(
        data_loader, stratify_by_concentration=True
    )
    
    # Print top markers
    logger.info(f"Top 10 markers by importance:")
    for i, (idx, weight) in enumerate(zip(top_indices[:10], top_weights[:10])):
        logger.info(f"  #{i+1}: Marker {idx} (weight: {weight:.4f})")
    
    # Print stratified importance if available
    if stratified_importance:
        logger.info("\nMarker importance by concentration range:")
        for range_name, importance in stratified_importance.items():
            if importance['count'] >= 5:  # Only show if enough samples
                logger.info(f"  {range_name} (n={importance['count']}):")
                for i, (idx, weight) in enumerate(zip(importance['indices'][:5], importance['weights'][:5])):
                    logger.info(f"    #{i+1}: Marker {idx} (weight: {weight:.4f})")
    
    # Save results
    results = {
        'predictions': all_preds.flatten().tolist(),
        'targets': all_targets.flatten().tolist(),
        'lower_ci': all_lower_ci.flatten().tolist(),
        'upper_ci': all_upper_ci.flatten().tolist(),
        'uncertainties': all_uncertainties.flatten().tolist(),
        'metrics': {
            'r2': float(r2),
            'mae': float(mae),
            'in_ci_percentage': float(in_ci * 100),
            'ci_width': float(ci_width)
        },
        'concentration_metrics': concentration_metrics,
        'uncertainty_metrics': uncertainty_metrics,
        'detection_metrics': {
            str(float(k)): {
                'auc': float(v['auc']), 
                'sensitivity_at_specificity': v['sensitivity_at_specificity'],
                'average_precision': float(v['average_precision'])
            } for k, v in detection_metrics.items()
        },
        'marker_importance': {
            'top_indices': top_indices.tolist(),
            'top_weights': top_weights.tolist(),
            'stratified': stratified_importance
        }
    }
    
    # Add background stats if available
    if all_bg_levels:
        bg_levels = np.concatenate(all_bg_levels)
        results['background_stats'] = {
            'mean': float(np.mean(bg_levels)),
            'median': float(np.median(bg_levels)),
            'min': float(np.min(bg_levels)),
            'max': float(np.max(bg_levels)),
            'range_min': float(model.bg_correction.min_bg),
            'range_max': float(model.bg_correction.max_bg)
        }
    
    # Save to file
    results_file = os.path.join(args.output_dir, f'{split_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"{split_name} results saved to {results_file}")
    
    # Create visualizations
    create_evaluation_visualizations(
        all_preds, all_targets, all_lower_ci, all_upper_ci, 
        detection_metrics, concentration_metrics,
        os.path.join(args.output_dir, 'visualizations', split_name)
    )
    
    return results

def create_evaluation_visualizations(predictions, targets, lower_ci, upper_ci, 
                                    detection_metrics, concentration_metrics, output_dir):
    """
    Create comprehensive visualizations for evaluation results
    
    Args:
        predictions: Predicted concentrations
        targets: True concentrations
        lower_ci: Lower confidence interval bounds
        upper_ci: Upper confidence interval bounds
        detection_metrics: Detection metrics from evaluation
        concentration_metrics: Concentration metrics from evaluation
        output_dir: Output directory for visualizations
    """
    try:
        import os
        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np
        import pandas as pd
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'true_value': targets.flatten(),
            'predicted_value': predictions.flatten(),
            'lower_ci': lower_ci.flatten(),
            'upper_ci': upper_ci.flatten(),
            'error': predictions.flatten() - targets.flatten(),
            'abs_error': np.abs(predictions.flatten() - targets.flatten()),
        })
        
        # Add relative error
        rel_error = np.full_like(targets.flatten(), np.nan, dtype=float)
        non_zero_mask = targets.flatten() > 0
        rel_error[non_zero_mask] = np.abs(predictions.flatten()[non_zero_mask] - targets.flatten()[non_zero_mask]) / targets.flatten()[non_zero_mask] * 100
        df['rel_error'] = rel_error
        
        # Add concentration range
        ranges = [
            (0, 0.0005, "0-0.05%"),
            (0.0005, 0.001, "0.05-0.1%"),
            (0.001, 0.005, "0.1-0.5%"),
            (0.005, 0.01, "0.5-1%"),
            (0.01, 0.05, "1-5%"),
            (0.05, 0.1, "5-10%"),
            (0.1, 1.0, ">10%")
        ]
        
        df['range'] = 'Unknown'
        for low, high, name in ranges:
            mask = (df['true_value'] >= low) & (df['true_value'] < high)
            df.loc[mask, 'range'] = name
        
        # Create scatter plot
        fig = px.scatter(
            df, 
            x='true_value', 
            y='predicted_value',
            color='range',
            log_x=True, 
            log_y=True,
            opacity=0.7,
            hover_data=['abs_error', 'rel_error', 'lower_ci', 'upper_ci']
        )
        
        # Add identity line
        fig.add_trace(
            go.Scatter(
                x=[1e-6, 1],
                y=[1e-6, 1],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect prediction'
            )
        )
        
        # Add R² annotation
        r2 = np.corrcoef(df['true_value'], df['predicted_value'])[0, 1]**2
        mae = np.mean(df['abs_error'])
        
        fig.add_annotation(
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
        fig.update_layout(
            title='Predicted vs True Concentration (Log Scale)',
            xaxis_title='True Concentration',
            yaxis_title='Predicted Concentration',
            template='plotly_white',
            width=900,
            height=700
        )
        
        # Format axes as percentages
        fig.update_xaxes(tickformat='.2%')
        fig.update_yaxes(tickformat='.2%')
        
        # Save figure
        fig.write_html(os.path.join(output_dir, 'scatter_plot_log.html'))
        fig.write_image(os.path.join(output_dir, 'scatter_plot_log.png'), scale=2)
        
        # Create stratified metrics plot
        if concentration_metrics and 'stratified_metrics' in concentration_metrics:
            # Create figure
            fig_stratified = go.Figure()
            
            # Prepare data
            ranges = []
            mae_values = []
            counts = []
            within_25pct_values = []
            
            for range_name, metrics in concentration_metrics['stratified_metrics'].items():
                ranges.append(range_name)
                mae_values.append(metrics['mae'])
                counts.append(metrics['count'])
                within_25pct_values.append(metrics.get('within_25pct', 0))
            
            # Add MAE bars
            fig_stratified.add_trace(
                go.Bar(
                    x=ranges,
                    y=mae_values,
                    name='MAE',
                    text=[f"n={count}" for count in counts],
                    marker_color='blue',
                    opacity=0.7
                )
            )
            
            # Update layout
            fig_stratified.update_layout(
                title='Stratified Mean Absolute Error by Concentration Range',
                xaxis_title='Concentration Range',
                yaxis_title='MAE',
                template='plotly_white',
                width=900,
                height=600
            )
            
            # Save figure
            fig_stratified.write_html(os.path.join(output_dir, 'stratified_mae.html'))
            fig_stratified.write_image(os.path.join(output_dir, 'stratified_mae.png'), scale=2)
            
            # Create within 25% accuracy plot
            fig_accuracy = go.Figure()
            
            # Add bars
            fig_accuracy.add_trace(
                go.Bar(
                    x=ranges,
                    y=within_25pct_values,
                    name='Within 25% Accuracy',
                    text=[f"n={count}" for count in counts],
                    marker_color='green',
                    opacity=0.7
                )
            )
            
            # Add reference line at 50% and 70%
            for target in [50, 70]:
                fig_accuracy.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=target,
                    x1=len(ranges) - 0.5,
                    y1=target,
                    line=dict(color="red" if target == 70 else "orange", dash="dash")
                )
            
            # Update layout
            fig_accuracy.update_layout(
                title='Within 25% Accuracy by Concentration Range',
                xaxis_title='Concentration Range',
                yaxis_title='Percentage within 25% error',
                template='plotly_white',
                width=900,
                height=600,
                yaxis=dict(range=[0, 100])
            )
            
            # Save figure
            fig_accuracy.write_html(os.path.join(output_dir, 'within_25pct_accuracy.html'))
            fig_accuracy.write_image(os.path.join(output_dir, 'within_25pct_accuracy.png'), scale=2)
    
    except Exception as e:
        print(f"Error creating evaluation visualizations: {e}")

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
# qrsh -b y -l h_vmem=2g -pe smp 32 -V -N train_oac -wd /users/zetzioni/sharedscratch/deepconv/src -o ~/sharedscratch/logs/train_oac.log 'cd /users/zetzioni/sharedscratch/deepconv/src && python -m deep_conv.detect.train --name oac_conc_focused --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell --data_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train_single_cell_clinical/OAC/ --atlas_path /users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed --target_cell_type OAC --target_cell_idx 9 --snr_profile high --dropout_rate 0.15 --feature_dim 128 --detection_thresholds 0.001,0.01 --critical_ranges "0.001,0.005,1.0;0.005,0.01,1.0" --detection_loss_weight 0.03 --min_reliable_coverage 5.0 --control_data_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/ --calibrate --calibrate_every 15 --early_stopping 30 --excluded_markers "44,58,111,133,77,95,127,38,108,115" --epochs 200 --weight_decay 0.01'

def main():
    """
    Main function for training and evaluating the improved cancer detection model
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
    logger.info(f"Starting improved cancer detection training pipeline")
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
    
    # Initialize improved model
    logger.info(f"Initializing improved model with {num_markers} markers...")
    try:
        model = EnhancedCancerDetectionModel(
            num_markers=num_markers,
            feature_dim=args.feature_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            detection_thresholds=args.detection_thresholds,
            min_reliable_coverage=args.min_reliable_coverage,
            marker_specific_bg=args.marker_specific_bg,
            enable_adaptive_thresholds=args.enable_adaptive_thresholds,
            critical_ranges=args.critical_ranges,
            snr_profile=args.snr_profile
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✓ Improved model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
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
            calibration_results = model.calibrate(
                val_loader, 
                control_val_loader, 
                device
            )
            # Update best model state with calibration results
            if best_model_state is not None:
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
            mu, uncertainty, _, attention_weights = model(marker_values, coverage)
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
    
    # Create visualization directory for visualise_results output
    vis_dir = os.path.join(args.output_dir, 'visualizations', 'test_legacy')
    
    # Call visualise_results
    visualise_results(
        test_preds, 
        test_targets, 
        vis_dir,
        ci_data=(test_lower_ci, test_upper_ci),
        marker_importance=test_attentions.mean(axis=0),
        prefix="Test "
    )
    
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
