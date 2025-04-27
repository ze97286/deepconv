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

from deep_conv.detect.preprocess import prepare_data_for_training,load_train_with_contrastive_data
from deep_conv.detect.model import EnhancedCancerDetectionModel, MarkerImportanceAnalyser, CancerDetectionEnsemble


def parse_args():
    parser = argparse.ArgumentParser(description='Train cfDNA methylation cancer detection model')
    
    parser.add_argument('--name', type=str, default=None, help='Name for this training run (used for output directory)')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default="/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval_single_cell_clinical/OAC/", help='Directory containing parquet files')
    parser.add_argument('--atlas_path', type=str, default="/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed", help='Path to atlas file')
    parser.add_argument('--target_cell_type', type=str, default='OAC', help='Target cell type')
    parser.add_argument('--target_cell_idx', type=int, default=9, help='Target cell index in ground truth')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers (ignored for Set Transformer)')
    
    parser.add_argument('--cell_profile', type=str, default=None, 
                       choices=['default', 'high_snr', 'low_snr', 'ultra_low_snr'],
                       help='Predefined optimisation profile for different cell types')
    parser.add_argument('--detection_loss_weight', type=float, default=None, 
                       help='Weight of detection loss relative to concentration loss')
    parser.add_argument('--focal_weight_factor', type=float, default=None,
                       help='Factor for focal weighting of low concentration samples')
    parser.add_argument('--low_concentration_threshold', type=float, default=None,
                       help='Threshold defining low concentration samples for special handling')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimiser')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default="/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell", help='Output directory')
    
    # Ensemble parameters
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of models')
    parser.add_argument('--ensemble_size', type=int, default=3, help='Number of models in ensemble')
    parser.add_argument('--ensemble_seeds', type=str, default=None, help='Comma-separated seeds for ensemble models')
    
    # Evaluation parameters
    parser.add_argument('--detection_thresholds', type=str, default="0.001,0.01,0.05", help='Comma-separated detection thresholds')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save checkpoint every N epochs')
    
    parser.add_argument('--control_data_dir', type=str, default=None, 
                   help='Directory containing control data for contrastive learning')
    parser.add_argument('--calibrate', action='store_true', 
                    help='Calibrate confidence intervals and background correction')
    parser.add_argument('--contrastive_weight', type=float, default=5.0,
                    help='Weight for contrastive loss component')

    args = parser.parse_args()
    
    if args.name:
        dir_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"run_{timestamp}"
    
    args.output_dir = os.path.join(args.output_dir, dir_name)

    # Process detection thresholds
    args.detection_thresholds = [float(x) for x in args.detection_thresholds.split(',')]
    
    # Process ensemble seeds if provided
    if args.ensemble and args.ensemble_seeds:
        args.ensemble_seeds = [int(x) for x in args.ensemble_seeds.split(',')]
    elif args.ensemble:
        # Generate random seeds if not provided
        base_seed = args.seed
        args.ensemble_seeds = [base_seed + i for i in range(args.ensemble_size)]
    
    if args.cell_profile:
        apply_cell_profile(args)

    return args

def apply_cell_profile(args):
    """Apply predefined parameter sets optimised for different cell types"""
    profiles = {
        'default': {
            # Default parameters, good for most cell types
            'detection_loss_weight': 0.2,
            'focal_weight_factor': 100,
            'low_concentration_threshold': 0.01
        },
        'high_snr': {  # For cells like OAC with good SNR
            'detection_loss_weight': 0.2,
            'focal_weight_factor': 100, 
            'low_concentration_threshold': 0.01
        },
        'low_snr': {  # For cells with moderate SNR issues
            'detection_loss_weight': 0.4,
            'focal_weight_factor': 150,
            'low_concentration_threshold': 0.02
        },
        'ultra_low_snr': {  # For T-cells and other very low SNR cases
            'detection_loss_weight': 0.6,
            'focal_weight_factor': 200,
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


def calculate_loss(model, mu, phi, detection_probs, y_true, args, control_mask=None):
    # Get concentration loss
    concentration_loss = model.compute_loss(mu, phi, y_true)
    
    # Add simple control penalty if controls present
    if control_mask is not None and control_mask.sum() > 0:
        # Extract predictions for control samples
        control_preds = mu[control_mask]
        
        # Simple L1 penalty for any prediction above minimal threshold on controls
        control_penalty = 3.0 * torch.mean(control_preds)
        concentration_loss = concentration_loss + control_penalty
    
    # Calculate detection losses (keep this part unchanged)
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


def coverage_weighted_loss(model, mu, phi, detection_probs, y_true, coverage):
    """
    Weight the loss by coverage to reduce the impact of low-coverage markers
    """
    # Standard loss component
    standard_loss = model.compute_loss(mu, phi, y_true)
    
    # Calculate weights based on coverage
    # Sigmoid function to smoothly transition from low to high weight
    # as coverage increases
    weights = 2.0 / (1.0 + torch.exp(-0.2 * (coverage.mean(dim=1, keepdim=True) - 5.0)))
    
    # Apply weights to loss
    weighted_loss = standard_loss * weights.mean()
    
    # Add a small regularization to maintain overall scale
    reg_loss = 0.03 * torch.abs(torch.log(phi)).mean()
    
    return weighted_loss + reg_loss

def train(model, train_loader, val_loader, args, device):
    """Train the model with progress bars and enhanced logging"""
    os.makedirs(args.output_dir, exist_ok=True)
    logger = logging.getLogger('cancer_detection')
    
    git_info = get_git_info()
    git_commit = git_info['commit']

    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() 
    
    # Apply cell-specific parameters to model if provided
    if hasattr(args, 'focal_weight_factor'):
        model.focal_weight_factor = args.focal_weight_factor
    if hasattr(args, 'low_concentration_threshold'):
        model.low_concentration_threshold = args.low_concentration_threshold
    
    logger.info(f"Starting training with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'concentration_loss': [],
        'detection_loss': [],
        'calibration_error': [],
        'r2_score': [],
        'mean_absolute_error': [],
        'lr': [],
        'detection_metrics': []
    }
    
    epoch_bar = tqdm(range(args.epochs), desc="Training", position=0)
    for epoch in epoch_bar:
        # Training phase
        model.train()
        train_loss = 0
        train_conc_loss = 0
        train_det_loss = 0
        optimiser.zero_grad()
        
        batch_bar = tqdm(enumerate(train_loader), 
                         desc=f"Epoch {epoch+1}/{args.epochs} [Train]", 
                         total=len(train_loader),
                         position=1, 
                         leave=False)
        
        for i, batch_data in batch_bar:
            # Handle both dataset types (with or without control_mask)
            if len(batch_data) == 4:  # Dataset includes control_mask
                marker_values, coverage, y_true, control_mask = batch_data
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                control_mask = control_mask.to(device)
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    output = model(marker_values, coverage, y_true, control_mask)
                    mu, phi, detection_probs, attention_weights, _, _ = output
                    
                    # Calculate loss with contrastive component
                    loss, conc_loss, det_loss = calculate_loss(
                        model, mu, phi, detection_probs, y_true, args, 
                        control_mask=control_mask, epoch=epoch
                    )
                    loss = loss / args.grad_accum_steps
            else:  # Standard dataset without control_mask
                marker_values, coverage, y_true = batch_data
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)

                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    # Use updated forward pass that returns components instead of loss
                    output = model(marker_values, coverage, y_true)
                    
                    if len(output) == 5:  # Enhanced model returns 5 values
                        mu, phi, detection_probs, attention_weights, _ = output
                        # Calculate loss with parameter-based function
                        loss, conc_loss, det_loss = calculate_loss(
                            model, mu, phi, detection_probs, y_true, args
                        )
                    else:  # Standard model returns 4 values
                        # Fallback for backward compatibility
                        mu, phi, loss, attention_weights = output
                        conc_loss = loss
                        det_loss = 0.0
                    
                    loss = loss / args.grad_accum_steps

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            batch_loss = loss.item() * args.grad_accum_steps
            train_loss += batch_loss
            train_conc_loss += conc_loss.item() / args.grad_accum_steps
            train_det_loss += det_loss.item() / args.grad_accum_steps
            
            batch_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
            
            # Gradient accumulation and optimisation step
            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_conc_loss = 0
        val_det_loss = 0
        calibration_error = 0
        all_preds = []
        all_targets = []
        
        # Use tqdm for validation
        val_bar = tqdm(val_loader, 
                      desc=f"Epoch {epoch+1}/{args.epochs} [Validate]", 
                      position=1, 
                      leave=False)
        
        with torch.no_grad():
            for batch_data in val_bar:
                if len(batch_data) == 4:  # Dataset includes control_mask
                    marker_values, coverage, y_true, _ = batch_data  # Ignore control_mask for validation
                else:
                    marker_values, coverage, y_true = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                # Handle different model return signatures
                output = model(marker_values, coverage, y_true)
                
                if len(output) == 5:  # Enhanced model returns 5 values
                    mu, phi, detection_probs, attention_weights, _ = output
                    # Calculate loss with parameter-based function
                    loss, conc_loss, det_loss = calculate_loss(
                        model, mu, phi, detection_probs, y_true, args
                    )
                else:  # Standard model returns 4 values
                    # Fallback for backward compatibility
                    mu, phi, loss, attention_weights = output
                    conc_loss = loss
                    det_loss = 0.0
                
                val_loss += loss.item()
                val_conc_loss += conc_loss.item()
                val_det_loss += det_loss.item()
                
                # Store predictions and targets for metrics
                all_preds.append(mu.cpu().numpy())
                all_targets.append(y_true.cpu().numpy())
                
                # Calculate calibration error
                estimate, ci, _ = model.get_estimate_and_ci(mu, phi)
                in_ci = (y_true >= ci[:, 0:1]) & (y_true <= ci[:, 1:2])
                calibration_error += (1.0 - in_ci.float().mean()).item()
                
                # Update validation progress bar
                val_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate metrics
        val_loss /= len(val_loader)
        val_conc_loss /= len(val_loader)
        val_det_loss /= len(val_loader)
        calibration_error /= len(val_loader)
        train_loss /= len(train_loader)
        train_conc_loss /= len(train_loader)
        train_det_loss /= len(train_loader)
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        
        # Get current learning rate
        current_lr = optimiser.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step()
        
        # Calculate detection metrics
        analyser = MarkerImportanceAnalyser(model)
        detection_metrics = analyser.analyse_detection_performance(val_loader, thresholds=args.detection_thresholds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['concentration_loss'].append(val_conc_loss)
        history['detection_loss'].append(val_det_loss)
        history['calibration_error'].append(calibration_error)
        history['r2_score'].append(r2)
        history['mean_absolute_error'].append(mae)
        history['lr'].append(current_lr)
        history['detection_metrics'].append(detection_metrics)
        
        # Update epoch progress bar
        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "R²": f"{r2:.4f}"
        })
        
        # Log performance metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, "
            f"Conc Loss: {val_conc_loss:.6f}, "
            f"Det Loss: {val_det_loss:.6f}, "
            f"Calibration Error: {calibration_error:.4f}, "
            f"R² Score: {r2:.4f}, "
            f"MAE: {mae:.6f}, "
            f"LR: {current_lr:.2e}"
        )
        
        # Log detection metrics for 1% threshold
        key_threshold = 0.01  # 1% threshold is often clinically relevant
        if key_threshold in detection_metrics:
            key_metrics = detection_metrics[key_threshold]
            logger.info(
                f"Detection at {key_threshold:.1%}: "
                f"AUC={key_metrics['auc']:.4f}, "
                f"Sensitivity@95%Spec={key_metrics['sensitivity_at_95spec']:.4f}"
            )
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            improvement = "inf" if best_val_loss == float('inf') else f"{(best_val_loss - val_loss) / best_val_loss * 100:.2f}%"
            best_val_loss = val_loss
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'concentration_loss': val_conc_loss,
                'detection_loss': val_det_loss,
                'r2_score': r2,
                'mae': mae,
                'calibration_error': calibration_error,
                'detection_metrics': detection_metrics,
                'commit-hash': git_commit,
            }
            torch.save(best_model_state, os.path.join(args.output_dir, 'best_model.pt'))
            patience_counter = 0
            logger.info(f"✓ New best model saved! Improvement: {improvement}")
        else:
            patience_counter += 1
            logger.info(f"× No improvement. Patience: {patience_counter}/{args.early_stopping}")
            
        # Early stopping check
        if patience_counter >= args.early_stopping:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimiser': optimiser.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'history': history,
                'args': vars(args),
                'commit-hash': git_commit,
            }
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'concentration_loss': val_conc_loss,
        'detection_loss': val_det_loss,
        'r2_score': r2,
        'mae': mae,
        'calibration_error': calibration_error,
        'commit-hash': git_commit
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training history with improved error handling
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        serialisable_history = {}
        for key, values in history.items():
            if key != 'detection_metrics':
                serialisable_history[key] = [float(v) for v in values]
            else:
                # Handle nested dictionaries for detection metrics
                serialisable_detection_metrics = []
                for epoch_metrics in values:
                    serialisable_epoch_metrics = {}
                    for thresh, metrics in epoch_metrics.items():
                        if isinstance(thresh, dict):
                            # If thresh is already a dict, something is wrong with the data structure
                            print(f"Warning: Unexpected dict as threshold key: {thresh}")
                            # Use a string representation as a fallback
                            thresh_key = str(thresh)
                        else:
                            # Normal case: thresh should be a number
                            thresh_key = str(float(thresh))
                        
                        # Similar safeguard for metrics
                        if isinstance(metrics, dict):
                            # Convert all metric values to float
                            serialisable_metrics = {k: float(v) if not isinstance(v, dict) else str(v) for k, v in metrics.items()}
                        else:
                            # If metrics is not a dict (unexpected), store as string
                            serialisable_metrics = {"value": str(metrics)}
                        
                        serialisable_epoch_metrics[thresh_key] = serialisable_metrics
                    serialisable_detection_metrics.append(serialisable_epoch_metrics)
                serialisable_history[key] = serialisable_detection_metrics
        json.dump(serialisable_history, f)
    
    logger.info(f"Training history saved to {history_path}")
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    logger.info(f"Training plots saved to {args.output_dir}")
    
    # Final performance summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_model_state['epoch']+1})")
    logger.info(f"Best R² score: {best_model_state['r2_score']:.4f}")
    logger.info(f"Best MAE: {best_model_state['mae']:.6f}")
    logger.info("="*50 + "\n")
    
    # Load best model
    model.load_state_dict(best_model_state['model'])
    return model, best_model_state

def plot_training_history(history, output_dir):
    """
    Plot and save training history with enhanced visualisations using Plotly
    """
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create figure with subplots
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
    
    # Get epochs
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # 1. Training and validation loss
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['train_loss'], 
            mode='lines', 
            name='Train Loss',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['val_loss'], 
            mode='lines', 
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
            mode='lines', 
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
            mode='lines', 
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
            mode='lines', 
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
                mode='lines', 
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
    
    # Update layout
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
    fig.update_yaxes(title_text='R²', row=2, col=1)
    fig.update_yaxes(title_text='MAE', row=2, col=2)
    fig.update_yaxes(title_text='Learning Rate', row=3, col=1)
    fig.update_yaxes(title_text='R² Score', row=3, col=2)
    
    # Save main figure
    fig.write_html(os.path.join(plots_dir, 'training_history.html'))
    fig.write_image(os.path.join(plots_dir, 'training_history.png'), scale=2)
    
    # Also create individual plots for better detail
    metrics = [
        ('loss', ['train_loss', 'val_loss'], ['Train Loss', 'Validation Loss'], ['blue', 'red']),
        ('r2_score', ['r2_score'], ['R² Score'], ['purple']),
        ('mae', ['mean_absolute_error'], ['Mean Absolute Error'], ['orange']),
        ('calibration', ['calibration_error'], ['Calibration Error'], ['green'])
    ]
    
    for name, keys, labels, colors in metrics:
        fig = go.Figure()
        
        for key, label, color in zip(keys, labels, colors):
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history[key],
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2)
                )
            )
        
        fig.update_layout(
            title=f'{labels[0]}' if len(labels) == 1 else 'Loss Curves',
            xaxis_title='Epoch',
            yaxis_title=name.replace('_', ' ').title(),
            template='plotly_white',
            width=800,
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Save individual figure
        fig.write_html(os.path.join(plots_dir, f'{name}_history.html'))
        fig.write_image(os.path.join(plots_dir, f'{name}_history.png'), scale=2)


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
    """Unified evaluation function for both validation and test sets"""
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
            
            # Handle different return signatures
            if isinstance(model, CancerDetectionEnsemble):
                mu, phi, det_probs = model.forward(marker_values, coverage)
                estimate, ci, uncertainty = model.get_estimate_and_ci(marker_values, coverage)
                # Use first model's attention weights for analysis
                _, _, _, attention_weights = model.models[0](marker_values, coverage)
            else:
                mu, phi, det_probs, attention_weights = model(marker_values, coverage)
                estimate, ci, uncertainty = model.get_estimate_and_ci(mu, phi)
            
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
    
    # Log basic results
    logger.info(f"\n{split_name.upper()} RESULTS:")
    logger.info(f"Number of {split_name} samples: {len(all_targets)}")
    logger.info(f"R² Score: {r2:.6f}")
    logger.info(f"Mean Absolute Error: {mae:.6f}")
    logger.info(f"Targets within CI: {in_ci * 100:.2f}%")
    logger.info(f"Average CI Width: {ci_width:.6f}")
    
    # Calculate detection metrics
    analyser = MarkerImportanceAnalyser(model)
    detection_metrics = analyser.analyse_detection_performance(data_loader, thresholds=args.detection_thresholds)
    
    # Log detection metrics
    logger.info(f"\n{split_name.upper()} DETECTION METRICS:")
    for threshold, metrics in detection_metrics.items():
        logger.info(f"At {threshold:.3%} threshold:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Sensitivity at 95% specificity: {metrics['sensitivity_at_95spec']:.4f}")
        logger.info(f"  Average precision: {metrics['average_precision']:.4f}")
    
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
        'detection_metrics': {
            str(float(k)): {
                'auc': float(v['auc']), 
                'sensitivity_at_95spec': float(v['sensitivity_at_95spec']),
                'average_precision': float(v['average_precision'])
            } for k, v in detection_metrics.items()
        }
    }
    
    # Save to file
    results_file = os.path.join(args.output_dir, f'{split_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"{split_name} results saved to {results_file}")
    
    # Create visualisations
    vis_dir = os.path.join(args.output_dir, 'visualisations')
    split_vis_dir = os.path.join(vis_dir, split_name)
    
    visualise_results(
        predictions=all_preds,
        ground_truth=all_targets,
        output_subdir=split_vis_dir,
        ci_data=(all_lower_ci, all_upper_ci),
        marker_importance=all_marker_attentions.mean(axis=0),
        prefix=f"{split_name.capitalize()} "
    )
    
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
    
    # 1. Calibrate confidence intervals
    best_factor = 1.0
    best_error = float('inf')
    best_low_factor = 1.0
    
    with torch.no_grad():
        # Test different calibration factors
        for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
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
                
                mu, phi, _, _ = model(marker_values, coverage)
                
                # Apply test calibration factor
                phi_calibrated = phi * factor
                
                # Calculate alpha, beta parameters
                alpha = mu * phi_calibrated
                beta = (1 - mu) * phi_calibrated
                
                # Move to numpy for scipy operations
                alpha_np = alpha.cpu().numpy()
                beta_np = beta.cpu().numpy()
                y_true_np = y_true.cpu().numpy()
                
                # Calculate 95% CI using scipy.stats.beta
                lower = np.zeros_like(y_true_np)
                upper = np.zeros_like(y_true_np)
                
                for i in range(len(alpha_np)):
                    a, b = float(alpha_np[i]), float(beta_np[i])
                    if a > 0 and b > 0:
                        try:
                            lower[i] = stats.beta.ppf(0.025, a, b)
                            upper[i] = stats.beta.ppf(0.975, a, b)
                        except:
                            # In case of numerical issues, use fallbacks
                            lower[i] = max(0.0, mu[i].item() - 2.0 * (1.0 / np.sqrt(phi_calibrated[i].item())))
                            upper[i] = min(1.0, mu[i].item() + 2.0 * (1.0 / np.sqrt(phi_calibrated[i].item())))
                
                # Calculate CI coverage
                in_ci = (y_true_np >= lower) & (y_true_np <= upper)
                ci_coverage = in_ci.mean()
                
                # Error relative to target 95%
                error = abs(ci_coverage - 0.95)
                coverage_error += error
                n_batches += 1
            
            avg_error = coverage_error / n_batches
            if avg_error < best_error:
                best_error = avg_error
                best_factor = factor
        
        # Calibration factor for low concentrations (separate calibration)
        best_low_error = float('inf')
        # Try different calibration factors for low concentrations
        for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
            coverage_error = 0
            n_batches = 0
            
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    marker_values, coverage, y_true, _ = batch_data
                else:
                    marker_values, coverage, y_true = batch_data
                
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                # Select only low concentration samples
                if hasattr(model, 'low_concentration_threshold'):
                    low_thresh = model.low_concentration_threshold
                else:
                    low_thresh = 0.01  # Default
                
                low_mask = y_true <= low_thresh
                if low_mask.sum() == 0:
                    continue  # Skip if no low concentration samples
                
                # Process only low concentration samples
                mu_low = mu[low_mask]
                phi_low = phi[low_mask]
                y_true_low = y_true[low_mask]
                
                # Apply test calibration factor
                phi_calibrated = phi_low * factor
                
                # Calculate alpha, beta parameters
                alpha = mu_low * phi_calibrated
                beta = (1 - mu_low) * phi_calibrated
                
                # Move to numpy for scipy operations
                alpha_np = alpha.cpu().numpy()
                beta_np = beta.cpu().numpy()
                y_true_np = y_true_low.cpu().numpy()
                
                # Calculate 95% CI
                lower = np.zeros_like(y_true_np)
                upper = np.zeros_like(y_true_np)
                
                for i in range(len(alpha_np)):
                    a, b = float(alpha_np[i]), float(beta_np[i])
                    if a > 0 and b > 0:
                        try:
                            lower[i] = stats.beta.ppf(0.025, a, b)
                            upper[i] = stats.beta.ppf(0.975, a, b)
                        except:
                            # Fallback on error
                            lower[i] = max(0.0, mu_low[i].item() - 2.0 * (1.0 / np.sqrt(phi_calibrated[i].item())))
                            upper[i] = min(1.0, mu_low[i].item() + 2.0 * (1.0 / np.sqrt(phi_calibrated[i].item())))
                
                # Calculate CI coverage for low concentrations
                in_ci = (y_true_np >= lower) & (y_true_np <= upper)
                ci_coverage = in_ci.mean()
                
                # Error relative to target 95%
                error = abs(ci_coverage - 0.95)
                coverage_error += error
                n_batches += 1
            
            if n_batches > 0:
                avg_error = coverage_error / n_batches
                if avg_error < best_low_error:
                    best_low_error = avg_error
                    best_low_factor = factor
    
    # 2. Calculate background level from controls if available
    background_level = 0.0
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
        
        # Calculate median of predictions on controls
        all_preds = np.concatenate(all_preds)
        background_level = float(np.median(all_preds))
    
    # Apply calibration factors to model
    with torch.no_grad():
        # Update model parameters if they exist
        if hasattr(model, 'calibration'):
            model.calibration.fill_(best_factor)
        if hasattr(model, 'low_calibration'):
            model.low_calibration.fill_(best_low_factor)
        if hasattr(model, 'background_level'):
            model.background_level.fill_(background_level)
    
    # Return calibration parameters
    calibration_results = {
        'calibration_factor': best_factor,
        'low_calibration_factor': best_low_factor,
        'background_level': background_level,
        'coverage_error': float(best_error),
        'low_coverage_error': float(best_low_error) if best_low_error != float('inf') else None
    }
    
    return calibration_results

def train_ensemble(args, train_loader, val_loader, test_loader, num_markers, device):
    """Train an ensemble of models with different random seeds"""
    logger = logging.getLogger('cancer_detection')
    logger.info(f"Training ensemble of {args.ensemble_size} models...")

    models = []
    best_states = []

    # Create subdirectory for individual models
    ensemble_dir = os.path.join(args.output_dir, 'ensemble_models')
    os.makedirs(ensemble_dir, exist_ok=True)

    # Train each model with a different seed
    for i, seed in enumerate(args.ensemble_seeds):
        logger.info(f"\n{'='*20} TRAINING ENSEMBLE MODEL {i+1}/{args.ensemble_size} (SEED: {seed}) {'='*20}\n")

        # Set seed for this model
        set_seed(seed)

        # Create model
        model = EnhancedCancerDetectionModel(
            num_markers=num_markers,
            feature_dim=args.feature_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            detection_thresholds=args.detection_thresholds,
            focal_weight_factor=args.focal_weight_factor,
            low_concentration_threshold=args.low_concentration_threshold,
        )

        # Create model directory
        model_dir = os.path.join(ensemble_dir, f'model_{i+1}_seed_{seed}')
        os.makedirs(model_dir, exist_ok=True)

        # Store original output_dir
        original_output_dir = args.output_dir

        # Temporarily set output_dir to model directory
        args.output_dir = model_dir

        # Train model
        model, best_state = train(model, train_loader, val_loader, args, device)

        # Reset output_dir
        args.output_dir = original_output_dir

        # Store model and best state
        models.append(model)
        best_states.append(best_state)

        # Log model results
        logger.info(f"Model {i+1}/{args.ensemble_size} training complete")
        logger.info(f"Best validation loss: {best_state['val_loss']:.6f}")
        logger.info(f"Best R² score: {best_state['r2_score']:.4f}")

    # Create ensemble model
    ensemble = CancerDetectionEnsemble(models)

    logger.info("\n" + "="*50)
    logger.info(f"ENSEMBLE TRAINING COMPLETE ({args.ensemble_size} models)")

    # Save ensemble model
    ensemble_state = {
        "model_states": [model.state_dict() for model in models],
        "ensemble_size": args.ensemble_size,
        "seeds": args.ensemble_seeds,
        "model_config": {
            "num_markers": num_markers,
            "feature_dim": args.feature_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout_rate": args.dropout_rate,
            "focal_weight_factor": args.focal_weight_factor,
            "detection_thresholds": args.detection_thresholds,
            "low_concentration_threshold": args.low_concentration_threshold,
        },
    }

    ensemble_path = os.path.join(args.output_dir, 'ensemble_model.pt')
    torch.save(ensemble_state, ensemble_path)
    logger.info(f"Ensemble model saved to {ensemble_path}")

    return ensemble, ensemble_state


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


# T-cells
# python -m deep_conv.detect.train \
# --name CpGenie_T-cells \
# --data_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train_single_cell_clinical/T-cells/ \
# --target_cell_type T-cells \
# --target_cell_idx 11 \
# --grad_accum_steps 8 \
# --cell_profile ultra_low_snr
# --calibrate


# OAC
# python -m deep_conv.detect.train \
# --name CpGenie_OAC \
# --data_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train_single_cell_clinical/OAC/ \
# --target_cell_type OAC \
# --target_cell_idx 9 \
# --cell_profile high_snr \
# --control_data_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/ \
# --calibrate

# python -m deep_conv.detect.train --ensemble --ensemble_size=3 --detection_thresholds=0.001,0.01,0.05 --name CpGenie_ensemble
def main():
    """Main function with enhanced logging and progress tracking"""
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
    logger.info(f"Starting cancer detection training pipeline")
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Save arguments
    args_file = os.path.join(args.output_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Arguments saved to {args_file}")
    
    # Prepare data
    logger.info(f"Preparing data from {args.data_dir}...")
    try:
        train_loader, val_loader, test_loader, num_markers = prepare_data_for_training(
            data_dir=args.data_dir,
            atlas_path=args.atlas_path,
            target_cell_type=args.target_cell_type,
            target_cell_idx=args.target_cell_idx,
        )
        logger.info(f"✓ Data preparation complete")
    except Exception as e:
        logger.error(f"× Error during data preparation: {str(e)}")
        raise
    
    # Save data stats
    # stats_file = os.path.join(args.output_dir, 'data_stats.json')
    # with open(stats_file, 'w') as f:
    #     # Convert numpy types to Python types for JSON serialisation
    #     serialisable_stats = {k: float(v) for k, v in data_stats.items()}
    #     json.dump(serialisable_stats, f, indent=2)
    # logger.info(f"Data statistics saved to {stats_file}")
    
    # Load control data if provided
    control_val_loader = None
    if hasattr(args, 'control_data_dir') and args.control_data_dir:
        logger.info(f"Loading control data from {args.control_data_dir}...")
        train_loader, control_val_loader = load_train_with_contrastive_data(train_loader, args.control_data_dir, args.atlas_path, args.target_cell_type, args.batch_size, logger)
    
    # Training phase
    if args.ensemble:
        logger.info(f"Training ensemble of {args.ensemble_size} models...")
        model, best_model_state = train_ensemble(
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_markers=num_markers,
            device=device
        )
    else:
        # Initialize single model
        logger.info(f"Initializing model with {num_markers} markers...")
        try:
            model = EnhancedCancerDetectionModel(
                num_markers=num_markers,
                feature_dim=args.feature_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout_rate=args.dropout_rate,
                detection_thresholds=args.detection_thresholds,
                focal_weight_factor=args.focal_weight_factor,
                low_concentration_threshold=args.low_concentration_threshold,
            )
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"✓ Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
        except Exception as e:
            logger.error(f"× Error initializing model: {str(e)}")
            raise
        
        # Train model
        logger.info("Starting model training...")
        try:
            model, best_model_state = train(model, train_loader, val_loader, args, device)
            logger.info(f"✓ Training completed successfully")
        except Exception as e:
            logger.error(f"× Error during training: {str(e)}")
            raise
        
        if hasattr(args, 'calibrate') and args.calibrate:
            logger.info("Calibrating model confidence intervals and background level...")
            try:
                calibration_results = calibrate_model(
                    model, val_loader, control_val_loader, device
                )
                best_model_state['calibration'] = calibration_results
                torch.save(best_model_state, os.path.join(args.output_dir, 'best_model.pt'))
                logger.info(f"✓ Model calibrated: factor={calibration_results['calibration_factor']:.4f}, {calibration_results['low_calibration_factor']:.4f}, background={calibration_results['background_level']:.6f}")
                logger.info(f"Calibration results saved with best model checkpoint")
            except Exception as e:
                logger.error(f"× Error during calibration: {str(e)}")
                logger.info("  Continuing without calibration")
    
    # Evaluation phase
    try:
        # Validate final model
        logger.info("Evaluating model on validation set...")
        val_results = evaluate(model, val_loader, args, device, split_name="validation")
        
        # Test final model
        logger.info("Evaluating model on test set...")
        test_results = evaluate(model, test_loader, args, device, split_name="test")
        
        # Print summary
        if args.ensemble:
            logger.info("\n" + "="*60)
            logger.info("ENSEMBLE EVALUATION COMPLETE")
        else:
            logger.info("\n" + "="*60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Best validation loss: {best_model_state['val_loss']:.6f}")
            logger.info(f"Best validation R²: {best_model_state['r2_score']:.4f}")
            
        logger.info(f"Validation R²: {val_results['metrics']['r2']:.4f}")
        logger.info(f"Validation MAE: {val_results['metrics']['mae']:.6f}")
        logger.info(f"Test R²: {test_results['metrics']['r2']:.4f}")
        logger.info(f"Test MAE: {test_results['metrics']['mae']:.6f}")
        
        # Log detection metrics
        key_threshold = 0.01  # 1% is often clinical threshold
        if str(float(key_threshold)) in test_results['detection_metrics']:
            metrics = test_results['detection_metrics'][str(float(key_threshold))]
            logger.info(f"Test detection at {key_threshold:.1%}:")
            logger.info(f"  AUC: {metrics['auc']:.4f}")
            logger.info(f"  Sensitivity@95%Spec: {metrics['sensitivity_at_95spec']:.4f}")
        
        logger.info(f"All results saved to: {args.output_dir}")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"× Error during evaluation: {str(e)}")
        logger.exception(e)
    
    # Return success
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
