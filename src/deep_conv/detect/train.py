import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import json
import logging
from datetime import datetime
from tqdm import tqdm

# Import our modules
from deep_conv.detect.preprocess import prepare_data_for_training
from deep_conv.detect.model import EnhancedCancerDetectionModel, MarkerImportanceAnalyzer



def parse_args():
    parser = argparse.ArgumentParser(description='Train cfDNA methylation cancer detection model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default="/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval_single_cell_clinical/OAC/", help='Directory containing parquet files')
    parser.add_argument('--atlas_path', type=str, default="/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed",help='Path to atlas file')
    parser.add_argument('--target_cell_type', type=str, default='OAC', help='Target cell type')
    parser.add_argument('--target_cell_idx', type=int, default=9, help='Target cell index in ground truth')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--use_pos_encoding', action='store_true', help='Use positional encoding')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default="/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell_oac", help='Output directory')
    
    # Misc parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    
    return parser.parse_args()


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
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def train(model, train_loader, val_loader, args, device):
    """Train the model with progress bars and enhanced logging"""
    # Create output directory and setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    # Log training configuration
    logger.info(f"Starting training with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'calibration_error': [],
        'r2_score': [],
        'mean_absolute_error': [],
        'lr': []
    }
    
    # Initialize tqdm for epochs
    epoch_bar = tqdm(range(args.epochs), desc="Training", position=0)
    
    for epoch in epoch_bar:
        # Training phase
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        # Use tqdm for batches
        batch_bar = tqdm(enumerate(train_loader), 
                         desc=f"Epoch {epoch+1}/{args.epochs} [Train]", 
                         total=len(train_loader),
                         position=1, 
                         leave=False)
        
        for i, (marker_values, coverage, y_true) in batch_bar:
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                _, _, loss, _ = model(marker_values, coverage, y_true)
                loss = loss / args.grad_accum_steps  # Normalize for gradient accumulation
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            batch_loss = loss.item() * args.grad_accum_steps
            train_loss += batch_loss
            
            # Update batch progress bar
            batch_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
            
            # Gradient accumulation and optimization step
            if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # Validation phase
        model.eval()
        val_loss = 0
        calibration_error = 0
        all_preds = []
        all_targets = []
        
        # Use tqdm for validation
        val_bar = tqdm(val_loader, 
                      desc=f"Epoch {epoch+1}/{args.epochs} [Validate]", 
                      position=1, 
                      leave=False)
        
        with torch.no_grad():
            for marker_values, coverage, y_true in val_bar:
                marker_values = marker_values.to(device)
                coverage = coverage.to(device)
                y_true = y_true.to(device)
                
                mu, phi, loss, _ = model(marker_values, coverage, y_true)
                val_loss += loss.item()
                
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
        calibration_error /= len(val_loader)
        train_loss /= len(train_loader)
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['calibration_error'].append(calibration_error)
        history['r2_score'].append(r2)
        history['mean_absolute_error'].append(mae)
        history['lr'].append(current_lr)
        
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
            f"Calibration Error: {calibration_error:.4f}, "
            f"R² Score: {r2:.4f}, "
            f"MAE: {mae:.6f}, "
            f"LR: {current_lr:.2e}"
        )
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100
            best_val_loss = val_loss
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'r2_score': r2,
                'mae': mae,
                'calibration_error': calibration_error
            }
            torch.save(best_model_state, os.path.join(args.output_dir, 'best_model.pt'))
            patience_counter = 0
            logger.info(f"✓ New best model saved! Improvement: {improvement:.2f}%")
        else:
            patience_counter += 1
            logger.info(f"× No improvement. Patience: {patience_counter}/{args.early_stopping}")
            
        # Early stopping check
        if patience_counter >= args.early_stopping:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'history': history,
                'args': vars(args)
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
        'r2_score': r2,
        'mae': mae,
        'calibration_error': calibration_error
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        serializable_history = {}
        for key, values in history.items():
            serializable_history[key] = [float(v) for v in values]
        json.dump(serializable_history, f)
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
    """Plot and save training history with enhanced visualizations"""
    # Set style for better visualizations
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create main figure with multiple subplots
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot train and validation loss
    epochs = range(1, len(history['train_loss']) + 1)
    
    axs[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axs[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    axs[0, 0].set_title('Training & Validation Loss', fontsize=14)
    axs[0, 0].set_xlabel('Epoch', fontsize=12)
    axs[0, 0].set_ylabel('Loss', fontsize=12)
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot calibration error
    axs[0, 1].plot(epochs, history['calibration_error'], 'g-', linewidth=2)
    axs[0, 1].set_title('Calibration Error', fontsize=14)
    axs[0, 1].set_xlabel('Epoch', fontsize=12)
    axs[0, 1].set_ylabel('Error', fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot R² score
    axs[1, 0].plot(epochs, history['r2_score'], 'purple', linewidth=2)
    axs[1, 0].set_title('R² Score', fontsize=14)
    axs[1, 0].set_xlabel('Epoch', fontsize=12)
    axs[1, 0].set_ylabel('R²', fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot MAE
    axs[1, 1].plot(epochs, history['mean_absolute_error'], 'orange', linewidth=2)
    axs[1, 1].set_title('Mean Absolute Error', fontsize=14)
    axs[1, 1].set_xlabel('Epoch', fontsize=12)
    axs[1, 1].set_ylabel('MAE', fontsize=12)
    axs[1, 1].grid(True, alpha=0.3)
    
    # Plot learning rate
    axs[2, 0].plot(epochs, history['lr'], 'c-', linewidth=2)
    axs[2, 0].set_title('Learning Rate', fontsize=14)
    axs[2, 0].set_xlabel('Epoch', fontsize=12)
    axs[2, 0].set_ylabel('Learning Rate', fontsize=12)
    axs[2, 0].set_yscale('log')  # Log scale for better visualization of LR decay
    axs[2, 0].grid(True, alpha=0.3)
    
    # Plot correlation between metrics (R² vs Loss)
    axs[2, 1].scatter(history['val_loss'], history['r2_score'], alpha=0.7, c=epochs, cmap='viridis')
    axs[2, 1].set_title('R² vs Validation Loss', fontsize=14)
    axs[2, 1].set_xlabel('Validation Loss', fontsize=12)
    axs[2, 1].set_ylabel('R² Score', fontsize=12)
    axs[2, 1].grid(True, alpha=0.3)
    
    for i, epoch in enumerate(epochs):
        if i % 5 == 0 or i == len(epochs) - 1:  # Label every 5th epoch and the last one
            axs[2, 1].annotate(
                f"{epoch}", 
                (history['val_loss'][i], history['r2_score'][i]),
                fontsize=9,
                alpha=0.8
            )
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    
    # Create individual plots for better detail
    metrics = [
        ('loss', ['train_loss', 'val_loss'], ['Train Loss', 'Validation Loss'], ['blue', 'red']),
        ('r2_score', ['r2_score'], ['R² Score'], ['purple']),
        ('mae', ['mean_absolute_error'], ['Mean Absolute Error'], ['orange']),
        ('calibration', ['calibration_error'], ['Calibration Error'], ['green'])
    ]
    
    for name, keys, labels, colors in metrics:
        plt.figure(figsize=(10, 6))
        for key, label, color in zip(keys, labels, colors):
            plt.plot(epochs, history[key], color=color, linewidth=2, label=label)
        plt.title(f'{labels[0]}' if len(labels) == 1 else 'Loss Curves', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(name.replace('_', ' ').title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        if len(keys) > 1:
            plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_history.png'), dpi=300)
        plt.close()
    
    # Close the main figure
    plt.close(fig)


def evaluate(model, test_loader, args, device):
    """Evaluate the model on the test set with visualization"""
    logger = logging.getLogger('cancer_detection')
    logger.info("Starting model evaluation on test set...")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_lower_ci = []
    all_upper_ci = []
    all_marker_attentions = []
    
    # Create progress bar for test evaluation
    test_bar = tqdm(test_loader, desc="Evaluating", position=0)
    
    with torch.no_grad():
        for marker_values, coverage, y_true in test_bar:
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            
            mu, phi, attention_weights = model(marker_values, coverage)
            estimate, ci, uncertainty = model.get_estimate_and_ci(mu, phi)
            
            all_preds.append(estimate.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
            all_lower_ci.append(ci[:, 0:1].cpu().numpy())
            all_upper_ci.append(ci[:, 1:2].cpu().numpy())
            all_marker_attentions.append(attention_weights.cpu().numpy())
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_lower_ci = np.concatenate(all_lower_ci)
    all_upper_ci = np.concatenate(all_upper_ci)
    all_marker_attentions = np.concatenate(all_marker_attentions)
    
    # Calculate metrics
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Calculate percentage of targets within CI
    in_ci = ((all_targets >= all_lower_ci) & (all_targets <= all_upper_ci)).mean()
    
    # Calculate average CI width
    ci_width = (all_upper_ci - all_lower_ci).mean()
    
    # Log results
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS:")
    logger.info(f"Number of test samples: {len(all_targets)}")
    logger.info(f"R² Score: {r2:.6f}")
    logger.info(f"Mean Absolute Error: {mae:.6f}")
    logger.info(f"Targets within CI: {in_ci * 100:.2f}%")
    logger.info(f"Average CI Width: {ci_width:.6f}")
    logger.info("="*50 + "\n")
    
    # Save predictions and metrics
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
        }
    }
    
    results_file = os.path.join(args.output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test results saved to {results_file}")
    
    # Create standard plots with matplotlib
    logger.info("Creating basic prediction visualizations...")
    plot_predictions(all_preds, all_targets, all_lower_ci, all_upper_ci, args.output_dir)
    
    # Analyze marker importance
    logger.info("Analyzing marker importance...")
    analyzer = MarkerImportanceAnalyzer(model)
    marker_importance_file = os.path.join(args.output_dir, 'marker_importance.npy')
    np.save(marker_importance_file, all_marker_attentions.mean(axis=0))
    logger.info(f"Marker importance saved to {marker_importance_file}")
    
    # Create advanced visualizations with Plotly
    try:
        from deep_conv.detect.visualise import create_visualizations
        
        logger.info("Creating advanced visualizations with Plotly...")
        # Create visualization directories
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        test_viz_dir = os.path.join(viz_dir, 'test')
        os.makedirs(test_viz_dir, exist_ok=True)
        
        # Generate visualizations for test data
        metrics = create_visualizations(
            predictions=all_preds.flatten(), 
            ground_truth=all_targets.flatten(),
            output_dir=test_viz_dir
        )
        
        logger.info(f"Advanced visualizations saved to {test_viz_dir}")
    except ImportError:
        logger.warning("Plotly visualization module not found. Skipping advanced visualizations.")
    except Exception as e:
        logger.error(f"Error creating advanced visualizations: {str(e)}")
    
    return results


def run_final_validation(model, val_loader, args, device):
    """Run a final validation pass with visualizations"""
    logger = logging.getLogger('cancer_detection')
    logger.info("Running final validation with visualizations...")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for marker_values, coverage, y_true in tqdm(val_loader, desc="Final Validation"):
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            
            mu, phi, _ = model(marker_values, coverage)
            
            all_preds.append(mu.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Create visualization directories
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    val_viz_dir = os.path.join(viz_dir, 'validation')
    os.makedirs(val_viz_dir, exist_ok=True)
    
    # Try to create advanced visualizations
    try:
        from visualize import create_visualizations
        
        # Generate visualizations for validation data
        metrics = create_visualizations(
            predictions=all_preds.flatten(), 
            ground_truth=all_targets.flatten(),
            output_dir=val_viz_dir
        )
        
        logger.info(f"Validation visualizations saved to {val_viz_dir}")
    except ImportError:
        logger.warning("Plotly visualization module not found. Skipping advanced visualizations.")
    except Exception as e:
        logger.error(f"Error creating validation visualizations: {str(e)}")


def plot_predictions(predictions, targets, lower_ci, upper_ci, output_dir):
    """Plot predictions vs targets with confidence intervals and enhanced visualizations"""
    # Create directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style for better visualizations
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Sort by targets for clearer visualization
    sorted_indices = np.argsort(targets.flatten())
    sorted_targets = targets.flatten()[sorted_indices]
    sorted_preds = predictions.flatten()[sorted_indices]
    sorted_lower = lower_ci.flatten()[sorted_indices]
    sorted_upper = upper_ci.flatten()[sorted_indices]
    
    # 1. Plot predictions with CI (sorted)
    plt.figure(figsize=(12, 8))
    
    # Plot confidence intervals as a shaded region
    plt.fill_between(
        np.arange(len(sorted_targets)),
        sorted_lower,
        sorted_upper,
        alpha=0.3,
        color='blue',
        label='95% Confidence Interval'
    )
    
    # Plot predictions
    plt.plot(
        np.arange(len(sorted_targets)),
        sorted_preds,
        'bo-',
        alpha=0.7,
        markersize=4,
        label='Predictions'
    )
    
    # Plot targets
    plt.plot(
        np.arange(len(sorted_targets)),
        sorted_targets,
        'ro-',
        alpha=0.7,
        markersize=4,
        label='True Values'
    )
    
    # Calculate and show R²
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    plt.title(f'Predictions vs Targets (Sorted) (R² = {r2:.4f}, MAE = {mae:.4f})', fontsize=14)
    plt.xlabel('Sample Index (sorted by true value)', fontsize=12)
    plt.ylabel('Cancer Concentration', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predictions_sorted.png'), dpi=300)
    plt.close()
    
    # 2. Create scatter plot of predictions vs targets
    plt.figure(figsize=(10, 10))
    
    # Plot points with error bars for CI
    plt.errorbar(
        targets.flatten(),
        predictions.flatten(),
        yerr=[predictions.flatten() - lower_ci.flatten(), upper_ci.flatten() - predictions.flatten()],
        fmt='o',
        alpha=0.5,
        ecolor='lightgray',
        capsize=0,
        markersize=6,
        label='Predictions with 95% CI'
    )
    
    # Add identity line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    buffer = (max_val - min_val) * 0.05  # 5% buffer
    plt.plot(
        [min_val - buffer, max_val + buffer],
        [min_val - buffer, max_val + buffer],
        'r--',
        linewidth=2,
        label='Perfect Prediction'
    )
    
    plt.xlabel('True Cancer Concentration', fontsize=14)
    plt.ylabel('Predicted Cancer Concentration', fontsize=14)
    plt.title(f'True vs Predicted (R² = {r2:.4f}, MAE = {mae:.4f})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Make square plot and set equal axis limits
    plt.axis('square')
    limit = [min_val - buffer, max_val + buffer]
    plt.xlim(limit)
    plt.ylim(limit)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'true_vs_predicted.png'), dpi=300)
    plt.close()
    
    # 3. Create histogram of errors
    errors = predictions.flatten() - targets.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    plt.title('Prediction Error Distribution', fontsize=14)
    plt.xlabel('Prediction Error (Predicted - True)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics annotations
    plt.annotate(
        f'Mean Error: {np.mean(errors):.4f}\n'
        f'Std Dev: {np.std(errors):.4f}\n'
        f'Median Error: {np.median(errors):.4f}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_distribution.png'), dpi=300)
    plt.close()
    
    # 4. Create heatmap of predictions vs targets
    plt.figure(figsize=(10, 8))
    
    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        targets.flatten(),
        predictions.flatten(),
        bins=20,
        range=[[min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer]]
    )
    
    # Plot heatmap
    plt.imshow(
        heatmap.T,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='viridis'
    )
    
    # Add identity line
    plt.plot(
        [min_val - buffer, max_val + buffer],
        [min_val - buffer, max_val + buffer],
        'r--',
        linewidth=2
    )
    
    plt.colorbar(label='Count')
    plt.title('Density of Predictions vs True Values', fontsize=14)
    plt.xlabel('True Cancer Concentration', fontsize=12)
    plt.ylabel('Predicted Cancer Concentration', fontsize=12)
    plt.grid(False)
    
    # Make square plot
    plt.axis('square')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prediction_density.png'), dpi=300)
    plt.close()
    
    # 5. CI width vs prediction value
    plt.figure(figsize=(10, 6))
    ci_widths = upper_ci.flatten() - lower_ci.flatten()
    
    plt.scatter(
        predictions.flatten(),
        ci_widths,
        alpha=0.5,
        c=np.abs(errors),  # Color by error magnitude
        cmap='coolwarm'
    )
    
    plt.colorbar(label='|Error|')
    plt.title('Uncertainty vs Predicted Value', fontsize=14)
    plt.xlabel('Predicted Cancer Concentration', fontsize=12)
    plt.ylabel('Width of 95% Confidence Interval', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(predictions.flatten(), ci_widths, 1)
    p = np.poly1d(z)
    plt.plot(
        sorted(predictions.flatten()),
        p(sorted(predictions.flatten())),
        "r--",
        linewidth=2,
        label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}'
    )
    plt.legend(fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'uncertainty_analysis.png'), dpi=300)
    plt.close()


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
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
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
        train_loader, val_loader, test_loader, num_markers, data_stats = prepare_data_for_training(
            data_dir=args.data_dir,
            atlas_path=args.atlas_path,
            target_cell_type=args.target_cell_type,
            target_cell_idx=args.target_cell_idx,
            batch_size=args.batch_size
        )
        logger.info(f"✓ Data preparation complete")
    except Exception as e:
        logger.error(f"× Error during data preparation: {str(e)}")
        raise
    
    # Save data stats
    stats_file = os.path.join(args.output_dir, 'data_stats.json')
    with open(stats_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_stats = {k: float(v) for k, v in data_stats.items()}
        json.dump(serializable_stats, f, indent=2)
    logger.info(f"Data statistics saved to {stats_file}")
    
    # Initialize model
    logger.info(f"Initializing model with {num_markers} markers...")
    try:
        model = EnhancedCancerDetectionModel(
            num_markers=num_markers,
            feature_dim=args.feature_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            use_pos_encoding=args.use_pos_encoding
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
    
    # Run final validation with visualizations
    logger.info("Running final validation with visualizations...")
    try:
        run_final_validation(model, val_loader, args, device)
        logger.info("✓ Final validation completed successfully")
    except Exception as e:
        logger.error(f"× Error during final validation: {str(e)}")
        logger.exception(e)
    
    # Evaluate model
    logger.info("Evaluating model on test set...")
    try:
        test_results = evaluate(model, test_loader, args, device)
        logger.info(f"✓ Evaluation completed successfully")
    except Exception as e:
        logger.error(f"× Error during evaluation: {str(e)}")
        logger.exception(e)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Best validation loss: {best_model_state['val_loss']:.6f}")
    logger.info(f"Best validation R²: {best_model_state['r2_score']:.4f}")
    logger.info(f"Best validation MAE: {best_model_state['mae']:.6f}")
    logger.info(f"Test R²: {test_results['metrics']['r2']:.4f}")
    logger.info(f"Test MAE: {test_results['metrics']['mae']:.6f}")
    logger.info(f"All results saved to: {args.output_dir}")
    logger.info("="*60 + "\n")
    
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