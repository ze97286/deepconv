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
    """
    Plot and save training history with enhanced visualizations using Plotly
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
    """
    Plot predictions vs targets with confidence intervals using Plotly
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Sort by targets for clearer visualization
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
        yaxis_title='Cancer Concentration',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        width=900,
        height=600
    )
    
    # Save figure
    fig.write_html(os.path.join(plots_dir, 'predictions_vs_targets.html'))
    fig.write_image(os.path.join(plots_dir, 'predictions_vs_targets.png'), scale=2)
    
    # 2. Create scatter plot
    fig = go.Figure()
    
    # Add error bars
    fig.add_trace(
        go.Scatter(
            x=targets.flatten(),
            y=predictions.flatten(),
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6
            ),
            error_y=dict(
                type='data',
                symmetric=False,
                array=upper_ci.flatten() - predictions.flatten(),
                arrayminus=predictions.flatten() - lower_ci.flatten(),
                thickness=1.5,
                width=3
            ),
            name='Predictions with 95% CI'
        )
    )
    
    # Add identity line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'True vs Predicted (R² = {r2:.4f}, MAE = {mae:.4f})',
        xaxis_title='True Cancer Concentration',
        yaxis_title='Predicted Cancer Concentration',
        template='plotly_white',
        width=800,
        height=800,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Make square plot
    fig.update_layout(yaxis=dict(scaleanchor='x', scaleratio=1))
    
    # Save figure
    fig.write_html(os.path.join(plots_dir, 'true_vs_predicted.html'))
    fig.write_image(os.path.join(plots_dir, 'true_vs_predicted.png'), scale=2)
    
    # 3. Create error histogram
    errors = predictions.flatten() - targets.flatten()
    
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=30,
            marker_color='blue',
            opacity=0.7
        )
    )
    
    # Add vertical line at zero
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Calculate error statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    
    # Add annotation
    fig.add_annotation(
        x=0.05,
        y=0.9,
        xref="paper",
        yref="paper",
        text=f"Mean: {mean_error:.4f}<br>Std Dev: {std_error:.4f}<br>Median: {median_error:.4f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title='Prediction Error Distribution',
        xaxis_title='Error (Predicted - True)',
        yaxis_title='Count',
        template='plotly_white',
        width=800,
        height=600
    )
    
    # Save figure
    fig.write_html(os.path.join(plots_dir, 'error_distribution.html'))
    fig.write_image(os.path.join(plots_dir, 'error_distribution.png'), scale=2)


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