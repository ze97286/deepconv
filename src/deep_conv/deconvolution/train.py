import argparse
from collections import defaultdict
import logging
import subprocess
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import random
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import wandb

from deep_conv.deconvolution.train_visualiser import *
from deep_conv.deconvolution.model import *


def get_git_info() -> Dict[str, str]:
    """Get git repository information"""
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


def setup_run_dir(base_dir: str, model_name: str) -> Path:
    """Setup experiment directory with timestamp and git info"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    git_info = get_git_info()
    
    exp_name = f"{model_name}_{timestamp}_{git_info['commit']}"
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save git info
    with open(exp_dir / 'git_info.json', 'w') as f:
        json.dump(git_info, f, indent=2)
    
    return exp_dir


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cell_types: List[str],
    output_dir: str,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 10,
    device: str = 'cuda',
    project_name: str = 'simple_encoder',
    run_name: Optional[str] = None
) -> Tuple[Dict[str, Any], DeconvolutionVisualiser]:
    """
    Train the deconvolution model with detailed validation logging and visualisations using wandb.
    
    Args:
        model (nn.Module): The LISTA-based deconvolution model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        cell_types (List[str]): List of cell type names.
        output_dir (str): Directory to save model checkpoints and logs.
        n_epochs (int, optional): Number of training epochs. Defaults to 1000.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay for optimiser. Defaults to 1e-4.
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 10.
        device (str, optional): Device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.
        project_name (str, optional): Name of the wandb project. Defaults to 'LISTA_Deconvolution'.
        run_name (Optional[str], optional): Optional name for the wandb run. Defaults to None.
    
    Returns:
        Tuple[Dict[str, Any], DeconvolutionVisualiser]: 
            - Best model checkpoint as a dictionary.
            - Visualiser instance.
    """
    
    # Initialise wandb
    wandb.init(project=project_name, name=run_name, config={
        "n_epochs": n_epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "model": type(model).__name__,
        "optimser": "Adam",
        "batch_size": train_loader.batch_size
    })
    
    # Initialise visualisation tool (assuming DeconvolutionVisualiser is defined elsewhere)
    visualiser = DeconvolutionVisualiser()
    
    # Move model to device
    model = model.to(device)
    
    # Initialise optimiser and scheduler
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    

    # Initialise tracking variables
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    running_train_loss = 0.0
    beta = 0.98  # For running average
    
    # Initialise mixed precision training
    scaler = GradScaler()
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    global_step = 0 
    for epoch in range(n_epochs):
        try:
            # Training Phase
            model.train()
            epoch_train_loss = 0.0
            epoch_train_components = defaultdict(float)
            n_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader):
                global_step += 1  
                methylation = batch['methylation'].to(device)
                coverage = batch['coverage'].to(device)
                true_proportions = batch['proportions'].to(device)
                weight = batch['weight'].to(device)  # Extract sample weights
                
                optimiser.zero_grad()
                
                with autocast():
                    pred_proportions, pred_uncertainties, reconstructed = model(methylation, coverage)
                    loss, loss_components = criterion(
                        pred_proportions, pred_uncertainties, 
                        true_proportions, methylation, reconstructed, 
                        weight=weight
                    )
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
                
                # Accumulate loss components
                for name, value in loss_components.items():
                    epoch_train_components[name] += value
                
                # Update running loss
                current_loss = loss.item()
                epoch_train_loss += current_loss
                running_train_loss = beta * running_train_loss + (1 - beta) * current_loss
                
                # Logging every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    wandb.log({
                        'Train/Current_Loss_Batch': current_loss,
                        'Train/Running_Avg_Loss': running_train_loss,
                        'Train/Epoch_Avg_Loss': epoch_train_loss / (batch_idx + 1)
                    }, step=global_step)
                    
            # Average training metrics
            epoch_train_loss /= n_batches
            avg_train_components = {name: value / n_batches for name, value in epoch_train_components.items()}
            
            # Logging training loss
            wandb.log({
                'Loss/Train_Total': epoch_train_loss,
                'Epoch': epoch,
            }, step=global_step)
            for name, value in avg_train_components.items():
                wandb.log({
                    f'Loss/Train_{name}': value
                }, step=global_step)
            
            # Validation Phase
            model.eval()
            val_loss = 0.0
            epoch_val_components = defaultdict(float)
            all_val_preds = []
            all_val_true = []
            val_uncertainties = []
            all_val_recon = []
            n_val_batches = len(val_loader)
    
            with torch.no_grad():
                for batch in val_loader:
                    methylation = batch['methylation'].to(device)
                    coverage = batch['coverage'].to(device)
                    true_proportions = batch['proportions'].to(device)
                    weight = batch['weight'].to(device)  # Extract sample weights
                    
                    pred_proportions, pred_uncertainties, reconstructed = model(methylation, coverage)
                    loss, loss_components = criterion(
                        pred_proportions, pred_uncertainties,
                        true_proportions, methylation, reconstructed, 
                        weight=weight
                    )
                    
                    # Accumulate validation metrics
                    val_loss += loss.item()
                    for name, value in loss_components.items():
                        epoch_val_components[name] += value
                    
                    all_val_preds.append(pred_proportions.cpu().numpy())
                    all_val_true.append(true_proportions.cpu().numpy())
                    val_uncertainties.append(pred_uncertainties.cpu().numpy())
                    all_val_recon.append(reconstructed.cpu().numpy())
                    
            # Average validation metrics
            val_loss /= n_val_batches
            avg_val_components = {name: value / n_val_batches for name, value in epoch_val_components.items()}
            
            # Logging validation loss
            wandb.log({
                'Loss/Validation_Total': val_loss,
                'Learning_Rate': optimiser.param_groups[0]['lr'],
            }, step=global_step)
            for name, value in avg_val_components.items():
                wandb.log({
                    f'Loss/Validation_{name}': value
                }, step=global_step)
            
            # Update visualiser with loss components
            visualiser.update_loss_components(
                train_components=avg_train_components,
                val_components=avg_val_components
            )
            
            # Stack all validation predictions and arrays
            val_preds = np.vstack(all_val_preds)
            val_true = np.vstack(all_val_true)
            val_uncertainties = np.vstack(val_uncertainties)
            val_reconstructed = np.vstack(all_val_recon)
            
            # Update visualiser with training metrics
            visualiser.update_training_metrics(
                epoch + 1,
                epoch_train_loss,
                val_loss,
                optimiser.param_groups[0]['lr']
            )
            
            # Calculate validation metrics including reconstruction error
            val_metrics = evaluate_predictions(val_true, val_preds, cell_types)
            # Calculate proportion range-specific metrics by cell type
            range_metrics_by_cell = analyse_proportion_ranges(val_preds, val_true, val_uncertainties, cell_types)

            # Group metrics by range for visualization
            consolidated_range_metrics = {}
            for cell_type, cell_ranges in range_metrics_by_cell.items():
                for range_name, metrics in cell_ranges.items():
                    if range_name not in consolidated_range_metrics:
                        consolidated_range_metrics[range_name] = {}
                    consolidated_range_metrics[range_name][cell_type] = metrics

            visualiser.update_proportion_range_metrics(epoch + 1, range_metrics_by_cell)

            # Update uncertainty metrics
            visualiser.update_uncertainty_metrics(epoch + 1, val_uncertainties, val_true)
            
            # Update cell type metrics
            visualiser.update_cell_type_metrics(val_metrics, cell_types)
            
            # Logging validation metrics to wandb
            for metric_name, metric_value in val_metrics.items():
                wandb.log({
                    f'Metrics/Validation_{metric_name}': metric_value
                }, global_step)
            for range_name, metrics in consolidated_range_metrics.items():
                for metric, value in metrics.items():
                    wandb.log({
                        f'Metrics/Range_{range_name}_{metric}': value
                    }, step=global_step)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'metrics': val_metrics,
                    'range_metrics': consolidated_range_metrics,
                    'uncertainties': val_uncertainties,
                    'reconstructed': val_reconstructed
                }
                
                # Save best model checkpoint
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save(best_model, best_model_path)
                
                patience_counter = 0
                print('\n✓ New best model saved')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
                
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print epoch summary
            print(f'\nEpoch {epoch + 1}/{n_epochs} Summary:')
            print(f'Train Loss: {epoch_train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {optimiser.param_groups[0]["lr"]:.2e}')
            
            # Print validation results
            print('\nValidation Metrics:')
            try:
                print_validation_results(
                    val_metrics, 
                    range_metrics_by_cell, 
                    cell_types, 
                    top_k=40, 
                    wandb_run=wandb,
                    global_step=global_step,
                )
            except Exception as print_error:
                import traceback
                print(f"Error in print_validation_results: {str(print_error)}")
                print("Stack trace:")
                print(traceback.format_exc())            
            print('---\n')
            
            # Save visualisations to wandb (assuming save_visualisations is adjusted accordingly)
            visualiser.save_to_wandb(wandb, global_step)
            save_visualisations(visualiser, output_dir)
        
        except Exception as e:
            import traceback
            print(f"Error during epoch {epoch + 1}:")
            print(f"Error message: {str(e)}")
            print("Stack trace:")
            print(traceback.format_exc())
            
            # Save error state for debugging
            error_state = {
                'epoch': epoch,
                'error_message': str(e),
                'model_state': model.state_dict(),
                'optimizer_state': optimiser.state_dict(),
                'last_batch': {
                    'methylation_shape': methylation.shape if 'methylation' in locals() else None,
                    'coverage_shape': coverage.shape if 'coverage' in locals() else None,
                    'true_proportions_shape': true_proportions.shape if 'true_proportions' in locals() else None,
                    'pred_proportions_shape': pred_proportions.shape if 'pred_proportions' in locals() else None,
                    'val_metrics': val_metrics if 'val_metrics' in locals() else None,
                    'range_metrics': range_metrics_by_cell if 'range_metrics_by_cell' in locals() else None,
                }
            }
            
            error_checkpoint = os.path.join(checkpoint_dir, f'error_state_epoch_{epoch + 1}.pt')
            torch.save(error_state, error_checkpoint)
            print(f"Error state saved to: {error_checkpoint}")
            break

        
    # Load best model before returning
    if best_model is not None:
        model.load_state_dict(best_model['model_state_dict'])
    
    # Finish wandb run
    wandb.finish()
    
    return best_model, visualiser


def analyse_proportion_ranges(pred: np.ndarray, 
                              true: np.ndarray, 
                              uncertainties: Optional[np.ndarray] = None, 
                              cell_types: Optional[List[str]] = None,
                              custom_ranges: Optional[List[Tuple[float, float, str]]] = None
                             ) -> Dict[str, Dict[str, float]]:
    """
    Analyse prediction performance across different proportion ranges for each cell type,
    including optional uncertainty analysis.
    
    Args:
        pred (np.ndarray): Predicted proportions (n_samples x n_cell_types).
        true (np.ndarray): True proportions (n_samples x n_cell_types).
        uncertainties (Optional[np.ndarray]): Predicted uncertainties (n_samples x n_cell_types).
        cell_types (Optional[List[str]]): List of cell type names. If None, cell types are named generically.
        custom_ranges (Optional[List[Tuple[float, float, str]]]): 
            Custom proportion ranges as a list of tuples (min_val, max_val, range_name). 
            If None, predefined ranges are used.
    
    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with cell types as keys, each containing
            a dictionary of metrics per proportion range.
            Example:
            {
                'CellType_A': {
                    'Ultra-low (0, 1e-4)': {metrics},
                    'Very-low (1e-4, 1e-3)': {metrics},
                    ...
                },
                'CellType_B': {
                    ...
                },
                ...
            }
    """
    if custom_ranges is not None:
        ranges = custom_ranges
    else:
        ranges = [
            (0, 1e-4, "Ultra-low (0, 1e-4)"),
            (1e-4, 1e-3, "Very-low (1e-4, 1e-3)"),
            (1e-3, 1e-2, "Low (1e-3, 1e-2)"),
            (1e-2, 1e-1, "Medium (1e-2, 1e-1)"),
            (1e-1, 1.0, "High (1e-1, 1.0)")
        ]
    
    n_samples, n_cell_types = true.shape
    
    if cell_types is None:
        cell_types = [f'CellType_{i}' for i in range(n_cell_types)]
    
    range_metrics = {cell: {} for cell in cell_types}
    
    for i, cell in enumerate(cell_types):
        for min_val, max_val, range_name in ranges:
            # Create mask for the current cell type and proportion range
            mask = (true[:, i] >= min_val) & (true[:, i] < max_val)
            
            if np.any(mask):
                # Extract relevant predictions and true values
                true_masked = true[mask, i]
                pred_masked = pred[mask, i]
                
                # Calculate R² only if there are enough unique true values
                unique_true = np.unique(true_masked)
                if len(unique_true) > 1:
                    r2 = r2_score(true_masked, pred_masked)
                else:
                    r2 = np.nan  # R² is undefined with a single unique value
                
                # Basic Metrics
                mae = np.mean(np.abs(pred_masked - true_masked))
                rmse = np.sqrt(np.mean((pred_masked - true_masked)**2))
                mean_true = np.mean(true_masked)
                mean_pred = np.mean(pred_masked)
                n_samples_in_range = np.sum(mask)
                
                metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mean_true': mean_true,
                    'mean_pred': mean_pred,
                    'n_samples': int(n_samples_in_range)
                }
                
                # Uncertainty Metrics (if provided)
                if uncertainties is not None:
                    uncert_masked = uncertainties[mask, i]
                    abs_errors = np.abs(pred_masked - true_masked)
                    
                    # Mean and Median Uncertainty
                    mean_uncertainty = np.mean(uncert_masked)
                    median_uncertainty = np.median(uncert_masked)
                    
                    # Correlation between Uncertainty and Absolute Errors
                    if np.std(uncert_masked) > 0 and np.std(abs_errors) > 0:
                        uncertainty_corr = np.corrcoef(uncert_masked, abs_errors)[0, 1]
                    else:
                        uncertainty_corr = np.nan  # Correlation undefined
                    
                    # Standard Deviation of Uncertainty
                    uncertainty_std = np.std(uncert_masked)
                    
                    # Add to metrics
                    metrics.update({
                        'mean_uncertainty': mean_uncertainty,
                        'median_uncertainty': median_uncertainty,
                        'uncertainty_corr': uncertainty_corr,
                        'uncertainty_std': uncertainty_std
                    })
                    
                    # Uncertainty Calibration
                    for p in [50, 90, 95, 99]:
                        error_p = np.percentile(abs_errors, p)
                        uncertainty_p = np.percentile(uncert_masked, p)
                        calibration = np.mean(abs_errors <= uncertainty_p)
                        
                        metrics.update({
                            f'error_p{p}': error_p,
                            f'uncertainty_p{p}': uncertainty_p,
                            f'calibration_{p}': calibration
                        })
                    
                    # Proportion-Weighted Metrics
                    if true_masked.sum() > 0:
                        weights = true_masked / true_masked.sum()
                        weighted_uncertainty = np.average(uncert_masked, weights=weights)
                        weighted_error = np.average(abs_errors, weights=weights)
                    else:
                        weighted_uncertainty = np.nan
                        weighted_error = np.nan
                        
                    metrics.update({
                        'weighted_uncertainty': weighted_uncertainty,
                        'weighted_error': weighted_error
                    })
                
                # Assign metrics to the appropriate range
                range_metrics[cell][range_name] = metrics
            else:
                # No samples in this range; assign NaN to all metrics
                metrics = {
                    'mae': np.nan,
                    'rmse': np.nan,
                    'r2': np.nan,
                    'mean_true': np.nan,
                    'mean_pred': np.nan,
                    'n_samples': 0
                }
                
                if uncertainties is not None:
                    metrics.update({
                        'mean_uncertainty': np.nan,
                        'median_uncertainty': np.nan,
                        'uncertainty_corr': np.nan,
                        'uncertainty_std': np.nan,
                        'weighted_uncertainty': np.nan,
                        'weighted_error': np.nan
                    })
                    for p in [50, 90, 95, 99]:
                        metrics.update({
                            f'error_p{p}': np.nan,
                            f'uncertainty_p{p}': np.nan,
                            f'calibration_{p}': np.nan
                        })
                
                range_metrics[cell][range_name] = metrics
    
    return range_metrics


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, cell_types: List[str]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for predictions.
    
    Args:
        y_true (np.ndarray): True cell type proportions (n_samples x n_cell_types).
        y_pred (np.ndarray): Predicted cell type proportions (n_samples x n_cell_types).
        cell_types (List[str]): List of cell type names.
    
    Returns:
        Dict[str, Any]: 
            - 'overall_mae': Float, mean absolute error across all cell types and samples.
            - 'overall_rmse': Float, root mean squared error across all cell types and samples.
            - 'cell_type_metrics': Dict[str, Dict[str, float]], 
                where each key is a cell type and each value is a dictionary of metrics for that cell type.
    """
    # Input validation
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be numpy arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.shape[1] != len(cell_types):
        raise ValueError("The number of cell types must match the second dimension of y_true and y_pred.")
    
    # Overall metrics
    overall_mae = np.mean(np.abs(y_true - y_pred))
    overall_rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    
    # Per cell type metrics
    cell_type_metrics = {}
    for i, cell_type in enumerate(cell_types):
        true_props = y_true[:, i]
        pred_props = y_pred[:, i]
        
        # Basic metrics
        mae = np.mean(np.abs(true_props - pred_props))
        rmse = np.sqrt(mean_squared_error(true_props, pred_props))
        
        # Handle R² calculation
        try:
            if np.all(true_props == true_props[0]):
                r2 = np.nan
            else:
                r2 = r2_score(true_props, pred_props)
        except Exception as e:
            r2 = np.nan
            print(f"R² calculation error for {cell_type}: {e}")
            
        # Handle correlation calculation
        try:
            if len(true_props) > 1 and not np.all(true_props == true_props[0]):
                corr_matrix = np.corrcoef(true_props, pred_props)
                corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else np.nan
            else:
                corr = np.nan
        except Exception as e:
            corr = np.nan
            print(f"Correlation calculation error for {cell_type}: {e}")
        
        cell_type_metrics[cell_type] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'corr': corr,
            'mean_true': np.mean(true_props),
            'mean_pred': np.mean(pred_props)
        }

    return {
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'cell_type_metrics': cell_type_metrics
    }


def print_validation_results(metrics: Dict[str, Any],
                           range_metrics: Dict[str, Dict[str, Dict[str, float]]],
                           cell_types: List[str],
                           top_k: int = 40,
                           wandb_run=None,
                           global_step=0):
    """
    Print formatted validation results with uncertainty information and optionally log to wandb.
    
    Args:
        metrics (Dict[str, Any]): Dictionary containing overall and per cell type metrics.
        range_metrics (Dict[str, Dict[str, Dict[str, float]]]): Nested dictionary with structure:
            {cell_type: {range_name: {metric_name: value}}}.
        cell_types (List[str]): List of cell type names.
        top_k (int, optional): Number of top cell types to display based on a sorting criterion. Defaults to 40.
        wandb_run: Optional wandb run instance to log metrics.
    """
    print("\nValidation Performance:")
    overall_mae = metrics.get('overall_mae', np.nan)
    overall_rmse = metrics.get('overall_rmse', np.nan)
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    
    if wandb_run:
        wandb_run.log({
            'Validation/Overall_MAE': overall_mae,
            'Validation/Overall_RMSE': overall_rmse
        }, step=global_step)
    
    print("\nPer Cell Type Performance:")
    header = (
        f"{'Cell Type':<20} {'True %':<8} {'Pred %':<8} {'MAE':<8} "
        f"{'RMSE':<8} {'R²':<8} {'Corr':<8}"
    )
    print(header)
    print("-" * len(header))
    
    cell_metrics = metrics.get('cell_type_metrics', {})
    # Sort cell types by MAE ascending
    sorted_types = sorted(cell_types, 
                         key=lambda x: cell_metrics.get(x, {}).get('mae', np.nan),
                         reverse=False)
    
    for cell_type in sorted_types[:top_k]:
        m = cell_metrics.get(cell_type, {})
        true_pct = m.get('mean_true', np.nan) * 100
        pred_pct = m.get('mean_pred', np.nan) * 100
        mae = m.get('mae', np.nan)
        rmse = m.get('rmse', np.nan)
        r2 = m.get('r2', np.nan)
        corr = m.get('corr', np.nan)
        
        r2_str = '-'.rjust(7) if np.isnan(r2) else f"{r2:7.4f}"
        corr_str = '-'.rjust(7) if np.isnan(corr) else f"{corr:7.4f}"
        print(f"{cell_type[:20]:<20} "
              f"{true_pct:7.2f}% "
              f"{pred_pct:7.2f}% "
              f"{mae:7.4f} "
              f"{rmse:7.4f} "
              f"{r2_str} "
              f"{corr_str}")
        
        if wandb_run:
            wandb_run.log({
                f'Validation/{cell_type}/Mean_True': true_pct,
                f'Validation/{cell_type}/Mean_Pred': pred_pct,
                f'Validation/{cell_type}/MAE': mae,
                f'Validation/{cell_type}/RMSE': rmse,
                f'Validation/{cell_type}/R2': r2,
                f'Validation/{cell_type}/Correlation': corr
            }, step=global_step)
    
    if len(sorted_types) > top_k:
        print("...")
    
    # Print range-specific performance
    print("\nPerformance by Proportion Range:")
    if not range_metrics:
        print("No range-specific metrics available.")
        return
    
    # Check if range metrics contain uncertainty information
    sample_cell_type = next(iter(range_metrics))
    sample_range = next(iter(range_metrics[sample_cell_type]))
    has_uncertainty = 'mean_uncertainty' in range_metrics[sample_cell_type][sample_range]
    
    if has_uncertainty:
        header = (
            f"{'Cell Type':<20} {'Range':<25} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'N':<8} "
            f"{'Uncert':<8} {'Cal-90':<8}"
        )
    else:
        header = (
            f"{'Cell Type':<20} {'Range':<25} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'N':<8}"
        )
    print(header)
    print("-" * len(header))
    
    # Iterate through cell types and their ranges
    for cell_type in sorted_types[:top_k]:
        if cell_type in range_metrics:
            cell_range_metrics = range_metrics[cell_type]
            for range_name, range_m in cell_range_metrics.items():
                mae = range_m.get('mae', np.nan)
                rmse = range_m.get('rmse', np.nan)
                r2 = range_m.get('r2', np.nan)
                n = range_m.get('n_samples', 0)
                
                r2_str = '-'.rjust(7) if np.isnan(r2) else f"{r2:7.4f}"
                
                if has_uncertainty:
                    mean_uncertainty = range_m.get('mean_uncertainty', np.nan)
                    cal_90 = range_m.get('calibration_90', np.nan)
                    uncert_str = '-'.rjust(8) if np.isnan(mean_uncertainty) else f"{mean_uncertainty:7.4f}"
                    cal90_str = '-'.rjust(8) if np.isnan(cal_90) else f"{cal_90:7.4f}"
                    print(f"{cell_type[:20]:<20} "
                          f"{range_name:<25} "
                          f"{mae:7.4f} "
                          f"{rmse:7.4f} "
                          f"{r2_str} "
                          f"{n:7} "
                          f"{uncert_str} "
                          f"{cal90_str}")
                    
                    if wandb_run:
                        wandb_run.log({
                            f'Validation/{cell_type}/Range/{range_name}/MAE': mae,
                            f'Validation/{cell_type}/Range/{range_name}/RMSE': rmse,
                            f'Validation/{cell_type}/Range/{range_name}/R2': r2,
                            f'Validation/{cell_type}/Range/{range_name}/N': n,
                            f'Validation/{cell_type}/Range/{range_name}/Mean_Uncertainty': mean_uncertainty,
                            f'Validation/{cell_type}/Range/{range_name}/Calibration_90': cal_90
                        }, step=global_step)
                else:
                    print(f"{cell_type[:20]:<20} "
                          f"{range_name:<25} "
                          f"{mae:7.4f} "
                          f"{rmse:7.4f} "
                          f"{r2_str} "
                          f"{n:7}")
                    
                    if wandb_run:
                        wandb_run.log({
                            f'Validation/{cell_type}/Range/{range_name}/MAE': mae,
                            f'Validation/{cell_type}/Range/{range_name}/RMSE': rmse,
                            f'Validation/{cell_type}/Range/{range_name}/R2': r2,
                            f'Validation/{cell_type}/Range/{range_name}/N': n
                        }, step=global_step)


def train_deconvolution_model(dataset_dict, model_save_path, batch_size=32):
    """Setup and train the deconvolution model with weighted sampling"""

    # Create datasets
    train_dataset = DeepConvDataset(
        dataset_dict['X_train'],
        dataset_dict['coverage_train'],
        dataset_dict['y_train']
    )
    val_dataset = DeepConvDataset(
        dataset_dict['X_val'],
        dataset_dict['coverage_val'],
        dataset_dict['y_val']
    )

    # Create weighted sampler
    train_sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset.weights),
        replacement=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    print("Initialising model...")
    n_regions = dataset_dict['X_train'].shape[1]
    n_cell_types = dataset_dict['y_train'].shape[1]
    
    model = DiffNNLS(
        n_markers=n_regions,
        n_cell_types=n_cell_types,
    )
    criterion = DiffNNLSLoss()
    
    # Initialise the reference profiles as a model parameter
    reference_profiles = torch.tensor(
        dataset_dict['reference_profiles'],
        dtype=torch.float32
    )
    model.register_buffer('reference_profiles', reference_profiles)
    
    # Count model parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialised with {n_params:,} parameters")

    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting training on {device}")
    
    best_model, visualiser = train_model(
        model=model, 
        criterion=criterion,
        train_loader=train_loader, 
        val_loader=val_loader, 
        cell_types=dataset_dict['cell_types'],
        output_dir=model_save_path,
        device=device
    )
    
    return model, best_model, visualiser


def load_dataset(dataset_path: str) -> Dict[str, np.ndarray]:
    """
    Load the synthetic dataset from a .npz file.

    Args:
        dataset_path (str): Path to the .npz dataset file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing dataset components.
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    data = np.load(dataset_path, allow_pickle=True)
    print(f"Loaded dataset from {dataset_path}")
    return {
        'X_train': data['X_train'],         # Shape: (n_train_samples, n_markers)
        'coverage_train': data['coverage_train'],  # Shape: (n_train_samples, n_markers)
        'y_train': data['y_train'],         # Shape: (n_train_samples, n_cell_types)
        'X_val': data['X_val'],             # Shape: (n_val_samples, n_markers)
        'coverage_val': data['coverage_val'],      # Shape: (n_val_samples, n_markers)
        'y_val': data['y_val'],             # Shape: (n_val_samples, n_cell_types)
        'cell_types': data['cell_types'].tolist(),   # List of cell type names
        'reference_profiles': data['reference_profiles']  # Shape: (n_cell_types, n_markers)
    }


def train(model_save_path, dataset_path, batch_size=32):
    dataset_dict = load_dataset(dataset_path)
    train_deconvolution_model(dataset_dict, model_save_path, batch_size)  


# python -m deep_conv.deconvolution.train --batch_size 32  --dataset_path path/to/training_set.npz --model_save_path /path/to/saved_models/
def main():
    parser = argparse.ArgumentParser(description="DeepConv")
    parser.add_argument("--dataset_path", help="path to dataset",required=True)
    parser.add_argument("--batch_size",type=int, help="training batch size",required=False, default=32)
    parser.add_argument("--model_save_path",help="path to save the trained model",required=True)
    parser.add_argument("--model_name",help="model name",required=False, default="scale_aware")
    inputs = parser.parse_args()   

    set_seed()
    exp_dir = setup_run_dir(inputs.model_save_path, inputs.model_name)
    logging.info(f"Experiment directory: {exp_dir}")

    # Save experiment config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(inputs), f, indent=2)

    train(
        model_save_path=str(exp_dir),
        dataset_path=inputs.dataset_path,
        batch_size=inputs.batch_size,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        raise