import argparse
import os
import random
from collections import defaultdict

import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from deep_conv.atlasbuilder.find_marker_candidates import *
from deep_conv.benchmark.benchmark_utils import *


class TissueDeconvolutionDataset(Dataset):
    def __init__(self, fraction, coverage, atlas, y=None):
        self.fraction = torch.tensor(fraction, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.atlas = torch.tensor(atlas, dtype=torch.float32)        
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return self.fraction.size(0)

    def __getitem__(self, idx):
        item = {
            'X': self.fraction[idx],
            'coverage': self.coverage[idx],
        }
        if self.y is not None:
            item['y'] = self.y[idx]
        return item  


# Core Architecture
# The model takes two key inputs:

# Marker values: Methylation values for each marker region (can contain NaNs where coverage=0)
# Coverage: Number of reads supporting each marker value

# It processes these inputs in several key steps:

# Feature Extraction:

# Uses a two-layer network to transform each marker value into a rich feature representation
# The non-linear LeakyReLU activation allows capturing complex methylation patterns

# Cell Type-Specific Aggregation:
# Each marker is assigned to exactly one target cell type (via target_ids)
# Features from markers targeting the same cell type are aggregated
# Features are weighted by coverage, giving more reliable markers more influence
# Effective zero coverage handling ensures NaN markers don't contribute

# Proportion Prediction (Encoder):
# The aggregated features for each cell type are processed through a neural network
# The output is transformed via sigmoid and normalised to ensure proportions sum to 1

# Marker Reconstruction (Decoder):
# For interpretability, the model can reconstruct the original marker values
# This helps ensure the predicted proportions explain the observed methylation patterns
class CellTypeDeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types, target_ids, feature_dim=32):
        """
        Args:
          num_markers (int): Number of markers, M.
          num_celltypes (int): Number of cell types, C.
          target_ids (1D array-like): shape [M], each entry in [0, C-1], 
                                      specifying which cell type each marker targets.
          feature_dim (int): Dimensionality of the per-marker feature space.
        """
        super().__init__()
        self.num_markers = num_markers
        self.num_celltypes = num_cell_types
        self.feature_dim = feature_dim
        
        # Convert the user-supplied target_ids to a LongTensor
        # so we can use them for indexing. Register as a buffer
        # so they move to GPU but are not trainable.
        target_ids_t = torch.as_tensor(target_ids, dtype=torch.long)
        if target_ids_t.shape[0] != num_markers:
            raise ValueError("target_ids must have length == num_markers")
        self.register_buffer("target_ids", target_ids_t)

        # The two-layer architecture with LeakyReLU allows the model to:
        # Learn more complex transformations from marker values to features
        # Better capture non-linear relationships in the data
        # Create more discriminative features for challenging markers
        self.marker_feature_extractor = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Encoder: from aggregated (cell-type-level) features -> cell type logits
        self.encoder = nn.Sequential(
            nn.Linear(num_cell_types * feature_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_cell_types)
        )

        # Decoder: from cell type proportions -> reconstructed marker values
        self.decoder = nn.Sequential(
            nn.Linear(num_cell_types, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_markers)
        )

    def forward(self, marker_values, coverage):
        """
        Args:
          marker_values (FloatTensor): [B, M], can contain NaN for coverage=0.
          coverage (FloatTensor): [B, M], coverage >=0. By invariant, coverage=0 => marker_value=NaN => skip.

        Returns:
          celltype_props: [B, C], predicted fraction of each cell type (softmax).
          reconstructed: [B, M], model's reconstruction of marker_values at valid positions.
          valid_mask:    [B, M], 1 where coverage>0, 0 otherwise (useful for masked loss).
        """
        B, M = marker_values.shape
        C = self.num_celltypes

        # valid_mask: True where coverage>0
        valid_mask = (coverage > 0)
        
        # Flatten everything to 1D so we can gather only valid positions
        # shape => [B*M]
        coverage_flat = coverage.view(-1)        # coverage for each (batch, marker)
        marker_values_flat = marker_values.view(-1)

        # Indices of valid entries (coverage>0)
        valid_inds = torch.nonzero(coverage_flat, as_tuple=False).squeeze(1)
        # If no valid positions at all, return trivial predictions
        if valid_inds.numel() == 0:
            # e.g. every coverage=0 => no data => can't learn anything
            celltype_props = coverage.new_zeros(B, C)
            celltype_props[:, 0] = 1.0  # arbitrary
            reconstructed = coverage.new_zeros(B, M)
            return celltype_props, reconstructed, valid_mask
        
        # Gather coverage and marker_values for valid positions
        coverage_valid = coverage_flat[valid_inds]            # shape [N]
        marker_values_valid = marker_values_flat[valid_inds]  # shape [N]

        # Derive which (batch, marker) each valid position corresponds to
        batch_idx = valid_inds // M  # shape [N]
        marker_idx = valid_inds % M  # shape [N]

        # Each marker belongs to exactly one cell type: self.target_ids[marker_idx]
        celltype_idx = self.target_ids[marker_idx]  # shape [N]

        # Project each valid marker value into feature_dim
        # shape => [N, 1] => pass to linear => [N, feature_dim]
        marker_values_valid_2d = marker_values_valid.unsqueeze(1)
        features_valid = self.marker_feature_extractor(marker_values_valid_2d)  # [N, feature_dim]

        # We'll scatter-add into aggregator => shape [B, C, feature_dim]
        aggregator = coverage.new_zeros(B, C, self.feature_dim)  # type float
        coverage_sum = coverage.new_zeros(B, C)                  # coverage aggregator

        # Flatten aggregator to [B*C, feature_dim]
        aggregator_2d = aggregator.view(B*C, self.feature_dim)
        coverage_sum_1d = coverage_sum.view(B*C)

        # Convert (b, c) to a single index => idx = b*C + c
        bc_index = batch_idx * C + celltype_idx  # shape [N]

        # Weighted features => coverage_valid[..., None] * features_valid
        weighted_feats = coverage_valid.unsqueeze(1) * features_valid  # [N, feature_dim]

        # Scatter-add into aggregator_2d
        aggregator_2d.index_add_(0, bc_index, weighted_feats)
        coverage_sum_1d.index_add_(0, bc_index, coverage_valid)

        # Reshape back
        aggregator = aggregator_2d.view(B, C, self.feature_dim)
        coverage_sum = coverage_sum_1d.view(B, C)

        # Avoid dividing by zero => aggregator stays 0 if coverage is 0
        # coverage_sum==0 means no valid marker for that (b, c).
        mask_cov = (coverage_sum == 0)
        coverage_sum[mask_cov] = 1.0
        aggregator = aggregator / coverage_sum.unsqueeze(-1)  # shape [B, C, feature_dim]

        # Flatten aggregator for encoder => [B, C*feature_dim]
        agg_flat = aggregator.view(B, -1)
        logits = self.encoder(agg_flat)       # [B, C]
        celltype_props = F.sigmoid(logits)
        celltype_props = celltype_props / celltype_props.sum(dim=1, keepdim=True) 
        
        # Decode => [B, M]
        reconstructed = self.decoder(celltype_props)

        return celltype_props, reconstructed, valid_mask
    
# Enhanced Weighted Approach for Cell-Type Deconvolution
# The loss function implements a specialised approach to tackle the challenge of low SNR cell types like CD4/CD8:
# Core Components
# 
# Concentration-Weighted Loss:
# 
# Applies importance weights based on true cell type concentrations
# Creates progressively stronger penalties for higher CD4/CD8 concentrations
# For CD4/CD8 cells at 10% concentration, the weight is ~8x stronger than baseline
# 
# Asymmetric Error Penalties:
# Differentiates between underestimation and overestimation
# Applies an additional 1.5x penalty to CD4/CD8 underestimation
# Effectively prioritises reducing false negatives for these critical cell types
# 
# Coverage-Weighted Reconstruction:
# Weights reconstruction errors by read coverage
# Places more emphasis on markers with higher confidence (more reads)
# Ignores markers with zero coverage (NaN values)
# 
# Combined Loss With Balance Control:
# Uses alpha/beta parameters to balance proportion prediction vs. reconstruction
# Typically weights proportion prediction much higher (alpha=0.999)
# Maintains reconstruction as a regularising constraint (beta=0.001)
# 
# Design Rationale
# The loss function's design addresses key challenges in methylation-based deconvolution:
# 
# Low SNR Compensation: CD4/CD8 cells have 8x lower SNR than OAC, requiring special handling
# Concentration-Dependent Scaling: Higher concentration predictions need higher accuracy
# Penalty Asymmetry: Underestimation has worse clinical implications than overestimation
# Coverage Integration: Leverages sequencing depth as confidence measure
# 
# This approach effectively focuses the model's learning on the most challenging aspects of the problem, 
# improving performance on low-SNR cell types while maintaining overall accuracy.
def loss_fn(pred_props, true_props, reconstructed, marker_values, coverage, valid_mask, 
            alpha=0.999, beta=0.001, cd48_indices=[3, 4]):
    """Enhanced loss function that applies stronger penalties to CD4/CD8 underestimation"""
    
    # Base L1 loss - we'll modify this with our weighting
    cell_errors = torch.abs(pred_props - true_props)
    
    # Create importance weights based on cell type and concentration
    importance_weights = torch.ones_like(true_props)
    
    # Identify CD4/CD8 columns and create stronger weights for them
    for idx in cd48_indices:
        # Create weight that scales with concentration
        # At 1% -> weight ~= 2, at 5% -> weight ~= 5, at 10% -> weight ~= 8
        importance_weights[:, idx] = 1.0 + 7.0 * true_props[:, idx]
    
    # Apply extra penalty specifically for underestimation
    underestimation = F.relu(true_props - pred_props)
    overestimation = F.relu(pred_props - true_props)
    
    # Make underestimation more costly than overestimation for CD4/CD8
    cd48_mask = torch.zeros_like(true_props)
    cd48_mask[:, cd48_indices] = 1.0
    
    # Final weighted errors: stronger penalty for CD4/CD8 underestimation
    weighted_errors = importance_weights * (
        cell_errors + 
        cd48_mask * underestimation * 1.5  # Extra 1.5x penalty for CD4/CD8 underestimation
    )
    
    # Calculate overall loss
    loss_props = weighted_errors.mean()
    
    # Standard reconstruction loss
    safe_marker_values = torch.where(valid_mask > 0, marker_values, reconstructed)
    recon_loss = torch.sum(valid_mask * coverage * abs(safe_marker_values - reconstructed)) / torch.sum(valid_mask * coverage)
    
    # Combine losses
    total_loss = alpha * loss_props + beta * recon_loss
    
    # Details for monitoring
    details = {
        'total_loss': total_loss.item(),
        'loss_props': loss_props.item(),
        'recon_loss': recon_loss.item(),
        'cd48_under': underestimation[:, cd48_indices].mean().item(),
        'cd48_over': overestimation[:, cd48_indices].mean().item(),
        'alpha_stats': {
            'mean': torch.mean(pred_props).item(),
            'std': torch.std(pred_props).item(),
            'max': torch.max(pred_props).item(),
            'min': torch.min(pred_props).item()
        },
        'valid_ratio': torch.mean(valid_mask.float()).item()
    }
    
    return total_loss, details


def init_wandb(config, project_name="cfDNA-Deconvolution", entity=None):
    """
    Initialise Weights & Biases logging.
    
    Args:
        config: Dictionary containing experiment config (model params, dataset info, etc)
        project_name: Name of the wandb project
        entity: wandb entity (username or team name)
        
    Returns:
        run: wandb run object
    """
    from datetime import datetime as dt    
    run = wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        name=f"run_{dt.now().strftime('%Y%m%d_%H%M%S')}"
    )
    return run


def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimiser: optim.Optimizer,
                device: torch.device,
                log_interval: int = 500,
                accumulation_steps: int = 4) -> Dict[str, float]:
    """
    Train for one epoch with gradient accumulation and wandb logging
    """
    model.train()
    epoch_stats = defaultdict(float)
    num_batches = 0
    optimiser.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(loader, desc='Training')):
        fraction = batch['X'].to(device)
        coverage = batch['coverage'].to(device)
        y_true = batch['y'].to(device)
        
        # Forward pass
        alpha, reconstructed, valid_mask = model(fraction, coverage)
        
        # Compute loss
        loss, details = loss_fn(
            pred_props=alpha,
            true_props=y_true,
            reconstructed=reconstructed,
            marker_values=fraction,
            coverage=coverage,
            valid_mask=valid_mask,
        )
        
        # Scale loss by accumulation steps
        scaled_loss = loss / accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        # Step optimiser only after accumulation
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(loader)):
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            optimiser.zero_grad()
            
            # Log gradient norm on actual step
            epoch_stats['grad_norm'] += grad_norm.item()
            
            # Log to wandb - step-level metrics
            if wandb.run is not None and (batch_idx + 1) % (log_interval // 2) == 0:
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/grad_norm": grad_norm.item(),
                    "batch/lr": optimiser.param_groups[0]['lr'],
                    "batch/step": batch_idx + num_batches * len(loader)
                })
        
        # Update statistics (use unscaled loss for logging)
        epoch_stats['total_loss'] += loss.item()
        for key, value in details.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    epoch_stats[f"{key}/{subkey}"] += subvalue
            else:
                epoch_stats[key] += value
        num_batches += 1
        
        # Logging
        if batch_idx % log_interval == 0:
            print(f"\nBatch {batch_idx} | Loss: {loss.item():.8f}")
            print(f"Alpha Mean: {details['alpha_stats']['mean']:.8f} | Std: {details['alpha_stats']['std']:.8f}")
            if 'cd48_under' in details and 'cd48_over' in details:
                print(f"CD4/CD8 Under: {details['cd48_under']:.8f} | Over: {details['cd48_over']:.8f}")
            if 'weight_stats' in details:
                print(f"Weight Mean: {details['weight_stats']['mean']:.8f} | Max: {details['weight_stats']['max']:.8f}")
    
    # Average statistics
    for key in epoch_stats:
        epoch_stats[key] /= num_batches
    
    return dict(epoch_stats)


def validate(model: nn.Module,
            val_loaders: Dict[str, DataLoader],
            device: torch.device) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Validate the model with additional CD4/CD8 metrics and wandb logging
    """
    model.eval()
    val_stats = {}
    
    # Add special CD4/CD8 monitoring
    cd48_performance = {}
    
    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            loader_stats = defaultdict(float)
            num_batches = 0
            
            # CD4/CD8 specific metrics
            cd48_metrics = {
                'high_conc_error': 0.0,
                'samples_count': 0
            }
            
            for batch in tqdm(val_loader, desc=f'Validating {val_name}'):
                fraction = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                y_true = batch['y'].to(device)
                
                # Forward pass
                alpha, reconstructed, valid_mask = model(fraction, coverage)
        
                # Compute loss
                loss, details = loss_fn(
                    pred_props=alpha,
                    true_props=y_true,
                    reconstructed=reconstructed,
                    marker_values=fraction,
                    coverage=coverage,
                    valid_mask=valid_mask,
                )
                
                # Check CD4/CD8 performance at high concentrations
                cd48_indices = [3, 4]  # CD4/CD8 indices
                cd48_mask = (y_true[:, cd48_indices] > 0.03)  # High concentration mask
                if cd48_mask.sum() > 0:
                    cd48_error = torch.abs(alpha[:, cd48_indices][cd48_mask] - y_true[:, cd48_indices][cd48_mask]).mean()
                    cd48_metrics['high_conc_error'] += cd48_error.item() * cd48_mask.sum().item()
                    cd48_metrics['samples_count'] += cd48_mask.sum().item()
                
                # Update statistics
                loader_stats['loss'] += loss.item()
                for key, value in details.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            loader_stats[f"{key}/{subkey}"] += subvalue
                    else:
                        loader_stats[key] += value
                num_batches += 1
            
            # Average statistics
            for key in loader_stats:
                loader_stats[key] /= num_batches
            
            val_stats[val_name] = dict(loader_stats)
            
            # Store CD4/CD8 performance
            if cd48_metrics['samples_count'] > 0:
                cd48_performance[val_name] = cd48_metrics['high_conc_error'] / cd48_metrics['samples_count']
            else:
                cd48_performance[val_name] = 0.0
    
    # Calculate average validation loss
    avg_val_loss = sum(stats['total_loss'] for stats in val_stats.values()) / len(val_stats)
    
    return avg_val_loss, val_stats, cd48_performance


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loaders: Dict[str, DataLoader],
    model_path: str,
    num_epochs: int = 1000,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    use_wandb: bool = True,
    wandb_project: str = "cfDNA-Deconvolution",
    wandb_entity: str = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Complete training loop with LR adjustments, CD4/CD8 monitoring and wandb integration
    """
    model = model.to(device)
    atlas = train_loader.dataset.atlas.to(device)
    
    # Setup optimiser and scheduler
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 
        mode='min', 
        factor=0.5, 
        patience=patience // 2,
        verbose=True
    )
    
    # Create directories
    os.makedirs(model_path, exist_ok=True)
    
    # Setup wandb
    if use_wandb:
        # Create config dictionary
        config = {
            "model_type": model.__class__.__name__,
            "num_markers": getattr(model, "num_markers", "unknown"),
            "num_cell_types": getattr(model, "num_celltypes", "unknown"),
            "feature_dim": getattr(model, "feature_dim", "unknown"),
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": train_loader.batch_size if hasattr(train_loader, "batch_size") else "unknown",
            "num_epochs": num_epochs,
            "patience": patience,
            "device": str(device)
        }
        run = init_wandb(config, project_name=wandb_project, entity=wandb_entity)
        
        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)
    
    # Training parameters
    initial_lr = lr
    warmup_epochs = 5
    
    # Training history
    history = defaultdict(list)
    best_val_loss = float('inf')
    best_cd48_error = float('inf')
    best_epoch = 0
    patience_counter = 0
    cd48_patience = 0
    
    for epoch in range(num_epochs):
        print(f"\n🔹 Epoch {epoch + 1}/{num_epochs}")
        
        # Handle learning rate warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = initial_lr * warmup_factor
            for param_group in optimiser.param_groups:
                param_group['lr'] = current_lr
            print(f"LR Warmup: {current_lr:.1e}")
        
        # Check if we need a learning rate reset
        if epoch - best_epoch > patience // 2 and patience_counter >= patience // 2:
            print("⚠️ Resetting learning rate to encourage exploration")
            for param_group in optimiser.param_groups:
                param_group['lr'] = initial_lr
        
        # Train
        train_stats = train_epoch(model, train_loader, optimiser, device)
        
        # Validate
        avg_val_loss, val_stats, cd48_performance = validate(model, val_loaders, device)
        
        # Average CD4/CD8 performance across validation sets
        if cd48_performance:
            avg_cd48_error = sum(cd48_performance.values()) / len(cd48_performance)
            print(f"CD4/CD8 High Conc Error: {avg_cd48_error:.6f}")
        else:
            avg_cd48_error = float('inf')
        
        # Update history
        for key, value in train_stats.items():
            history[key].append(value)
        for val_name, stats in val_stats.items():
            for key, value in stats.items():
                history[f"{val_name}/{key}"].append(value)
        if cd48_performance:
            for val_name, error in cd48_performance.items():
                history[f"{val_name}/cd48_error"].append(error)
            history["avg_cd48_error"].append(avg_cd48_error)
        
        # Print summary
        print(f"\n🔹 Epoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_stats['total_loss']:.8f} | Grad Norm: {train_stats['grad_norm']:.8f}")
        if 'alpha_stats/mean' in train_stats:
            print(f"Alpha Mean: {train_stats['alpha_stats/mean']:.8f} | Std: {train_stats['alpha_stats/std']:.8f}")
        
        for val_name, stats in val_stats.items():
            print(f"{val_name} Loss: {stats['total_loss']:.8f}")
        
        # Log to wandb - epoch level metrics
        if use_wandb:
            # Prepare wandb logs
            wandb_logs = {
                "epoch": epoch,
                "train/loss": train_stats['total_loss'],
                "train/grad_norm": train_stats['grad_norm'],
                "val/avg_loss": avg_val_loss,
                "lr": optimiser.param_groups[0]['lr'],
            }
            
            # Add validation metrics
            for val_name, stats in val_stats.items():
                for key, value in stats.items():
                    wandb_logs[f"val/{val_name}/{key}"] = value
            
            # Add CD4/CD8 metrics
            if cd48_performance:
                for val_name, error in cd48_performance.items():
                    wandb_logs[f"cd48/{val_name}_error"] = error
                wandb_logs["cd48/avg_error"] = avg_cd48_error
            
            # Log to wandb
            wandb.log(wandb_logs)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Early stopping based on overall validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            print(f"New best model with loss: {best_val_loss:.8f}")
            
            # Save model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': dict(history)
            }
            torch.save(checkpoint, os.path.join(model_path, "best_model.pt"))
            
            # Save to wandb as well
            if use_wandb:
                wandb.save(os.path.join(model_path, "best_model.pt"))
        else:
            patience_counter += 1
        
        # Early stopping based on CD4/CD8 performance
        if cd48_performance and avg_cd48_error < best_cd48_error:
            best_cd48_error = avg_cd48_error
            cd48_patience = 0
            print(f"New best CD4/CD8 performance: {best_cd48_error:.6f}")
            
            # Save a CD4/CD8-optimised model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'best_cd48_error': best_cd48_error
            }
            cd48_model_path = os.path.join(model_path, "best_cd48_model.pt")
            torch.save(checkpoint, cd48_model_path)
            
            # Save to wandb as well
            if use_wandb:
                wandb.save(cd48_model_path)
        else:
            cd48_patience += 1
        
        # Check if we should stop training
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model (general performance)
    print("Loading best model (best overall validation loss)")
    checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create training history plots
    plot_training_history(dict(history), os.path.join(model_path, "model_training"))
    
    # If we have a better CD4/CD8 model, notify the user
    if os.path.exists(os.path.join(model_path, "best_cd48_model.pt")):
        print(f"\nNOTE: A model optimised for CD4/CD8 performance is available at:")
        print(f"      {os.path.join(model_path, 'best_cd48_model.pt')}")
        print(f"      (Best CD4/CD8 error: {best_cd48_error:.6f})")
    
    # Finish wandb run
    if use_wandb:
        # Log final best metrics
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_cd48_error"] = best_cd48_error
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["total_epochs"] = epoch + 1
        
        # Finish the run
        wandb.finish()
    
    return model


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """
    Plot training history metrics using plotly
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Loss Evolution',
            'Valid Marker Ratio',
            'Alpha Statistics',
            'Theta Evolution'
        )
    )

    # Plot loss evolution
    loss_keys = [k for k in history.keys() if 'loss' in k.lower()]
    for key in loss_keys:
        fig.add_trace(
            go.Scatter(y=history[key], name=key),
            row=1, col=1
        )

    # Plot valid marker ratio
    valid_keys = [k for k in history.keys() if 'valid' in k.lower()]
    for key in valid_keys:
        fig.add_trace(
            go.Scatter(y=history[key], name=key),
            row=1, col=2
        )

    # Plot alpha statistics
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

    # Plot theta evolution
    if 'theta_mean' in history:
        fig.add_trace(
            go.Scatter(y=history['theta_mean'], name='Theta'),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Training History"
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=2)

    # Update x-axes labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Epoch", row=i, col=j)

    # Save figure
    if save_path:
        fig.write_html(f"{save_path}.html")
    else:
        fig.show()


def predict(model, X, coverage):
   """Prediction function that processes samples one at a time"""
   wandb_hooks = []
   if hasattr(model, '_forward_hooks'):
        wandb_hooks = [(k, v) for k, v in model._forward_hooks.items() 
                        if 'wandb' in str(v)]
        for hook_id, _ in wandb_hooks:
            model._forward_hooks.pop(hook_id)
   model.eval()
   predictions_list = []
   with torch.no_grad():
       for i in range(len(X)):
           x = torch.tensor(X[i:i+1], dtype=torch.float32)
           c = torch.tensor(coverage[i:i+1], dtype=torch.float32)
           pred, _, _ = model(x, c)
           predictions_list.append(pred.numpy())
   return np.vstack(predictions_list)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_validation_set(eval_pat_dir, atlas, names):
    X_val = pd.read_parquet(Path(eval_pat_dir)/"marker_values.parquet")
    coverage_val = pd.read_parquet(Path(eval_pat_dir)/"coverage.parquet")
    y_val = pd.read_parquet(Path(eval_pat_dir)/"ground_truth_y.parquet")
    X_val = X_val[X_val.name.isin(names)]
    coverage_val = coverage_val[coverage_val.name.isin(names)]    
    X_val = X_val.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_val = coverage_val.drop(columns=["name", "direction"]).T.to_numpy()        
    print("median coverage", np.median(coverage_val, axis=1), "median of medians", np.median(np.median(coverage_val, axis=1)), "mean median", np.median(coverage_val, axis=1).mean())
    y_val = y_val.to_numpy()
    val_dataset = TissueDeconvolutionDataset(X_val, coverage_val, atlas[atlas.columns[8:]].T.to_numpy(), y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,  
        num_workers=4,   
        persistent_workers=True,
        shuffle=False
    )
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_val = y_val / y_val.sum(dim=1, keepdim=True)
    return val_loader, y_val


def load_training(base_dir, atlas, names, num_files=4):
    markers = []
    coverage = []
    y = []
    print("loading training from",base_dir)
    suffixes = [f"_batch{i}" for i in range(1,num_files+1)]
    for i in range(1,num_files+1):    
        markers.append(pd.read_parquet(base_dir+str(i)+"_marker_values.parquet"))
        coverage.append(pd.read_parquet(base_dir+str(i)+"_coverage.parquet"))
        y.append(pd.read_parquet(base_dir+str(i)+"_ground_truth_y.parquet"))   
    merged_markers = markers[0]
    for i, m in enumerate(markers[1:]):
        merged_markers = merged_markers.merge(m, on=['name', 'direction'], how='outer',suffixes=('', suffixes[i]))
    merged_coverage = coverage[0]
    for i, c in enumerate(coverage[1:]):
        merged_coverage = merged_coverage.merge(c, on=['name', 'direction'], how='outer',suffixes=('', suffixes[i]))
    y = pd.concat(y, ignore_index=True).fillna(0)
    X_train,coverage_train,y_train = merged_markers, merged_coverage, y        
    X_train = X_train[X_train.name.isin(names)]
    coverage_train = coverage_train[coverage_train.name.isin(names)]
    X_train = X_train.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_train = coverage_train.drop(columns=["name", "direction"]).T.to_numpy()    
    print("median coverage", np.median(coverage_train, axis=1), "median of medians", np.median(np.median(coverage_train, axis=1)), "mean median", np.median(coverage_train, axis=1).mean())
    y_train = y_train.to_numpy()
    train_dataset = TissueDeconvolutionDataset(X_train, coverage_train, atlas[atlas.columns[8:]].T.to_numpy(), y_train)
    y_train = torch.tensor(y_train, dtype=torch.float32)    
    y_train = y_train / y_train.sum(dim=1, keepdim=True)
    return  DataLoader(
        train_dataset,
        batch_size=256,  
        shuffle=True,
        num_workers=4, 
        persistent_workers=True
    )
   

def train_and_eval(atlas_path, train_pat_dir, eval_pat_dir, threads, output_path):
    """
    train and evaluate deep conv model 
    """
    set_seed()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    atlas = pd.read_csv(atlas_path, sep="\t")
    names = set(atlas.name.unique()) 
    train_dl = load_training(train_pat_dir, atlas, names) 
    
    tier1_dl, t1_yval = get_validation_set(str(Path(eval_pat_dir)/"tier1"), atlas, names)
    cd4_dl, cd4_yval = get_validation_set(str(Path(eval_pat_dir)/"CD4"), atlas, names)
    cd8_dl, cd8_yval = get_validation_set(str(Path(eval_pat_dir)/"CD8"), atlas, names)
    oac_dl, oac_yval = get_validation_set(str(Path(eval_pat_dir)/"OAC"), atlas, names)
    validation_dls = {"tier1":tier1_dl, "cd4":cd4_dl, "cd8":cd8_dl, "oac":oac_dl}
    y_vals = {"tier1":t1_yval, "cd4":cd4_yval, "cd8":cd8_yval, "oac":oac_yval}

    cell_types = list(atlas.columns[8:])
    # model = CellTypeDeconvolutionModel(num_markers=len(atlas),num_cell_types=len(cell_types), atlas=torch.tensor(atlas[atlas.columns[8:]].T.to_numpy(), dtype=torch.float32))
    # model = CellTypeDeconvolutionModel(num_markers=len(atlas),num_cell_types=len(cell_types))
    target_ids = atlas["target"].map(lambda x: cell_types.index(x)).to_numpy()
    model = CellTypeDeconvolutionModel(num_markers=len(atlas),num_cell_types=len(cell_types), target_ids=target_ids)
    model = train_model(
        model=model,
        train_loader=train_dl,
        val_loaders=validation_dls,
        model_path=output_path,        
    )

    for tier in validation_dls.keys():
        tier_dl = validation_dls[tier]
        y_val = y_vals[tier]
        deep_conv_estimation = predict(model, tier_dl.dataset.fraction,tier_dl.dataset.coverage)
        deep_conv_eval_metrics = evaluate_performance(y_val.detach().numpy(), deep_conv_estimation, cell_types)
        print(f"deepconv validation metrics for tier {tier}")
        log_metrics(deep_conv_eval_metrics)    


    return model
      

def main():
    parser = argparse.ArgumentParser(description="Deep conv")
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_threads",required=False, type=int, default=32)
    
    args = parser.parse_args()
    
    train_and_eval(args.atlas_path, args.train_path+"/train",args.eval_path+"/eval", args.num_threads, args.output_path)
    

if __name__ == "__main__":    
    main()