import torch 
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from deep_conv.presence.loss import presence_loss_fn
from deep_conv.presence.model import *

def train_presence_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loaders: dict,
    model_path: str,
    num_epochs: int = 1000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    presence_threshold: float = 0.0005,  # 0.05%
    patience: int = 10,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    """
    Train the cell type presence detection model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loaders: Dict of validation DataLoaders
        model_path: Path to save the model
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        weight_decay: L2 regularization parameter
        presence_threshold: Threshold for considering a cell type present
        patience: Patience for early stopping
        device: Device to train on
        
    Returns:
        The trained model
    """
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience//2, verbose=True
    )
    
    # Create directory for saving models
    os.makedirs(model_path, exist_ok=True)
    
    # Track training history
    history = defaultdict(list)
    
    # Track best model
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_metrics = defaultdict(float)
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Get batch data
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            true_props = batch['y'].to(device)
            
            # Forward pass
            presence_logits, valid_mask = model(marker_values, coverage)
            
            # Calculate loss
            loss, details = presence_loss_fn(
                presence_logits=presence_logits,
                true_props=true_props,
                valid_mask=valid_mask,
                presence_threshold=presence_threshold,
                device=device
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate metrics
            for metric, value in details.items():
                if isinstance(value, dict):
                    continue
                train_metrics[metric] += value
            
            num_batches += 1
        
        # Average training metrics
        for metric in train_metrics:
            train_metrics[metric] /= num_batches
            history[f"train_{metric}"].append(train_metrics[metric])
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}, "
              f"Specificity: {train_metrics['specificity']:.4f}")
        
        # Validation phase
        model.eval()
        val_metrics = defaultdict(lambda: defaultdict(float))
        
        for val_name, val_loader in val_loaders.items():
            val_set_metrics = defaultdict(float)
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating {val_name}"):
                    # Get batch data
                    marker_values = batch['X'].to(device)
                    coverage = batch['coverage'].to(device)
                    true_props = batch['y'].to(device)
                    
                    # Forward pass
                    presence_logits, valid_mask = model(marker_values, coverage)
                    
                    # Calculate metrics
                    _, details = presence_loss_fn(
                        presence_logits=presence_logits,
                        true_props=true_props,
                        valid_mask=valid_mask,
                        presence_threshold=presence_threshold,
                        device=device
                    )
                    
                    # Accumulate metrics
                    for metric, value in details.items():
                        if isinstance(value, dict):
                            continue
                        val_set_metrics[metric] += value
                    
                    num_val_batches += 1
            
            # Average validation metrics
            for metric in val_set_metrics:
                val_set_metrics[metric] /= num_val_batches
                val_metrics[val_name][metric] = val_set_metrics[metric]
                history[f"val_{val_name}_{metric}"].append(val_set_metrics[metric])
            
            print(f"{val_name} - F1: {val_set_metrics['f1']:.4f}, "
                  f"Precision: {val_set_metrics['precision']:.4f}, "
                  f"Recall: {val_set_metrics['recall']:.4f}, "
                  f"Specificity: {val_set_metrics['specificity']:.4f}")
        
        # Calculate average F1 across validation sets
        avg_val_f1 = np.mean([metrics['f1'] for metrics in val_metrics.values()])
        
        # Update learning rate scheduler
        scheduler.step(avg_val_f1)
        
        # Check for improvement
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'history': dict(history)
            }
            torch.save(checkpoint, os.path.join(model_path, 'best_presence_model.pt'))
            
            print(f"New best model saved! F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_path, 'best_presence_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with F1 {checkpoint['best_val_f1']:.4f}")
    
    return model