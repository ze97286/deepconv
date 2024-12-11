import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
import time 

from deep_conv.deconvolution.preprocess_pats import pats_to_homog
from deep_conv.benchmark.benchmark_utils import *
from deep_conv.benchmark.nnls import run_weighted_nnls
from tqdm import tqdm
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
from deep_conv.deconvolution.experimental_model import *


class TissueDeconvolutionDataset(Dataset):
    def __init__(self, X, coverage, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, idx):
        return {"X": self.X[idx], "coverage": self.coverage[idx], "y": self.y[idx]}
 

class DeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types, distinguishability_df):
        super().__init__()
        self.num_markers = num_markers
        self.num_cell_types = num_cell_types
        self.distinguishability_df = distinguishability_df
        self.concentrations = sorted(distinguishability_df["concentration"].unique())
        
        # Pre-compute weight lookup table during initialization
        self.weight_lookup = {}
        weight_tensors = []
        for cell_type in range(num_cell_types):
            cell_type_data = distinguishability_df[distinguishability_df["cell_type"] == cell_type]
            weights_for_type = []
            for conc in self.concentrations:
                row = cell_type_data[cell_type_data["concentration"] == conc].iloc[0]
                top_markers = eval(row["top_markers"]) if isinstance(row["top_markers"], str) else row["top_markers"]
                n_distinguishable = row["n_distinguishable"]
                weights = torch.zeros(self.num_markers)
                if n_distinguishable > 0:
                    importance = n_distinguishable / len(top_markers)
                    weights[top_markers] = importance
                weights_for_type.append(weights)
            weight_tensors.append(torch.stack(weights_for_type))
        self.weight_lookup = torch.stack(weight_tensors)  # [num_cell_types, num_concentrations, num_markers]
        
        # Much smaller concentration estimator
        self.concentration_estimator = nn.Sequential(
            nn.Linear(num_markers * 2, 64),  # Reduced from 128
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_cell_types),
        )
        
        # Simplified marker importance network
        self.marker_importance = nn.Sequential(
            nn.Linear(num_markers * 2, 64),  # Reduced from 128
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_cell_types * 32),  # Reduced intermediate representation
            nn.LayerNorm(32 * num_cell_types),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_cell_types * 32, num_markers * num_cell_types),
        )
        
        input_size = num_cell_types + num_markers  # weighted_input size + coverage size
        self.features = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.output = nn.Linear(64, num_cell_types)  # Matches features output size
        self.softmax = nn.Softmax(dim=1)

    def get_marker_weights(self, estimated_concentrations):
        device = estimated_concentrations.device
        batch_size = estimated_concentrations.size(0)  
        
        # Move lookup table to correct device if needed
        if self.weight_lookup.device != device:
            self.weight_lookup = self.weight_lookup.to(device)
        
        # Move concentrations to correct device if needed
        if not hasattr(self, 'concentration_values') or self.concentration_values.device != device:
            self.concentration_values = torch.tensor(self.concentrations, device=device)
        
        # Vectorized closest concentration finding
        conc_expanded = estimated_concentrations.unsqueeze(-1)  # [batch, cell_types, 1]
        conc_diff = torch.abs(conc_expanded - self.concentration_values)  # [batch, cell_types, num_concs]
        closest_indices = torch.argmin(conc_diff, dim=-1)  # [batch, cell_types]
        
        cell_indices = torch.arange(self.num_cell_types, device=device).expand(batch_size, -1)
        # Gather weights for the closest concentrations [batch, cell_types, num_markers]
        weights = self.weight_lookup[cell_indices, closest_indices]
        return weights

    def forward(self, X, coverage):
        batch_size = X.shape[0]
        
        # Handle missing values
        valid_mask = ~torch.isnan(X)
        X = torch.where(valid_mask, X, torch.zeros_like(X))
        coverage = coverage * valid_mask.float()
        
        # Combine input features once
        combined_input = torch.cat([X, coverage], dim=1)
        
        # Estimate concentrations
        estimated_concentrations = torch.sigmoid(self.concentration_estimator(combined_input))
        
        # Get marker importance with simplified pathway
        learned_weights = self.marker_importance(combined_input)
        learned_weights = learned_weights.view(batch_size, self.num_cell_types, self.num_markers)
        learned_weights = torch.sigmoid(learned_weights)
        
        # Get distinguishability weights (now fully vectorized)
        distinguishability_weights = self.get_marker_weights(estimated_concentrations.detach())
        marker_weights = learned_weights * distinguishability_weights
        
        # Apply weighted features more efficiently
        weighted_input = (X.unsqueeze(1) * marker_weights).sum(dim=2)  # Shape: [batch, num_cell_types]
        
        # Combine with coverage for feature processing
        feature_input = torch.cat([weighted_input, coverage], dim=1)  # Shape: [batch, num_cell_types + num_markers]
        
        # Process through reduced main pathway
        features = self.features(feature_input)
        logits = self.output(features)
        predictions = self.softmax(logits)
        estimated_concentrations = torch.sigmoid(self.concentration_estimator(combined_input))
        concentration_confidence = torch.sigmoid(5 * estimated_concentrations)
        final_predictions = predictions * concentration_confidence
        
        return final_predictions, estimated_concentrations, marker_weights

class ConcentrationAwareLoss(nn.Module):
    def __init__(self, distinguishability_df, num_markers, num_cell_types, concentration_weight=0.4, distinguishability_weight=0.1):
        super().__init__()
        self.concentration_weight = concentration_weight
        self.distinguishability_weight = distinguishability_weight
        self.num_markers = num_markers
        self.num_cell_types = num_cell_types  # Add this
        
        # Pre-compute all weights as a single tensor
        self.concentrations = torch.tensor(sorted(distinguishability_df['concentration'].unique()))
        cell_types = sorted(distinguishability_df['cell_type'].unique())
        
        # Initialize tensors for pre-computed weights and n_distinguishable
        weights_tensor = []
        n_dist_tensor = []
        
        for cell_type in cell_types:
            cell_weights = []
            cell_n_dist = []
            cell_data = distinguishability_df[distinguishability_df['cell_type'] == cell_type]
            
            for conc in self.concentrations:
                matching_rows = cell_data[cell_data['concentration'] == conc]
                if len(matching_rows) > 0:
                    row = matching_rows.iloc[0]
                    markers = eval(row['top_markers']) if isinstance(row['top_markers'], str) else row['top_markers']
                    n_distinguishable = row['n_distinguishable']
                else:
                    # Handle missing concentration for this cell type
                    markers = []
                    n_distinguishable = 0
                
                weights = torch.zeros(num_markers)
                if n_distinguishable > 0:
                    weights[markers] = 1.0
                
                cell_weights.append(weights)
                cell_n_dist.append(n_distinguishable)
            
            weights_tensor.append(torch.stack(cell_weights))
            n_dist_tensor.append(torch.tensor(cell_n_dist))
        
        self.weights_lookup = torch.stack(weights_tensor)  # [num_cell_types, num_concentrations, num_markers]
        self.n_dist_lookup = torch.stack(n_dist_tensor)    # [num_cell_types, num_concentrations]

    
    def _calculate_distinguishability_loss(self, marker_weights, true_concentrations):
        device = marker_weights.device
        batch_size = true_concentrations.size(0)  # Get batch size from input
        
        # Move tensors to correct device
        if self.weights_lookup.device != device:
            self.weights_lookup = self.weights_lookup.to(device)
            self.n_dist_lookup = self.n_dist_lookup.to(device)
            self.concentrations = self.concentrations.to(device)
        
        # Vectorized concentration matching
        conc_expanded = true_concentrations.unsqueeze(-1)
        conc_diff = torch.abs(conc_expanded - self.concentrations)
        closest_indices = torch.argmin(conc_diff, dim=-1)
        
        cell_indices = torch.arange(self.num_cell_types, device=device).expand(batch_size, -1)
        
        # Gather expected weights and n_distinguishable values
        expected_weights = self.weights_lookup[cell_indices, closest_indices]
        n_dist = self.n_dist_lookup[cell_indices, closest_indices]
        
        importance_factor = torch.log1p(n_dist).unsqueeze(-1)
        loss = torch.mean((marker_weights - expected_weights)**2 * importance_factor)
        
        return loss


    def forward(self, predictions, estimated_concentrations, marker_weights, true_concentrations):
        prediction_loss = F.mse_loss(predictions, true_concentrations)
        concentration_loss = F.mse_loss(estimated_concentrations, true_concentrations)
        distinguishability_loss = self._calculate_distinguishability_loss(marker_weights, true_concentrations)
        
        return (prediction_loss + 
                self.concentration_weight * concentration_loss + 
                self.distinguishability_weight * distinguishability_loss)
    

def visualise_predictions(epoch, predictions, targets, concentrations):
    """
    Plotly visualization to track predictions during training across an entire epoch.
    """
    fig = go.Figure()
    for conc in concentrations:
        # Mask for specific concentration range
        mask = (torch.abs(targets - conc) < 1e-4).any(
            dim=1
        )  # Identify samples near each concentration
        pred_values = (
            predictions[mask].mean(axis=0).cpu().numpy() if mask.sum() > 0 else None
        )
        if pred_values is not None:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(pred_values))),
                    y=pred_values,
                    name=f"Concentration {conc:.4f}",
                )
            )
    fig.update_layout(
        title=f"Predictions at Epoch {epoch}",
        xaxis_title="Cell Types",
        yaxis_title="Mean Prediction",
        barmode="group",
    )
    fig.write_html("/users/zetzioni/sharedscratch/deepconv/data/images/training.html")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,    
    model_path,
    num_epochs=100,
    patience=10,
    lr=1e-4,
):
    """
    Train the concentration-aware deconvolution model

    Args:
        model: ConcentrationAwareDeconvolution model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        distinguishability_df: DataFrame with marker distinguishability data
        model_path: Path to save model checkpoints
        num_epochs: Maximum number of epochs to train
        patience: Early stopping patience
        lr: Learning rate
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    
    os.makedirs(model_path, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    device = next(model.parameters()).device
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_predictions = []
        all_targets = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            X, coverage, y = (
                batch["X"].to(device),
                batch["coverage"].to(device),
                batch["y"].to(device),
            )
            optimizer.zero_grad()
            predictions, estimated_concentrations, marker_weights = model(X, coverage)
            all_predictions.append(predictions.detach())
            all_targets.append(y.detach())
            loss = criterion(predictions, estimated_concentrations, marker_weights, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                X, coverage, y = (
                    batch["X"].to(device),
                    batch["coverage"].to(device),
                    batch["y"].to(device),
                )
                predictions, estimated_concentrations, marker_weights = model(
                    X, coverage
                )
                val_predictions.append(predictions)
                val_targets.append(y)
                loss = criterion(
                    predictions, estimated_concentrations, marker_weights, y
                )
                val_loss += loss.item()
        val_loss /= len(val_loader)
        # Combine predictions and targets for visualization
        all_epoch_predictions = torch.cat(all_predictions)
        all_epoch_targets = torch.cat(all_targets)
        # Visualize predictions at different concentrations
        visualise_predictions(
            epoch, all_epoch_predictions, all_epoch_targets, [0.001, 0.01, 0.05, 0.1]
        )
        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.8f}, "
            f"Val Loss: {val_loss:.8f}"
        )
        # Learning rate scheduling
        scheduler.step(val_loss)
        # Early stopping and model checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, os.path.join(model_path, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    # Load best model before returning
    checkpoint = torch.load(os.path.join(model_path, "best_model.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def predict(model, X, coverage):
    """Prediction function that processes samples one at a time"""
    model.eval()
    device = next(model.parameters()).device
    predictions_list = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i : i + 1], dtype=torch.float32).to(device)
            c = torch.tensor(coverage[i : i + 1], dtype=torch.float32).to(device)
            predictions, _, _ = model(x, c)
            predictions_list.append(predictions.cpu().numpy())
    return np.vstack(predictions_list)

