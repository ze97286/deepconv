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


class ConcentrationAwareLoss(nn.Module):
    def __init__(
        self,
        distinguishability_df,
        concentration_weight=0.3,
        distinguishability_weight=0.2,
    ):
        super().__init__()
        self.concentration_weight = concentration_weight
        self.distinguishability_weight = distinguishability_weight
        self.distinguishability_df = distinguishability_df
    
    def forward(
        self, predictions, estimated_concentrations, marker_weights, true_concentrations
    ):
        # Main prediction loss
        prediction_loss = nn.MSELoss()(predictions, true_concentrations)
        # Concentration estimation loss
        concentration_loss = nn.MSELoss()(estimated_concentrations, true_concentrations)
        # Marker weight consistency loss
        distinguishability_loss = self._calculate_distinguishability_loss(
            marker_weights, true_concentrations
        )
        total_loss = (
            prediction_loss
            + self.concentration_weight * concentration_loss
            + self.distinguishability_weight * distinguishability_loss
        )
        return total_loss
    
    def _calculate_distinguishability_loss(self, marker_weights, true_concentrations):
        """
        Calculate loss term encouraging marker weights to align with known distinguishability
        """
        batch_size = true_concentrations.shape[0]
        loss = 0.0
        for i in range(batch_size):
            for cell_type in range(true_concentrations.shape[1]):
                # Get true concentration for this sample and cell type
                conc = true_concentrations[i, cell_type].item()
                # Find the closest concentration in our distinguishability data
                cell_type_data = self.distinguishability_df[
                    self.distinguishability_df["cell_type"] == cell_type
                ]
                closest_conc_idx = (
                    (cell_type_data["concentration"] - conc).abs().idxmin()
                )
                row = cell_type_data.loc[closest_conc_idx]
                # Get the top markers and convert from string if needed
                top_markers = row["top_markers"]
                if isinstance(top_markers, str):
                    top_markers = eval(top_markers)
                n_distinguishable = row["n_distinguishable"]
                if n_distinguishable > 0:
                    # Create expected weights: 1 for top markers, 0 for others
                    expected_weights = torch.zeros_like(marker_weights[i, cell_type])
                    expected_weights[top_markers] = 1.0
                    # Calculate loss only for cases where we have distinguishable markers
                    weights = marker_weights[i, cell_type]
                    # Higher weight for matching top markers when n_distinguishable is higher
                    importance_factor = torch.log1p(torch.tensor(n_distinguishable))
                    # Calculate weighted MSE for this cell type
                    cell_loss = (
                        nn.MSELoss()(weights, expected_weights) * importance_factor
                    )
                    loss += cell_loss
        return loss / batch_size


class DeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types, distinguishability_df):
        super().__init__()
        self.num_markers = num_markers
        self.num_cell_types = num_cell_types
        self.distinguishability_df = distinguishability_df
        self.concentrations = sorted(distinguishability_df["concentration"].unique())
        # These layers remain the same
        self.concentration_estimator = nn.Sequential(
            nn.Linear(num_markers * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_cell_types),
        )
        self.marker_importance = nn.Sequential(
            nn.Linear(num_markers * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_markers * num_cell_types),
        )
        # Update features layer to accept correct input size
        input_size = num_markers * num_cell_types + num_markers  # weighted_input + coverage
        self.features = nn.Sequential(
            nn.Linear(input_size, 512),  # Modified this line
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.output = nn.Linear(256, num_cell_types)
        self.softmax = nn.Softmax(dim=1)
    
    def get_marker_weights(self, estimated_concentrations):
        batch_size = estimated_concentrations.shape[0]
        weights = torch.zeros(
            batch_size,
            self.num_cell_types,
            self.num_markers,
            device=estimated_concentrations.device,
        )
        for i in range(batch_size):
            for cell_type in range(self.num_cell_types):
                conc = estimated_concentrations[i, cell_type].item()
                # Find closest concentration in data
                cell_type_data = self.distinguishability_df[
                    self.distinguishability_df["cell_type"] == cell_type
                ]
                closest_conc_idx = (
                    (cell_type_data["concentration"] - conc).abs().idxmin()
                )
                row = cell_type_data.loc[closest_conc_idx]
                # Get top markers and convert string representation to list if needed
                top_markers = row["top_markers"]
                if isinstance(top_markers, str):
                    top_markers = eval(
                        top_markers
                    )  # Convert string representation to list
                n_distinguishable = row["n_distinguishable"]
                if n_distinguishable > 0:
                    # Weight importance by number of distinguishable markers
                    importance = n_distinguishable / len(top_markers)
                    weights[i, cell_type, top_markers] = importance
        return weights
    
    def forward(self, X, coverage):
        batch_size = X.shape[0]
        valid_mask = ~torch.isnan(X)
        X = torch.where(valid_mask, X, torch.zeros_like(X))
        coverage = coverage * valid_mask.float()
        combined_input = torch.cat([X, coverage], dim=1)
        estimated_concentrations = torch.sigmoid(self.concentration_estimator(combined_input))
        learned_weights = self.marker_importance(combined_input)
        learned_weights = learned_weights.view(batch_size, self.num_cell_types, self.num_markers)
        learned_weights = torch.sigmoid(learned_weights)
        distinguishability_weights = self.get_marker_weights(estimated_concentrations.detach())
        distinguishability_weights = distinguishability_weights.to(learned_weights.device)
        marker_weights = learned_weights * distinguishability_weights
        weighted_input = X.unsqueeze(1) * marker_weights
        weighted_input = weighted_input.view(batch_size, -1)
        features = self.features(torch.cat([weighted_input, coverage], dim=1))
        logits = self.output(features)
        predictions = self.softmax(logits)
        final_predictions = (predictions + estimated_concentrations) / 2
        return final_predictions, estimated_concentrations, marker_weights


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
    distinguishability_df,
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
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = ConcentrationAwareLoss(distinguishability_df)
    os.makedirs(model_path, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    device = next(model.parameters()).device
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        # Store predictions for visualization
        all_predictions = []
        all_targets = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            start = time.time()
            X, coverage, y = (
                batch["X"].to(device),
                batch["coverage"].to(device),
                batch["y"].to(device),
            )
            optimizer.zero_grad()
            print("forward pass")
            predictions, estimated_concentrations, marker_weights = model(X, coverage)
            # Store predictions and targets for visualization
            all_predictions.append(predictions.detach())
            all_targets.append(y.detach())
            print("calculating loss")
            loss = criterion(predictions, estimated_concentrations, marker_weights, y)
            print("calculated loss")
            loss.backward()
            print("finished back pass")
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            print("finished optimisation step")
            train_loss += loss.item()
            print("batch took", time.time()-start)
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

