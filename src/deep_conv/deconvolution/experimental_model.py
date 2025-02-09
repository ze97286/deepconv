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
        return {
            'X': self.X[idx],
            'coverage': self.coverage[idx],
            'y': self.y[idx]
        }


class DeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Linear(num_markers, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output = nn.Linear(256, num_cell_types)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, X, coverage):
        # Convert missing values and their coverage to zeros 
        # to completely remove them from computation
        valid_mask = ~torch.isnan(X)
        X = torch.where(valid_mask, X, torch.zeros_like(X))
        coverage = coverage * valid_mask
        
        # Weight valid measurements by their coverage
        X_weighted = X * torch.log1p(coverage)  # Using log to dampen extreme coverage values
        
        x = self.features(X_weighted)
        logits = self.output(x)
        return self.softmax(logits)
    

def train_model(model, train_loader, val_loader, model_path, num_epochs=100, patience=10, lr=1e-4):
   optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
   os.makedirs(model_path, exist_ok=True)
   best_val_loss = float('inf')
   patience_counter = 0
   for epoch in range(num_epochs):
       model.train()
       train_loss = 0
       all_predictions = []
       all_targets = []
       for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
           X, y = batch['X'], batch['y']
           coverage = batch['coverage']
           optimizer.zero_grad()
           predictions = model(X, coverage)
           all_predictions.append(predictions.detach())
           all_targets.append(y.detach())
           loss = F.mse_loss(predictions, y)
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
           optimizer.step()
           train_loss += loss.item()
       train_loss /= len(train_loader)
       # Validation
       model.eval()
       val_loss = 0
       with torch.no_grad():
           for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
               X, y = batch['X'], batch['y']
               coverage = batch['coverage']
               predictions = model(X, coverage)
               loss = F.mse_loss(predictions, y)
               val_loss += loss.item()
       val_loss /= len(val_loader)
       all_epoch_predictions = torch.cat(all_predictions)
       all_epoch_targets = torch.cat(all_targets)
       # Visualize predictions at different concentrations
       visualise_predictions(
            epoch, all_epoch_predictions, all_epoch_targets, [0.001, 0.01, 0.05, 0.1]
       )
       print(f"Epoch {epoch+1}/{num_epochs}, "
             f"Train Loss: {train_loss:.8f}, "
             f"Val Loss: {val_loss:.8f}")
       scheduler.step(val_loss)
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience_counter = 0
           torch.save(model.state_dict(), os.path.join(model_path, "best_model.pt"))
       else:
           patience_counter += 1
           if patience_counter >= patience:
               print("Early stopping triggered.")
               break
   model.load_state_dict(torch.load(os.path.join(model_path, "best_model.pt")))
   return model


def predict(model, X, coverage):
   """Prediction function that processes samples one at a time"""
   model.eval()
   predictions_list = []
   
   with torch.no_grad():
       for i in range(len(X)):
           x = torch.tensor(X[i:i+1], dtype=torch.float32)
           c = torch.tensor(coverage[i:i+1], dtype=torch.float32)
           pred = model(x, c)
           predictions_list.append(pred.numpy())
   
   return np.vstack(predictions_list)


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
