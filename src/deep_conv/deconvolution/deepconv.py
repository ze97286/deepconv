import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
from deep_conv.deconvolution.preprocess_pats import pats_to_homog
from deep_conv.benchmark.benchmark_utils import *
from deep_conv.benchmark.nnls import run_weighted_nnls
from tqdm import tqdm
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt


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
    

def calculate_marker_importance(reference_profiles):
    """
    Calculate importance score for each marker based on its discriminative power 
    for all cell types
    """
    n_cell_types = reference_profiles.shape[0]
    n_markers = reference_profiles.shape[1]
    
    # For each marker, calculate how well it discriminates between cell types
    marker_importance = np.zeros(n_markers)
    for i in range(n_markers):
        marker_values = reference_profiles[:, i]
        # Calculate pairwise differences between cell types for this marker
        for j in range(n_cell_types):
            for k in range(j+1, n_cell_types):
                diff = abs(marker_values[j] - marker_values[k])
                marker_importance[i] += diff
    
    # Normalize to sum to 1
    return marker_importance / marker_importance.sum()


def train_model(model, train_loader, val_loader, model_path, num_epochs=100, patience=10, lr=1e-4):
   optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
   os.makedirs(model_path, exist_ok=True)
   best_val_loss = float('inf')
   patience_counter = 0
   for epoch in range(num_epochs):
       model.train()
       train_loss = 0
       for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
           X, y = batch['X'], batch['y']
           coverage = batch['coverage']
           optimizer.zero_grad()
           predictions = model(X, coverage)
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


def augment_low_concentration_samples(X, y, coverage, factor=2):
    """
    Augment samples that have any cell types at low concentration
    """
    # Find samples where at least one cell type is < 1%
    low_conc_mask = (y < 0.01).any(axis=1)
    low_conc_indices = np.where(low_conc_mask)[0]
    
    print(f"Total samples: {len(y)}")
    print(f"Samples with any concentration < 1%: {np.sum(low_conc_mask)}")
    print(f"Indices of low conc samples: {len(low_conc_indices)}")
    
    if len(low_conc_indices) == 0:
        return X, y, coverage
        
    X_aug = np.concatenate([X, X[low_conc_indices].repeat(factor-1, axis=0)])
    y_aug = np.concatenate([y, y[low_conc_indices].repeat(factor-1, axis=0)])
    coverage_aug = np.concatenate([coverage, coverage[low_conc_indices].repeat(factor-1, axis=0)])
    
    print(f"Shape after augmentation: {X_aug.shape}")
    return X_aug, y_aug, coverage_aug


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detailed_error_analysis(predictions, y, concentration_ranges=None):
    """
    Detailed analysis of prediction errors across concentration ranges
    
    Args:
        predictions: array/list of predictions for a single cell type
        y: array/list of true values for that cell type
        concentration_ranges: list of tuples (min_conc, max_conc, name)
    """
    # Convert inputs to numpy arrays
    predictions = np.array(predictions)
    y = np.array(y)
    
    if concentration_ranges is None:
        concentration_ranges = [
            (0, 0.0001, "Ultra-low"),
            (0.0001, 0.001, "Very low"),
            (0.001, 0.005, "Low"),
            (0.005, 0.01, "Critical-low"),
            (0.01, 0.05, "Medium"),
            (0.05, 1.0, "High")
        ]
    
    errors = predictions - y
    abs_errors = np.abs(errors)
    
    analysis = {}
    for min_conc, max_conc, name in concentration_ranges:
        mask = (y >= min_conc) & (y < max_conc)
        if np.any(mask):
            true_vals = y[mask]
            pred_vals = predictions[mask]
            range_errors = errors[mask]
            
            analysis[name] = {
                'mean_error': np.mean(range_errors),
                'median_error': np.median(range_errors),
                'std_error': np.std(range_errors),
                'bias': np.mean(range_errors > 0),  # Proportion of overestimates
                'mean_abs_error': np.mean(abs_errors[mask]),
                'mean_true': np.mean(true_vals),
                'mean_pred': np.mean(pred_vals),
                'n_samples': np.sum(mask)
            }
    
    return pd.DataFrame.from_dict(analysis, orient='index')


def plot_marker_analysis(analysis, path):
    """Plot comparison of attended vs discriminative markers"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Marker Analysis by Concentration Range')
    
    for i, (range_name, stats) in enumerate(analysis.items()):
        ax = axes[i//3, i%3]
        
        # Plot correlation between attention and importance
        ax.scatter(stats['attention_weights'], 
                  stats['attended_marker_importance'],
                  alpha=0.5)
        ax.set_title(f"{range_name}\nOverlap: {stats['overlap_count']}/10")
        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('Marker Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, "marker_analysis.png"))
    


def analyze_marker_importance_across_ranges(model, X, coverage, y, reference_profiles, cell_type_idx, concentration_ranges=None):
    """
    Compare model attention vs actual marker importance from reference profiles
    """
    if concentration_ranges is None:
        concentration_ranges = [
            (0, 0.0001, "Ultra-low"),
            (0.0001, 0.001, "Very low"),
            (0.001, 0.005, "Low"),
            (0.005, 0.01, "Critical-low"),
            (0.01, 0.05, "Medium"),
            (0.05, 1.0, "High")
        ]
    # Get model's attention weights
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        coverage_tensor = torch.tensor(coverage, dtype=torch.float32)
        X_masked = torch.nan_to_num(X_tensor, nan=0.0) * (coverage_tensor > 0).float()
        attention_weights = model.attention(X_masked)
        attention_weights = attention_weights.numpy()
    # Get marker differences from reference profiles
    cell_type_profile = reference_profiles[cell_type_idx]
    other_profiles = np.delete(reference_profiles, cell_type_idx, axis=0)
    marker_differences = np.abs(cell_type_profile - np.mean(other_profiles, axis=0))
    analysis = {}
    for min_conc, max_conc, name in concentration_ranges:
        mask = (y >= min_conc) & (y < max_conc)
        range_attention = attention_weights[mask]
        mean_attention = np.mean(range_attention, axis=0)
        top_attended = np.argsort(-mean_attention)[:10]
        top_discriminative = np.argsort(-marker_differences)[:10]
        overlap = len(set(top_attended) & set(top_discriminative))
        analysis[name] = {
            'top_attended_markers': top_attended,
            'top_discriminative_markers': top_discriminative,
            'overlap_count': overlap,
            'attended_marker_importance': marker_differences[top_attended],
            'attention_weights': mean_attention[top_attended],
            'ignored_important_markers': set(top_discriminative) - set(top_attended),
            'marker_differences': marker_differences  # Include this in the analysis dict
        }
    return analysis


def print_marker_analysis(analysis):
    """Print detailed analysis of marker usage"""
    for range_name, stats in analysis.items():
        print(f"\n{range_name} Concentration Range:")
        print("Top attended markers:")
        for i, marker in enumerate(stats['top_attended_markers']):
            importance = stats['attended_marker_importance'][i]
            attention = stats['attention_weights'][i]
            print(f"Marker {marker}: Attention={attention:.3f}, Importance={importance:.3f}")
        
        print("\nIgnored important markers:")
        for marker in stats['ignored_important_markers']:
            importance = stats['marker_differences'][marker]
            print(f"Marker {marker}: Importance={importance:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Deep conv")
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--training_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--pats_path", type=str, required=True)
    parser.add_argument("--wgbs_tools_exec_path", required=True)
    parser.add_argument("--cell_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_threads",required=True, type=int, default=20)
    args = parser.parse_args()
    set_seed()

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(1)
    model_name = args.model_name
    data = load_dataset(args.training_path)

    # X_train, y_train, coverage_train = augment_low_concentration_samples(data['X_train'], data['y_train'], data['coverage_train'], 1)
    X_train, y_train, coverage_train = data['X_train'], data['y_train'], data['coverage_train']
    train_dataset = TissueDeconvolutionDataset(
        X_train, coverage_train, y_train
    )
    # X_val, y_val, coverage_val = augment_low_concentration_samples(data['X_val'], data['y_val'], data['coverage_val'], 1)
    X_val, y_val, coverage_val = data['X_val'], data['y_val'], data['coverage_val']
    val_dataset = TissueDeconvolutionDataset(
        X_val, coverage_val, y_val
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    model = DeconvolutionModel(
        num_markers=data['X_train'].shape[1],
        num_cell_types=len(data['cell_types']),
    )
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1000,
        patience=10,
        model_path="/users/zetzioni/sharedscratch/deepconv/src/deep_conv/saved_models/"+model_name,
    )

    deep_conv_estimation = predict(model, data['X_val'], data['coverage_val'])
    deep_conv_eval_metrics = evaluate_performance(data['y_val'], deep_conv_estimation, data['cell_types'])
    print("deepconv validation metrics")
    log_metrics(deep_conv_eval_metrics)

    nnls_estimation = run_weighted_nnls(data['X_val'], data['coverage_val'], data['reference_profiles'])
    nnls_eval_metrics = evaluate_performance(data['y_val'], nnls_estimation, data['cell_types'])
    print("nnls validation metrics")
    log_metrics(nnls_eval_metrics)

    cell_type_index = data['cell_types'].index(args.cell_type)

    # Predict on test data
    marker_read_proportions, counts = pats_to_homog(
        args.atlas_path, args.pats_path, args.wgbs_tools_exec_path
    )
    df = pd.read_csv("/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed", sep="\t").dropna()
    df = df.drop_duplicates(df.columns[8:]).dropna()    
    atlas_names = set(df.name.unique())
    counts = counts[counts.name.isin(atlas_names)]
    marker_read_proportions = marker_read_proportions[marker_read_proportions.name.isin(atlas_names)]
    marker_read_proportions = marker_read_proportions.drop(columns=['name', 'direction'])[columns()].T.to_numpy()
    counts = counts.drop(columns=['name', 'direction'])[columns()].T.to_numpy()

    estimated_val = predict(model, marker_read_proportions, counts)
    cell_type_contribution = estimated_val[:, cell_type_index]
    results_df = pd.DataFrame({
        "dilution": dilutions(),
        "contribution": cell_type_contribution
    })
    metrics = calculate_dilution_metrics(results_df)
    print(metrics)

    all_predictions_df = pd.DataFrame(
        estimated_val,
        columns=data['cell_types'],
        index=columns()
    )

    plot_dilution_results(
        results_df, args.cell_type, all_predictions_df, args.save_path
    )
    metrics.to_csv(os.path.join(args.save_path, "metrics_out.csv"), index=False)

    cell_type_analysis = detailed_error_analysis(cell_type_contribution, dilutions())
    print(cell_type_analysis.head())
    cell_type_analysis.to_csv(os.path.join(args.save_path, "error_analysis.csv"), index=False)

    analysis = analyze_marker_importance_across_ranges(model, marker_read_proportions, counts, np.array(dilutions()), data['reference_profiles'], 3)
    plot_marker_analysis(analysis, args.save_path)


if __name__ == "__main__":    
    main()