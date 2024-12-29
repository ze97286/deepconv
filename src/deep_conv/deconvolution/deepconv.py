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
import pywt


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


class WaveletDeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types, wavelet='db4', level=3):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        
        # Calculate expected coefficient length after wavelet transform
        dummy_signal = torch.zeros(num_markers)
        coeffs = pywt.wavedec(dummy_signal, wavelet, level=level)
        coeff_len = sum(len(c) for c in coeffs)
        
        self.features = nn.Sequential(
            nn.Linear(coeff_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output = nn.Linear(512, num_cell_types)
        self.softmax = nn.Softmax(dim=1)
        
    def wavelet_transform(self, x):
        # Apply wavelet transform to each sample
        batch_coeffs = []
        for sample in x:
            coeffs = pywt.wavedec(sample.detach().cpu().numpy(), self.wavelet, level=self.level)
            flat_coeffs = np.concatenate([c.flatten() for c in coeffs])
            batch_coeffs.append(flat_coeffs)
        return torch.tensor(np.stack(batch_coeffs), dtype=torch.float32).to(x.device)


    def forward(self, X, coverage):
        valid_mask = ~torch.isnan(X)
        X = torch.where(valid_mask, X, torch.zeros_like(X))
        coverage = coverage * valid_mask
        X_weighted = X * torch.log1p(coverage)
        
        # Transform to wavelet domain
        wavelet_features = self.wavelet_transform(X_weighted)
        
        # Process wavelet coefficients
        features = self.features(wavelet_features)
        return self.softmax(self.output(features))

def wavelet_custom_loss(predictions, targets):
    # Standard MSE in concentration space
    mse_loss = F.mse_loss(predictions, targets)
    
    # Penalty for non-zero predictions below 1%
    low_mask = targets < 0.01
    zero_loss = torch.mean(predictions[low_mask]**2)
    
    # Added wavelet coherence penalty
    batch_size = predictions.shape[0]
    coherence_loss = 0
    for i in range(batch_size):
        pred_wave = pywt.wavedec(predictions[i].detach().cpu().numpy(), 'db4', level=3)
        target_wave = pywt.wavedec(targets[i].detach().cpu().numpy(), 'db4', level=3)
        
        # Compare wavelet coefficients at each level
        level_loss = sum(F.mse_loss(torch.tensor(p), torch.tensor(t)) 
                        for p, t in zip(pred_wave, target_wave))
        coherence_loss += level_loss
    
    coherence_loss /= batch_size
    
    return mse_loss + 5.0 * zero_loss + coherence_loss


def generate_enriched_data_from_df(
    atlas_df,
    compositions,
    n_samples_per_comp=100,
    coverage_mean=30,
    coverage_std=10,
    seed=42
):
    """
    Generate synthetic methylation data using an atlas DataFrame of shape
    (num_markers, num_cell_types) and a list of composition dicts.

    Args:
        atlas_df (pd.DataFrame):
            index=markers, columns=cell types. 
            Each cell is the unmethylation fraction in [0..1].
        compositions (list of dict):
            Each dict: cell_type -> fraction. 
            E.g. {"CD4": 0.01, "Mono":0.24, "Neutro":0.20, ...} 
            sums ~1.0
        n_samples_per_comp (int):
            How many synthetic samples to generate for each composition.
        coverage_mean (float):
            Mean coverage to sample from for each marker.
        coverage_std (float):
            Stddev for coverage distribution across markers.
        seed (int):
            Random seed for reproducibility.

    Returns:
        X (np.ndarray) shape (N, M):
            Unmethylation fraction for each sample (N) and marker (M).
        coverage_arr (np.ndarray) shape (N, M):
            Coverage count for each sample and marker.
        Y (np.ndarray) shape (N, C):
            True fractions for each sample (N) across all cell types (C).
        cell_types (list):
            The list of atlas_df.columns in sorted order (or original order).
    """
    rng = np.random.default_rng(seed)
    # cell_types as list (maintain original or sorted order)
    cell_types = list(atlas_df.columns)
    num_markers = atlas_df.shape[0]  # each row is a marker
    X_list = []
    coverage_list = []
    Y_list = []
    for comp_dict in compositions:
        # Build fraction vector for each cell type
        # cell_types is the reference order
        fraction_vector = np.array([comp_dict.get(ct, 0.0) for ct in cell_types], dtype=float)
        # (Optionally) normalize if needed:
        # if fraction_vector.sum() > 0:
        #     fraction_vector /= fraction_vector.sum()
        for _ in range(n_samples_per_comp):
            # Draw coverage per marker
            cov = rng.normal(loc=coverage_mean, scale=coverage_std, size=num_markers)
            cov = np.clip(np.round(cov), 0, None).astype(int)
            # Weighted average of unmethylation fraction across cell types => final fraction
            # shape: (num_markers,)
            # We sum over cell_types: fraction_vector[i] * atlas_df.iloc[:, i]
            # but we have marker x celltype => we want marker x celltype, for each celltype i -> atlas_df[cell_types[i]]
            # atlas_df for cell_types[i] => shape (num_markers,) 
            weighted_unmeth = np.zeros(num_markers, dtype=float)
            for i, ct in enumerate(cell_types):
                # column in df
                ct_vector = atlas_df[ct].values  # shape (num_markers,)
                weighted_unmeth += fraction_vector[i] * ct_vector
            # Now simulate reads
            synthetic_X = np.zeros(num_markers, dtype=float)
            for m in range(num_markers):
                c_m = cov[m]
                if c_m > 0:
                    num_unmeth = rng.binomial(n=c_m, p=weighted_unmeth[m])
                    synthetic_X[m] = num_unmeth / c_m
                else:
                    synthetic_X[m] = np.nan  # coverage=0 => treat as missing
            X_list.append(synthetic_X)
            coverage_list.append(cov)
            Y_list.append(fraction_vector)
    # Convert to arrays
    X_arr = np.vstack(X_list)               # shape (N, num_markers)
    coverage_arr = np.vstack(coverage_list) # shape (N, num_markers)
    Y_arr = np.vstack(Y_list)               # shape (N, num_cell_types)
    return X_arr, coverage_arr, Y_arr, cell_types


class DeconvolutionModel(nn.Module):
    """
    Multi-bin approach to handle distinct fraction ranges:
       Bin A: <0.5%
       Bin B: [0.5%, 2%)
       Bin C: >=2%
    We do a 3-way classification for each cell type, plus
    3 separate regressors (A,B,C). 
    """

    def __init__(self, num_markers, num_cell_types):
        super().__init__()
        self.num_cell_types = num_cell_types
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(num_markers, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3-way classifier for each cell type => shape (batch, 3*C)
        # We'll reshape to (batch, C, 3) and use CrossEntropy in the loss
        self.classifier = nn.Linear(256, num_cell_types * 3)
        
        # One regressor per bin
        # For each bin's regressor, we output shape (batch, C)
        # Using Sigmoid -> final range [0..1]. You can also do ReLU or Beta transform
        self.regressor_binA = nn.Sequential(
            nn.Linear(256, num_cell_types),
            nn.Sigmoid()
        )
        self.regressor_binB = nn.Sequential(
            nn.Linear(256, num_cell_types),
            nn.Sigmoid()
        )
        self.regressor_binC = nn.Sequential(
            nn.Linear(256, num_cell_types),
            nn.Sigmoid()
        )

    def forward(self, X, coverage):
        """
        1) coverage weighting
        2) classifier => (batch, C, 3)
        3) regressors => binA/binB/binC => each (batch, C)
        """
        valid_mask = ~torch.isnan(X)
        X_filled = torch.where(valid_mask, X, torch.zeros_like(X))
        coverage_filled = coverage * valid_mask
        
        # Weighted by log(1+coverage)
        X_weighted = X_filled * torch.log1p(coverage_filled)
        
        # Shared features
        feats = self.features(X_weighted)
        
        # Classifier raw logits => shape (batch, 3*C)
        raw_logits = self.classifier(feats)
        # Reshape to (batch, C, 3) so we can do cross_entropy easily
        # or do so in the loss function
        # We'll just return raw_logits as is; the loss will handle it
        # But we can also reshape here if we want
        # raw_logits_reshaped = raw_logits.view(-1, self.num_cell_types, 3)
        
        # Regressors for each bin
        outA = self.regressor_binA(feats)  # (batch, C)
        outB = self.regressor_binB(feats)  # (batch, C)
        outC = self.regressor_binC(feats)  # (batch, C)
        
        # Return everything; the custom loss + predict function decide how to use them
        return raw_logits, outA, outB, outC


def multi_bin_loss(
    raw_logits,    # shape (batch, 3*C)
    outA, outB, outC,  # each (batch, C)
    targets,       # (batch, C)
    boundary1=0.005,   # e.g. 0.5%
    boundary2=0.02,    # e.g. 2%
    classification_weight=10.0,
    regression_weight=1.0
):
    """
    We define bins:
      Bin A: < boundary1 (0.5%)
      Bin B: [boundary1, boundary2) => [0.5%, 2%)
      Bin C: >= boundary2 (2%)
    
    1) cross-entropy over 3 bins
    2) MSE only for the correct bin's regressor
    """

    # 1) Construct bin labels
    # bin_label[i,j] in {0,1,2}
    # shape => same as targets => (batch, C)
    binA_mask = (targets < boundary1)
    binB_mask = (targets >= boundary1) & (targets < boundary2)
    binC_mask = (targets >= boundary2)
    
    # We'll store as integer label
    bin_label = torch.zeros_like(targets, dtype=torch.long)  # default 0 => binA
    bin_label[binB_mask] = 1
    bin_label[binC_mask] = 2

    # Flatten for cross-entropy: we have batch*C samples, each with 3 classes
    batch_size, C = targets.shape
    logits_reshaped = raw_logits.view(batch_size * C, 3)  # (batch*C, 3)
    bin_label_flat = bin_label.view(batch_size * C)       # (batch*C)
    
    # 2) cross-entropy
    # We'll rely on torch.cross_entropy with raw logits
    ce_loss = F.cross_entropy(logits_reshaped, bin_label_flat, reduction='mean')

    # 3) MSE for the correct bin
    # flatten outA,outB,outC => each shape (batch*C,)
    outA_flat = outA.view(batch_size * C)
    outB_flat = outB.view(batch_size * C)
    outC_flat = outC.view(batch_size * C)
    targets_flat = targets.view(batch_size * C)
    
    # We'll pick the correct regressor for each sample
    # i.e. if bin_label_flat=0 => use outA_flat
    mseA = 0.0
    if (bin_label_flat == 0).any():
        mseA = F.mse_loss(
            outA_flat[bin_label_flat == 0],
            targets_flat[bin_label_flat == 0]
        )
    mseB = 0.0
    if (bin_label_flat == 1).any():
        mseB = F.mse_loss(
            outB_flat[bin_label_flat == 1],
            targets_flat[bin_label_flat == 1]
        )
    mseC = 0.0
    if (bin_label_flat == 2).any():
        mseC = F.mse_loss(
            outC_flat[bin_label_flat == 2],
            targets_flat[bin_label_flat == 2]
        )
    
    # Combine them
    # We can do a simple average or sum them
    # This is up to you. We'll do an average for fairness:
    bins_used = 0
    if (bin_label_flat == 0).any():
        bins_used += 1
    if (bin_label_flat == 1).any():
        bins_used += 1
    if (bin_label_flat == 2).any():
        bins_used += 1
    if bins_used == 0:
        mse_total = 0.0
    else:
        mse_total = (mseA + mseB + mseC) / bins_used

    total_loss = classification_weight * ce_loss + regression_weight * mse_total
    return total_loss


def train_model(
    model,
    train_loader,
    val_loader,
    model_path,
    num_epochs=100,
    patience=10,
    lr=1e-4
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    os.makedirs(model_path, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            X = batch['X']          # (batch, num_markers)
            y = batch['y']          # (batch, num_cell_types)
            coverage = batch['coverage']

            optimizer.zero_grad()
            raw_logits, outA, outB, outC = model(X, coverage)
            
            loss = multi_bin_loss(
                raw_logits=raw_logits,
                outA=outA, outB=outB, outC=outC,
                targets=y,
                boundary1=0.005,  # 0.5%
                boundary2=0.02,   # 2%
                classification_weight=10.0,
                regression_weight=1.0
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                X = batch['X']
                y = batch['y']
                coverage = batch['coverage']

                raw_logits, outA, outB, outC = model(X, coverage)
                loss = multi_bin_loss(
                    raw_logits=raw_logits,
                    outA=outA, outB=outB, outC=outC,
                    targets=y,
                    boundary1=0.005,
                    boundary2=0.02,
                    classification_weight=10.0,
                    regression_weight=1.0
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} => Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_path, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best
    model.load_state_dict(torch.load(os.path.join(model_path, "best_model.pt")))
    return model


def predict(model, X, coverage):
    """
    1) Classify each cell type into bin A,B, or C (3-way)
    2) Use the corresponding regressor's output as final fraction
    """
    model.eval()
    predictions_list = []
    
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i:i+1], dtype=torch.float32)
            c = torch.tensor(coverage[i:i+1], dtype=torch.float32)
            raw_logits, outA, outB, outC = model(x, c)
            
            # shape (1, 3*C)
            # reshape => (1, C, 3)
            # argmax over dim=2 => bin index
            batch_size, c_dim_3 = raw_logits.shape
            c_dim = c_dim_3 // 3
            logits_reshaped = raw_logits.view(batch_size, c_dim, 3)
            
            # shape (1, C)
            bin_indices = torch.argmax(logits_reshaped, dim=2)  # => (1, C)
            
            # pick from outA,outB,outC
            row_preds = []
            for cell_idx in range(c_dim):
                bin_idx = bin_indices[0, cell_idx].item()
                if bin_idx == 0:
                    val = outA[0, cell_idx].item()  # Bin A
                elif bin_idx == 1:
                    val = outB[0, cell_idx].item()  # Bin B
                else:
                    val = outC[0, cell_idx].item()  # Bin C
                row_preds.append(val)
            predictions_list.append(row_preds)
    
    return np.array(predictions_list) 


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
    pd.read_csv("/users/zetzioni/sharedscratch/atlas/atlas_dmr_by_read.blood+gi+tum.U100.l4.bed", sep="\t").dropna()
    df = df.drop_duplicates(df.columns[8:]).dropna()    
    compositions = [
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD4-T-cells": 0.0},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD4-T-cells": 0.001},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD4-T-cells": 0.005},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD4-T-cells": 0.01},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD8-T-cells": 0.0},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD8-T-cells": 0.001},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD8-T-cells": 0.005},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD8-T-cells": 0.01},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD4-T-cells": 0.0,"CD8-T-cells": 0.0},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD4-T-cells": 0.0005,"CD8-T-cells": 0.0005},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD4-T-cells": 0.0025,"CD8-T-cells": 0.0025},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.24, "Neutrophils":0.20, "CD4-T-cells": 0.005,"CD8-T-cells": 0.005},
        {"CD34-erythroblasts":0.15, "CD34-megakaryocytes":0.40, "Monocytes":0.23, "Neutrophils":0.20, "CD4-T-cells": 0.01,"CD8-T-cells": 0.01},
    ]

    # X_train, y_train, coverage_train = augment_low_concentration_samples(data['X_train'], data['y_train'], data['coverage_train'], 1)
    X_train, y_train, coverage_train = data['X_train'], data['y_train'], data['coverage_train']
    X_syn, coverage_syn, Y_syn, _ = generate_enriched_data_from_df(
        atlas_df=df[df.columns[8:]],
        compositions=compositions,
        n_samples_per_comp=200,
        coverage_mean=5,
        coverage_std=5,
        seed=42
    )
    X_train = np.vstack([X_train, X_syn])
    coverage_train = np.vstack([coverage_train, coverage_syn])
    y_train = np.vstack([y_train, Y_syn])

    train_dataset = TissueDeconvolutionDataset(
        X_train, coverage_train, y_train
    )
    # X_val, y_val, coverage_val = augment_low_concentration_samples(data['X_val'], data['y_val'], data['coverage_val'], 1)
    X_val, y_val, coverage_val = data['X_val'], data['y_val'], data['coverage_val']
    X_val_syn,cov_val_syn,Y_val_syn,_ = generate_enriched_data_from_df(
        atlas_df=df[df.columns[8:]],
        compositions=compositions, 
        n_samples_per_comp=100,
        coverage_mean=10,
        coverage_std=5, 
        seed=43)
    X_val = np.vstack([X_val, X_val_syn])
    coverage_val = np.vstack([coverage_val, cov_val_syn])
    y_val = np.vstack([y_val, Y_val_syn])
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