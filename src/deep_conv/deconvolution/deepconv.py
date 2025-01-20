import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from deep_conv.benchmark.benchmark_utils import *
from deep_conv.benchmark.nnls import run_weighted_nnls
from tqdm import tqdm
import torch.nn.functional as F
import random
import os
import pywt
from deep_conv.atlasbuilder.find_marker_candidates import *


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


class DeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(num_markers, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Binary classifier for >1%
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Regressor for concentration when >1%
        self.regressor = nn.Linear(256, num_cell_types)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, X, coverage):
        valid_mask = ~torch.isnan(X)
        X = torch.where(valid_mask, X, torch.zeros_like(X))
        coverage = coverage * valid_mask
        X_weighted = X * torch.log1p(coverage)
        
        features = self.features(X_weighted)
        is_high = self.classifier(features)
        predictions = self.softmax(self.regressor(features))
        
        # Zero out predictions where classifier says <1%
        return predictions * is_high


def custom_loss(predictions, targets):
    # Binary classification loss
    is_high = (targets >= 0.01).float()
    pred_high = (predictions >= 0.01).float()
    binary_loss = F.binary_cross_entropy(pred_high, is_high)
    
    # Regression loss only for high values
    high_mask = targets >= 0.01
    regression_loss = F.mse_loss(predictions[high_mask], targets[high_mask]) if torch.any(high_mask) else 0.0
    
    return binary_loss * 10.0 + regression_loss


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
           loss = custom_loss(predictions, y)
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
           optimizer.step()
           train_loss += loss.item()
       train_loss /= len(train_loader)
       model.eval()
       val_loss = 0
       with torch.no_grad():
           for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
               X, y = batch['X'], batch['y']
               coverage = batch['coverage']
               predictions = model(X, coverage)
               loss = custom_loss(predictions, y)
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


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_eval(training_path, num_threads, output_path):
    set_seed()
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    data = load_dataset(training_path)
    X_train, y_train, coverage_train = (data["X_train"], data["y_train"], data["coverage_train"])
    train_dataset = TissueDeconvolutionDataset(X_train, coverage_train, y_train)
    X_val, y_val, coverage_val = data["X_val"], data["y_val"], data["coverage_val"]
    val_dataset = TissueDeconvolutionDataset(X_val, coverage_val, y_val)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_train = y_train / y_train.sum(dim=1, keepdim=True)
    y_val = y_val / y_val.sum(dim=1, keepdim=True)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    model = DeconvolutionModel(num_markers=data["X_train"].shape[1],num_cell_types=len(data["cell_types"]))
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1000,
        patience=10,
        model_path=output_path,
    )
    deep_conv_estimation = predict(model, data['X_val'], data['coverage_val'])
    deep_conv_eval_metrics = evaluate_performance(data['y_val'], deep_conv_estimation, data['cell_types'])
    print("deepconv validation metrics")
    log_metrics(deep_conv_eval_metrics)
    nnls_estimation = run_weighted_nnls(data['X_val'], data['coverage_val'], data['reference_profiles'])
    nnls_eval_metrics = evaluate_performance(data['y_val'], nnls_estimation, data['cell_types'])
    print("nnls validation metrics")
    log_metrics(nnls_eval_metrics)
    return model, data
    
def process_pat_file_with_name(regions_df, pat_file, min_cpgs):
    return {'file': pat_file.name, 'result': process_pat_file(pat_file=pat_file, min_cpgs=min_cpgs, regions_df=regions_df)}


def create_marker_matrices(atlas_path: str, pat_dir: str, min_cpgs: int, threads=4) -> tuple[pd.DataFrame, pd.DataFrame]:
   """
   Create marker values matrix and coverage matrix from atlas markers and pat files.
   
   Args:
       atlas_path: Path to atlas file containing markers
       pat_dir: Directory containing pat files
       min_cpgs: Minimum CpGs required for overlap
   
   Returns:
       tuple of (marker_matrix, coverage_matrix) where:
       - Rows are markers from atlas
       - First columns are name, direction
       - Remaining columns are values/coverage for each pat file
   """
   # Read atlas
   print(f"Loading markers from {atlas_path}...")
   markers_df = pd.read_csv(atlas_path, sep='\t')
   # Get pat files
   pat_files = list(Path(pat_dir).glob('*.pat.gz'))
   print(f"Found {len(pat_files)} pat files in {pat_dir}")
   with mp.Pool(threads) as pool:
        process_func = partial(process_pat_file_with_name, markers_df, min_cpgs=min_cpgs)
        results = list(tqdm(
            pool.imap(process_func, pat_files),
            total=len(pat_files),
            desc="Processing pat files"
        ))
    # Sort results by filename
   results = sorted(results, key=lambda x: x['file'])  # or key=lambda x: x[0] if using tuples
    # If you need just the results in order:
   results = [r['result'] for r in results]
   # Create base matrix with name and direction
   base_df = markers_df[['name', 'direction']]
   # Build matrices
   marker_matrix = base_df.copy()
   coverage_matrix = base_df.copy()
   for uxm_df, coverage_df, cell_type in results:
       # Add columns for this pat file's values
       marker_matrix[cell_type] = pd.merge(
           base_df, 
           uxm_df[['name', 'direction', 'value']], 
           on=['name', 'direction'], 
           how='left'
       )['value']
       coverage_matrix[cell_type] = pd.merge(
           base_df, 
           coverage_df[['name', 'direction', 'value']], 
           on=['name', 'direction'], 
           how='left'
       )['value'].fillna(0)
   
   return marker_matrix, coverage_matrix


def generate_mixture_from_atlas(marker_matrix: pd.DataFrame, mixture_props: dict, n_samples: int = 1000,
                              coverage_mean: float = 10, coverage_std: float = 5, 
                              base_noise_std: float = 0.02):
    """
    Generate more realistic mixture data from atlas.
    
    Args:
        marker_matrix: DataFrame with markers' UXM values for each cell type
        mixture_props: Dictionary of cell type -> proportion
        n_samples: Number of mixture samples to generate
        coverage_mean: Mean coverage to simulate
        coverage_std: Standard deviation of coverage
        base_noise_std: Base level of noise
    """
    mixture_markers = marker_matrix[['name', 'direction']].copy()
    mixture_coverage = marker_matrix[['name', 'direction']].copy()
    
    # Calculate base mixed values
    marker_values = np.zeros(len(marker_matrix))
    
    # Add concentration-dependent effects
    for cell_type, prop in mixture_props.items():
        cell_values = marker_matrix[cell_type].values
        
        # Add non-linear effect at low concentrations
        if prop < 0.02:  # For concentrations below 2%
            # Reduce effective signal and increase noise
            effective_prop = prop * (0.8 + 0.2 * np.random.random())  # Random reduction 20-40%
            noise_multiplier = 1 + (0.02 - prop) * 10  # More noise at lower concentrations
        else:
            effective_prop = prop
            noise_multiplier = 1
            
        # Add cell-type specific noise
        cell_noise = np.random.normal(0, base_noise_std * noise_multiplier, size=len(cell_values))
        marker_values += cell_values * effective_prop + cell_noise
    
    # Generate n_samples with varying noise levels
    for i in range(n_samples):
        # Add marker-specific noise (some markers are noisier than others)
        marker_noise = np.random.normal(0, base_noise_std, size=len(marker_values))
        marker_noise *= (1 + np.random.random(size=len(marker_values)))  # Varying noise per marker
        
        noisy_values = marker_values + marker_noise
        
        # Add background interference
        background = np.random.normal(0.02, 0.01, size=len(marker_values))  # Base background level
        background *= (1 + np.random.random(size=len(marker_values)) * 0.5)  # Varying background
        noisy_values += background
        
        # Clip to valid range [0,1]
        noisy_values = np.clip(noisy_values, 0, 1)
        
        mixture_markers[f'mixture_{i+1}'] = noisy_values
        
        # Generate coverage with more realistic distribution
        coverage = np.random.negative_binomial(n=coverage_mean, p=0.5, size=len(marker_values))
        mixture_coverage[f'mixture_{i+1}'] = coverage
    
    return mixture_markers, mixture_coverage


def evaluate_cell_type_concentration(atlas_path, model, mixed_path, cell_type, data, output_path):
    atlas = pd.read_csv(atlas_path,sep="\t").dropna()
    atlas = atlas.drop_duplicates(atlas.columns[8:]).dropna()
    atlas_names = set(atlas.name.unique())
    cell_type_index = list(atlas.columns[8:]).index(cell_type)
    marker_read_proportions, counts = create_marker_matrices(atlas_path,mixed_path, min_cpgs=4, threads=4)
    counts = counts[counts.name.isin(atlas_names)]
    marker_read_proportions = marker_read_proportions[marker_read_proportions.name.isin(atlas_names)]
    marker_read_proportions = marker_read_proportions.drop(columns=["name", "direction"]).T.to_numpy()
    counts = counts.drop(columns=["name", "direction"]).T.to_numpy()
    estimated_val = predict(model, marker_read_proportions, counts)
    cell_type_contribution = estimated_val[:, cell_type_index]
    results_df = pd.DataFrame({"dilution": dilutions(), "contribution": cell_type_contribution})
    metrics = calculate_dilution_metrics(results_df)
    print(metrics)
    os.makedirs(output_path+cell_type, exist_ok=True)
    all_predictions_df = pd.DataFrame(estimated_val,columns=data['cell_types'],index=columns())
    plot_dilution_results(results_df, cell_type, all_predictions_df, output_path+cell_type)
    
    nnls_estimation = run_weighted_nnls(marker_read_proportions, counts, data['reference_profiles'])
    cell_type_contribution = nnls_estimation[:, cell_type_index]
    results_df = pd.DataFrame({"dilution": dilutions(), "contribution": cell_type_contribution})
    metrics = calculate_dilution_metrics(results_df)
    print(metrics)    


def main():
    parser = argparse.ArgumentParser(description="Deep conv")
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--training_path", type=str, required=True)
    parser.add_argument("--pats_path", type=str, required=True)
    parser.add_argument("--wgbs_tools_exec_path", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_threads",required=True, type=int, default=32)
    
    args = parser.parse_args()
    cd4_pats_path = args.pats_path + "cd4"
    cd8_pats_path = args.pats_path + "cd8"
    model, data = train_and_eval(args.training_path, args.num_threads, args.output_path)
    evaluate_cell_type_concentration(
        atlas_path=args.atlas_path, 
        model=model, 
        mixed_path=cd4_pats_path, 
        cell_type="CD4-T-cells", 
        output_path=args.output_path,
        data=data
    )
    # evaluate_cell_type_concentration(
    #     atlas_path=args.atlas_path, 
    #     model=model, 
    #     wgbs_tools_exec_path=args.wgbs_tools_exec_path, 
    #     mixed_path=cd8_pats_path, 
    #     cell_type="CD8-T-cells", 
    #     output_path=args.output_path,
    #     data=data
    # )


if __name__ == "__main__":    
    main()