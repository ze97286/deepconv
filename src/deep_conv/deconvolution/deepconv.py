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
from deep_conv.atlasbuilder.find_marker_candidates import *


def custom_loss(predictions, targets, epoch, alpha=0.25, gamma=2.0):
    # Binary classification loss (focal loss)
    is_high = (targets >= 0.01).float()
    pred_high = (predictions >= 0.01).float()
    bce_loss = F.binary_cross_entropy(pred_high, is_high, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    focal_loss = focal_loss.mean()
    # Weighted regression loss
    high_mask = targets >= 0.01
    weights = torch.where(targets < 0.01, 10.0, 1.0)  # Higher weight for low concentrations
    regression_loss = F.mse_loss(predictions[high_mask], targets[high_mask], reduction='none')
    regression_loss = (regression_loss * weights[high_mask]).mean()
    # Dynamic binary loss weight (gradually reduce over epochs)
    binary_weight = max(10.0 - epoch * 0.1, 1.0)  # Start at 10.0, reduce to 1.0
    return binary_weight * focal_loss + regression_loss


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


class CalibratedDeconvModel(nn.Module):
    def __init__(self, num_markers, num_cell_types):
        super().__init__()
        
        # Core deconvolution module
        self.signal_weights = nn.Parameter(torch.randn(num_markers, num_cell_types))
        self.marker_bias = nn.Parameter(torch.zeros(num_markers))
        
        # Calibration network - purposely simple
        self.calibration = nn.Sequential(
            nn.Linear(num_cell_types, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_cell_types)
        )
        
    def get_base_predictions(self, X, coverage):
        # Handle missing values
        valid_mask = ~torch.isnan(X)
        X = torch.where(valid_mask, X, torch.zeros_like(X))
        coverage = coverage * valid_mask
        X_weighted = X * torch.log1p(coverage)
        
        # Initial signal unmixing (similar to NNLS)
        signal = X_weighted - self.marker_bias.unsqueeze(0)
        raw_proportions = F.linear(signal, self.signal_weights.t())
        
        # Ensure non-negativity and sum-to-one
        base_pred = F.softmax(raw_proportions, dim=1)
        return base_pred
        
    def forward(self, X, coverage):
        # Get base predictions
        base_pred = self.get_base_predictions(X, coverage)
        
        # Calibrate predictions
        calibrated = self.calibration(base_pred)
        final_pred = F.softmax(calibrated, dim=1)
        
        return base_pred, final_pred

def calibrated_loss(predictions, targets, eps=1e-8):
    base_pred, final_pred = predictions
    
    # Base prediction loss - standard cross-entropy
    base_loss = -torch.sum(targets * torch.log(base_pred + eps), dim=1).mean()
    
    # Calibrated prediction losses
    # MSE weighted by concentration range
    mse = (final_pred - targets) ** 2
    
    # Higher weight for critical range (0.5% - 2%)
    mid_range_mask = (targets >= 0.005) & (targets <= 0.02)
    mse[mid_range_mask] *= 5.0
    
    # Highest weight for exact zeros
    zero_mask = targets < 0.001
    mse[zero_mask] *= 10.0
    
    calibrated_loss = mse.mean()
    
    # Add constraint that base predictions should be close to NNLS-like behavior
    constraint_loss = F.mse_loss(base_pred, targets)
    
    total_loss = (
        base_loss * 0.5 +        # Guide base predictions
        calibrated_loss * 1.0 +   # Focus on final calibrated predictions
        constraint_loss * 0.3     # Maintain NNLS-like behavior in base predictions
    )
    
    return total_loss


class CellTypeDeconvolutionModel(nn.Module):
    def __init__(self, num_markers, num_cell_types):
        super().__init__()
        
        # Deeper encoder with skip connections
        self.encoder1 = nn.Sequential(
            nn.Linear(num_markers, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Concentration estimator
        self.regressor = nn.Sequential(
            nn.Linear(1536, 512),  # Fixed: added out_features
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_cell_types)
        )
        
        # Uncertainty estimator
        self.uncertainty = nn.Sequential(
            nn.Linear(1536, 512),  # Fixed: added out_features
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_cell_types),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, X, coverage):
        valid_mask = ~torch.isnan(X)
        X = torch.where(valid_mask, X, torch.zeros_like(X))
        coverage = coverage * valid_mask
        X_weighted = X * torch.log1p(coverage)
        
        # Forward pass with skip connection
        e1 = self.encoder1(X_weighted)
        e2 = self.encoder2(e1)
        
        # Combine features
        combined = torch.cat([e1, e2], dim=1)
        
        # Get predictions and uncertainty
        pred = F.softmax(self.regressor(combined), dim=1)
        uncert = self.uncertainty(combined)
        
        return pred, uncert
    

def improved_loss(predictions, targets, eps=1e-8):
    pred, uncert = predictions
    
    # Ensure predictions and uncertainty are positive and in valid range
    pred = torch.clamp(pred, eps, 1-eps)
    uncert = torch.clamp(uncert, eps, 1.0)
    
    # Basic squared error
    mse = (pred - targets) ** 2
    
    # Weighted MSE based on uncertainty
    weighted_mse = (mse / uncert).mean()
    
    # Uncertainty regularization (prevent too small/large uncertainties)
    uncertainty_reg = uncert.mean()
    
    # Mid-range focus (0.5% - 2%)
    mid_range_mask = (targets >= 0.005) & (targets <= 0.02)
    mid_range_loss = mse[mid_range_mask].mean() if mid_range_mask.any() else torch.tensor(0.0, device=pred.device)
    
    # Zero concentration penalty
    zero_mask = targets < 0.001
    zero_loss = pred[zero_mask].mean() if zero_mask.any() else torch.tensor(0.0, device=pred.device)
    
    total_loss = (
        weighted_mse +
        mid_range_loss * 2.0 +
        zero_loss * 5.0 +
        uncertainty_reg * 0.1
    )
    
    return total_loss


def train_model(model, train_loader, val_loader, model_path, num_epochs=1000, patience=10, lr=1e-4):  # Reduced learning rate
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    os.makedirs(model_path, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        max_grad_norm = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            X, y = batch['X'], batch['y']
            coverage = batch['coverage']
            
            optimizer.zero_grad()
            predictions = model(X, coverage)
            loss = calibrated_loss(predictions, y)
            loss.backward()
            
            # Monitor gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            max_grad_norm = max(max_grad_norm, grad_norm.item())
            
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
                loss = calibrated_loss(predictions, y)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.8f}, "
              f"Val Loss: {val_loss:.8f}, "
              f"Max Grad Norm: {max_grad_norm:.4f}")
        
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
    model.eval()
    # Convert numpy arrays to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    coverage_tensor = torch.tensor(coverage, dtype=torch.float32)
    
    with torch.no_grad():
        base_pred, final_pred = model(X_tensor, coverage_tensor)
        # Return only the final predictions as numpy array
        return final_pred.cpu().numpy()

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_eval(atlas_path, train_pat_dir, eval_pat_dir, min_cpgs, threads, output_path, cell_types):
    set_seed()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if os.path.exists(Path(train_pat_dir)/"marker_values.parquet") and os.path.exists(Path(train_pat_dir)/"coverage.parquet"):
        X_train = pd.read_parquet(Path(train_pat_dir)/"marker_values.parquet")
        coverage_train = pd.read_parquet(Path(train_pat_dir)/"coverage.parquet")
    else:
        X_train, coverage_train = create_marker_matrices(atlas_path, train_pat_dir, min_cpgs, threads)

    X_train = X_train.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_train = coverage_train.drop(columns=["name", "direction"]).T.to_numpy()
    
    if os.path.exists(Path(train_pat_dir)/"ground_truth_y.parquet"):
        y_train = pd.read_parquet(Path(train_pat_dir)/"ground_truth_y.parquet")
    else:
        y_train = get_ground_truth(train_pat_dir,X_train.columns[2:], cell_types)
    y_train = y_train.to_numpy()

    if os.path.exists(Path(eval_pat_dir)/"marker_values.parquet") and os.path.exists(Path(eval_pat_dir)/"coverage.parquet"):
        X_val = pd.read_parquet(Path(eval_pat_dir)/"marker_values.parquet")
        coverage_val = pd.read_parquet(Path(eval_pat_dir)/"coverage.parquet")
    else:
        X_val, coverage_val = create_marker_matrices(atlas_path, eval_pat_dir, min_cpgs,threads)
    X_val = X_val.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_val = coverage_val.drop(columns=["name", "direction"]).T.to_numpy()

    if os.path.exists(Path(eval_pat_dir)/"ground_truth_y.parquet"):
        y_val = pd.read_parquet(Path(eval_pat_dir)/"ground_truth_y.parquet")
    else:
        y_val = get_ground_truth(eval_pat_dir,X_val.columns[2:], cell_types)
    y_val = y_val.to_numpy()

    train_dataset = TissueDeconvolutionDataset(X_train, coverage_train, y_train)
    val_dataset = TissueDeconvolutionDataset(X_val, coverage_val, y_val)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_train = y_train / y_train.sum(dim=1, keepdim=True)
    y_val = y_val / y_val.sum(dim=1, keepdim=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    model = CalibratedDeconvModel(
        num_markers=X_train.shape[1],
        num_cell_types=len(cell_types), 
        # cell_types=list(cell_types)
    )
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1000,
        patience=10,
        model_path=output_path,
    )
    deep_conv_estimation = predict(model, X_val,coverage_val)
    deep_conv_eval_metrics = evaluate_performance(y_val.detach().numpy(), deep_conv_estimation, cell_types)
    print("deepconv validation metrics")
    log_metrics(deep_conv_eval_metrics)    
    return model
    

def eval_dilution(atlas_path, eval_pat_dir, cell_type, model, output_dir, cell_types=None, min_cpgs=None, threads=None):
    atlas = pd.read_csv(atlas_path, sep="\t")
    cell_type_index = list(atlas.columns[8:]).index(cell_type)
    # load marker values, coverage, and ground truth
    if os.path.exists(Path(eval_pat_dir)/"marker_values.parquet") and os.path.exists(Path(eval_pat_dir)/"coverage.parquet"):
        X_val = pd.read_parquet(Path(eval_pat_dir)/"marker_values.parquet").drop(columns=["name", "direction"]).T.to_numpy()
        coverage_val = pd.read_parquet(Path(eval_pat_dir)/"coverage.parquet").drop(columns=["name", "direction"]).T.to_numpy()
        y_val = pd.read_parquet(Path(eval_pat_dir)/"ground_truth_y.parquet").to_numpy()
    else:
        X_val, coverage_val = create_marker_matrices(atlas_path, eval_pat_dir, min_cpgs,threads)
        X_val = X_val.drop(columns=["name", "direction"]).T.to_numpy()
        coverage_val = coverage_val.drop(columns=["name", "direction"]).T.to_numpy()
        y_val = get_ground_truth(eval_pat_dir,X_val.columns[2:], cell_types).to_numpy()
        y_val = y_val.to_numpy()
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_val = y_val / y_val.sum(dim=1, keepdim=True)
    # nnls estimation
    marker_read_proportions, counts = X_val, coverage_val
    nnls_estimation = run_weighted_nnls(marker_read_proportions, counts, atlas[atlas.columns[8:]].T.values)
    nnls_cell_type = nnls_estimation[:, cell_type_index]
    nnls_results_df = pd.DataFrame({"dilution": dilutions(), "contribution": nnls_cell_type})
    nnls_all_predictions_df = pd.DataFrame(nnls_estimation,columns=atlas.columns[8:],index=columns())
    plot_dilution_results(nnls_results_df, cell_type, nnls_all_predictions_df, output_dir+"nnls/") 
    # deepconv estimation
    deep_conv_estimation = predict(model, X_val,coverage_val)
    deep_conv_cell_type = deep_conv_estimation[:, cell_type_index] 
    deepconv_results_df = pd.DataFrame({"dilution": dilutions(), "contribution": deep_conv_cell_type})
    deepconv_all_predictions_df = pd.DataFrame(deep_conv_estimation,columns=atlas.columns[8:],index=columns())
    plot_dilution_results(deepconv_results_df, cell_type, deepconv_all_predictions_df, output_dir+"deepconv/")
     

def train_and_evaluate(model_name):
    atlas_path = "/users/zetzioni/sharedscratch/atlas/atlas/atlas_zohar.blood+gi+tum.l4.bed"
    train_pat_dir = "/users/zetzioni/sharedscratch/atlas/training/general/train/"
    eval_pat_dir = "/users/zetzioni/sharedscratch/atlas/training/TCELLS"
    min_cpgs = 4
    threads = 10
    output_path = Path("/users/zetzioni/sharedscratch/atlas/saved_models/"+model_name+"/")
    atlas = pd.read_csv(atlas_path,sep="\t")
    cell_types = list(atlas.columns[8:])
    return train_and_eval(atlas_path=atlas_path, 
                            train_pat_dir=train_pat_dir, 
                            eval_pat_dir=eval_pat_dir, 
                            min_cpgs=min_cpgs, 
                            threads=threads,
                            output_path=output_path, 
                            cell_types=cell_types)
    

def eval_model(model_name):
    atlas_path = "/users/zetzioni/sharedscratch/atlas/atlas/atlas_zohar.blood+gi+tum.l4.bed"
    atlas = pd.read_csv(atlas_path,sep="\t")
    model = CellTypeDeconvolutionModel(num_markers=len(atlas),num_cell_types=len(atlas.columns[8:]))
    model.load_state_dict(torch.load(f"/users/zetzioni/sharedscratch/atlas/saved_models/{model_name}/best_model.pt"))
    eval_pat_dir4 = "/users/zetzioni/sharedscratch/atlas/training/CD4/"
    eval_pat_dir8 = "/users/zetzioni/sharedscratch/atlas/training/CD8/"
    eval_dilution(
        atlas_path=atlas_path, 
        eval_pat_dir=eval_pat_dir4, 
        cell_type="CD4-T-cells", 
        model=model, 
        output_dir=eval_pat_dir4,
        min_cpgs=4,
        cell_types=list(atlas.columns[8:]),
        threads=10)
    eval_dilution(
            atlas_path=atlas_path, 
            eval_pat_dir=eval_pat_dir8, 
            cell_type="CD8-T-cells", 
            model=model, 
            output_dir=eval_pat_dir8,
            min_cpgs=4,
            cell_types=list(atlas.columns[8:]),
            threads=10)


def plot_analysis(df, cell_types, title):
    """
    Create a combined figure with stacked bar plot and heatmap
    Args:
        df: pandas DataFrame with 'sample' column and cell types
        cell_types: list of cell type columns
        title: str, title for stacked bar plot
    Returns:
        plotly Figure object
    """
    # Create figure with subplots with more spacing
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(title, ""),
        row_heights=[0.35, 0.65],  # Adjusted ratio to push heatmap down
        vertical_spacing=0.25  # Increased vertical spacing
    )

    # Get colors for stacked bars
    colors = px.colors.qualitative.Set3[:len(cell_types)]
    if len(cell_types) > 12:
        colors.extend(px.colors.qualitative.Plotly[:(len(cell_types)-12)])

    # Add stacked bar traces
    for cell_type, color in zip(cell_types, colors):
        fig.add_trace(
            go.Bar(
                name=cell_type,
                x=df['sample'],
                y=df[cell_type],
                marker_color=color,
                hovertemplate=f"{cell_type}: %{{y}}<extra></extra>"
            ),
            row=1, col=1
        )

    # Create matrix for heatmap
    samples = df['sample'].unique()
    matrix = []
    for cell_type in cell_types:
        row = []
        for sample in samples:
            value = df[df['sample'] == sample][cell_type].iloc[0]
            row.append(value)
        matrix.append(row)

    # Custom colorscale for heatmap
    colors_heatmap = [
        [0, 'rgb(0, 0, 255)'],
        [0.2, 'rgb(115, 155, 255)'],
        [0.4, 'rgb(230, 230, 255)'],
        [0.5, 'rgb(255, 255, 255)'],
        [0.6, 'rgb(255, 230, 230)'],
        [0.8, 'rgb(255, 155, 115)'],
        [1, 'rgb(255, 0, 0)']
    ]

    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=samples,
            y=cell_types,
            colorscale=colors_heatmap,
            zmin=0,
            zmax=1,
            hoverongaps=False,
            hovertemplate='Sample: %{x}<br>Cell Type: %{y}<br>Value: %{z:.3f}<extra></extra>',
            colorbar=dict(
                title='Value',
                x=1.02,
                y=0.25,
                len=0.5
            )
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=1300,  # Slightly increased height
        width=1400,
        showlegend=True,
        barmode='stack',
        plot_bgcolor='white',
        legend=dict(
            x=1.02,
            y=0.9,
            xanchor='left',
            yanchor='top'
        ),
        margin=dict(r=150, t=100, b=50)  # Adjusted margins
    )

    # Update xaxis for both plots to show all labels
    fig.update_xaxes(
        tickangle=90,
        title='Sample',
        row=1,
        tickmode='array',
        ticktext=samples,
        tickvals=list(range(len(samples))),
        dtick=1
    )

    fig.update_xaxes(
        tickangle=90,
        title='Sample',
        row=2,
        tickmode='array',
        ticktext=samples,
        tickvals=list(range(len(samples))),
        dtick=1
    )

    # Update yaxis
    fig.update_yaxes(title='Proportion', row=1)
    fig.update_yaxes(title='Cell Type', row=2, autorange='reversed')

    return fig


def eval_OAC(atlas_path, pat_dir, title, prefix, atlas_name, batch, model, type, out_dir,cd_tissue_mapping, model_name=None, min_cpgs=4, threads=10):
    pat_dir = Path(pat_dir)
    atlas = pd.read_csv(atlas_path,sep="\t")
    if os.path.exists(Path(pat_dir)/"marker_values.parquet"):
        X_val = pd.read_parquet(Path(pat_dir)/"marker_values.parquet")
        coverage_val = pd.read_parquet(Path(pat_dir)/"coverage.parquet")
    else:
         X_val, coverage_val = create_marker_matrices(atlas_path, pat_dir, min_cpgs, threads)
         X_val.to_parquet(pat_dir/"marker_values.parquet", index=False)
         coverage_val.to_parquet(pat_dir/"coverage.parquet", index=False)
    samples = list(X_val.columns[2:])
    atlas = atlas.dropna()
    names = set(atlas.name.unique()) 
    X_val = X_val[X_val.name.isin(names)]
    coverage_val = coverage_val[coverage_val.name.isin(names)]
    X_val = X_val.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_val = coverage_val.drop(columns=["name", "direction"]).T.to_numpy()
    if model_name is not None:
        model = CellTypeDeconvolutionModel(num_markers=len(atlas),num_cell_types=len(atlas.columns[8:]), cell_types=list(atlas.columns[8:]))
        model.load_state_dict(torch.load(f"/users/zetzioni/sharedscratch/atlas/saved_models/{model_name}/best_model.pt"))
        estimation = predict(model, X_val,coverage_val)
    else:
        estimation = run_weighted_nnls(X_val, coverage_val, atlas[atlas.columns[8:]].T.values)
    df = pd.DataFrame(estimation, columns=list(atlas.columns[8:]))
    df.rename(columns={"duodenum":"Duodenum"}, inplace=True)
    cols = sorted(list(df.columns))
    df['sample'] = samples    
    def extract_sample(sample):
        sample=sample.split("_plasma")[0]
        sample=sample.split("_md")[0]
        return sample
    def map_sample_index_to_name(sample):
        index = sample.split("-")[-1]
        name=cd_tissue_mapping[index]
        return name
    
    df['sample'] = df['sample'].apply(extract_sample)
    if cd_tissue_mapping is not None:
        df['sample'] = df['sample'].apply(map_sample_index_to_name)
    df['atlas_name'] = atlas_name
    df['batch'] = batch
    df['model'] = model 
    df['type'] = type
    out_dir = Path(out_dir)
    df.to_csv(out_dir/f"{prefix}_deconvolution.csv", sep="\t", index=False)
    fig = plot_analysis(df[cols+['sample']], cols, title)
    fig.write_html(out_dir/f"{prefix}_deconvolution.html")
    

def plot_oac_analysis(df, name):
    # Create figure with custom specs for layout
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
        subplot_titles=('OAC Comparison by Sample', 
                       'ZOHAR_OAC vs ichorCNA', 'BEN_OAC vs ichorCNA'),
        specs=[[{"colspan": 2}, None],
               [{}, {}]]
    )
    
    metrics = ['OAC_nnls', 'OAC_deepconv', 'tf']
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']
    
    # Add bar plots spanning full width
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(name=metric, x=df['sample'], y=df[metric], marker_color=colors[i]),
            row=1, col=1
        )
    
    # Add scatter plots in bottom row
    fig.add_trace(
        go.Scatter(x=df['tf'], y=df['OAC_deepconv'], mode='markers',
                  hovertext=df['sample'], hoverinfo='text+x+y',
                  name='ZOHAR_OAC vs ichorCNA'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['tf'], y=df['OAC_nnls'], mode='markers',
                  hovertext=df['sample'], hoverinfo='text+x+y',
                  name='BEN_OAC vs ichorCNA'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        showlegend=True,
        title_text="OAC Analysis Dashboard",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Sample", row=1, col=1, tickangle=-45)
    fig.update_xaxes(title_text="ichorCNA", row=2, col=1)
    fig.update_xaxes(title_text="ichorCNA", row=2, col=2)
    
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="ZOHAR_OAC", row=2, col=1)
    fig.update_yaxes(title_text="BEN_OAC", row=2, col=2)
    
    # Write to file
    fig.write_html(f"{name}.html")


def run_oac_analysis():
    out_dir = "/users/zetzioni/sharedscratch/atlas/OAC/analysis"

    # cd_tissue_mapping_table = pd.read_csv(out_dir+"/cd_tissue_metadata.tsv",sep="\t")
    # cd_tissue_mapping = dict(zip(cd_tissue_mapping_table['sample_index'].values, cd_tissue_mapping_table['sample'].values))
    cd_tissue_mapping = None
    none_tissue_mapping = None

    zohar_model = "deepconv"
    zohar_model_name="zohar"
    zohar_atlas_path = "/users/zetzioni/sharedscratch/atlas/atlas/atlas_zohar.blood+gi+tum.l4.bed"

    zohar_atlas_name = "atlas_zohar.blood+gi+tum.l4"
    zohar_batch="AB"
    zohar_type = "tissue"
    zohar_prefix_ab_tissue = "deep_conv_ab_tissue"
    zohar_pat_dir_ab_tissue = "/users/zetzioni/sharedscratch/atlas/OAC/atlas_zohar.blood+gi+tum.l4/AB/tissue"
    zohar_title_ab_tissue=f"DeepConv deconvolution using atlas {zohar_atlas_path} on AB tissue"
    eval_OAC(zohar_atlas_path, zohar_pat_dir_ab_tissue, zohar_title_ab_tissue, zohar_prefix_ab_tissue, zohar_atlas_name, zohar_batch, zohar_model, zohar_type, out_dir, none_tissue_mapping, zohar_model_name)

    zohar_type = "cfDNA"
    zohar_prefix_ab_cf = "deep_conv_ab_cfDNA"
    zohar_pat_dir_ab_cf = "/users/zetzioni/sharedscratch/atlas/OAC/atlas_zohar.blood+gi+tum.l4/AB/cfDNA"
    zohar_title_ab_cf=f"DeepConv deconvolution using atlas {zohar_atlas_path} on AB cfDNA"
    eval_OAC(zohar_atlas_path, zohar_pat_dir_ab_cf, zohar_title_ab_cf, zohar_prefix_ab_cf, zohar_atlas_name, zohar_batch, zohar_model, zohar_type, out_dir, none_tissue_mapping, zohar_model_name)

    zohar_batch="CD"
    zohar_prefix_cd_tissue = "deep_conv_cd_tissue"
    zohar_pat_dir_cd_tissue = "/users/zetzioni/sharedscratch/atlas/OAC/atlas_zohar.blood+gi+tum.l4/CD/tissue"
    zohar_title_cd_tissue=f"DeepConv deconvolution using atlas {zohar_atlas_path} on CD tissue"
    eval_OAC(zohar_atlas_path, zohar_pat_dir_cd_tissue, zohar_title_cd_tissue, zohar_prefix_cd_tissue, zohar_atlas_name, zohar_batch, zohar_model, zohar_type, out_dir, cd_tissue_mapping, zohar_model_name)

    zohar_type = "cfDNA"
    zohar_prefix_cd_cf = "deep_conv_cd_cfDNA"
    zohar_pat_dir_cd_cf = "/users/zetzioni/sharedscratch/atlas/OAC/atlas_zohar.blood+gi+tum.l4/CD/cfDNA"
    zohar_title_cd_cf=f"DeepConv deconvolution using atlas {zohar_atlas_path} on CD cfDNA"
    eval_OAC(zohar_atlas_path, zohar_pat_dir_cd_cf, zohar_title_cd_cf, zohar_prefix_cd_cf, zohar_atlas_name, zohar_batch, zohar_model, zohar_type, out_dir, none_tissue_mapping, zohar_model_name)

    ben_model_name = None
    ben_atlas_name = "Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed"
    ben_atlas_path = "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed"

    ben_model = "nnls"
    ben_batch="AB"
    ben_type = "tissue"
    ben_prefix_ab_tissue = "nnls_ab_tissue"
    ben_pat_dir_ab_tissue = "/users/zetzioni/sharedscratch/atlas/OAC/Atlas_dmr_by_read.blood+gi+tum.U100.l4/AB/tissue"
    ben_title_ab_tissue = f"NNLS deconvolution using atlas {ben_atlas_path} on AB tissue"
    eval_OAC(ben_atlas_path, ben_pat_dir_ab_tissue, ben_title_ab_tissue, ben_prefix_ab_tissue, ben_atlas_name, ben_batch, ben_model, ben_type, out_dir, none_tissue_mapping, ben_model_name)

    ben_type = "cfDNA"
    ben_prefix_ab_cf = "nnls_ab_cfDNA"
    ben_pat_dir_ab_cf = "/users/zetzioni/sharedscratch/atlas/OAC/Atlas_dmr_by_read.blood+gi+tum.U100.l4/AB/cfDNA"
    ben_title_ab_cf = f"NNLS deconvolution using atlas {ben_atlas_path} on AB cfDNA"
    eval_OAC(ben_atlas_path, ben_pat_dir_ab_cf, ben_title_ab_cf, ben_prefix_ab_cf, ben_atlas_name, ben_batch, ben_model, ben_type, out_dir, none_tissue_mapping, ben_model_name)

    ben_batch="CD"
    ben_type = "tissue"
    ben_prefix_cd_tissue = "nnls_cd_tissue"
    ben_pat_dir_cd_tissue = "/users/zetzioni/sharedscratch/atlas/OAC/Atlas_dmr_by_read.blood+gi+tum.U100.l4/CD/tissue"
    ben_title_cd_tissue = f"NNLS deconvolution using atlas {ben_atlas_path} on CD tissue"
    eval_OAC(ben_atlas_path, ben_pat_dir_cd_tissue, ben_title_cd_tissue, ben_prefix_cd_tissue, ben_atlas_name, ben_batch, ben_model, ben_type, out_dir, cd_tissue_mapping, ben_model_name)

    ben_type = "cfDNA"
    ben_prefix_cd_cf= "nnls_cd_cfDNA"
    ben_pat_dir_cd_cf = "/users/zetzioni/sharedscratch/atlas/OAC/Atlas_dmr_by_read.blood+gi+tum.U100.l4/CD/cfDNA"
    ben_title_cd_cf = f"NNLS deconvolution using atlas {ben_atlas_path} on CD cfDNA"
    eval_OAC(ben_atlas_path, ben_pat_dir_cd_cf, ben_title_cd_cf, ben_prefix_cd_cf, ben_atlas_name, ben_batch, ben_model, ben_type, out_dir, none_tissue_mapping, ben_model_name)

    # plot AB cohort cfDNA Ben's Atlas with NNLS vs Deepcon with Zohar's atlas OAC concentration vs ichorCNA
    ben_cf_ab = pd.read_csv(out_dir+"/nnls_ab_cfDNA_deconvolution.csv",sep="\t")
    zohar_cf_ab = pd.read_csv(out_dir+"/deep_conv_ab_cfDNA_deconvolution.csv",sep="\t")
    merged_cf_ab_oac = zohar_cf_ab.merge(ben_cf_ab, suffixes=('_deepconv','_nnls'), on='sample')[['sample','OAC_deepconv', 'OAC_nnls']]
    ichorcna_cf_ab = pd.read_csv(out_dir+"/ab_ichorcna_cfdna.csv", sep="\t")
    ichorcna_cf_ab.columns=['sample', 'tf','ploidy']
    merged_cf_ab_oac = merged_cf_ab_oac.merge(ichorcna_cf_ab,on="sample", how="outer").dropna()
    name = out_dir+"/ab_cf_vs_ichorcna"
    plot_oac_analysis(merged_cf_ab_oac, name)

    # plot CD cohort cfDNA Ben's Atlas with NNLS vs Deepcon with Zohar's atlas OAC concentration vs ichorCNA
    ben_cf_cd = pd.read_csv(out_dir+"/nnls_cd_cfDNA_deconvolution.csv",sep="\t")
    zohar_cf_cd = pd.read_csv(out_dir+"/deep_conv_cd_cfDNA_deconvolution.csv",sep="\t")
    merged_cf_cd_oac = zohar_cf_cd.merge(ben_cf_cd, suffixes=('_deepconv','_nnls'), on='sample')[['sample','OAC_deepconv', 'OAC_nnls']]
    ichorcna_cf_cd = pd.read_csv(out_dir+"/cd_ichorcna_cfdna.csv", sep="\t")
    ichorcna_cf_cd.columns=['sample', 'tf','ploidy']
    merged_cf_cd_oac = merged_cf_cd_oac.merge(ichorcna_cf_cd,on="sample", how="outer").dropna()
    name = out_dir+"/cd_cf_vs_ichorcna"
    plot_oac_analysis(merged_cf_cd_oac, name)


def main():
    parser = argparse.ArgumentParser(description="Deep conv")
    parser.add_argument("--atlas_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--pats_path", type=str, required=False)
    parser.add_argument("--min_cpgs", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_threads",required=False, type=int, default=32)
    parser.add_argument("--cd4_path", type=str, required=True)
    parser.add_argument("--cd8_path", type=str, required=True)
    
    args = parser.parse_args()
    
    CELL_TYPES = [
        'B-cells', 
        'CD34-erythroblasts', 
        'CD34-megakaryocytes', 
        'CD4-T-cells', 
        'CD8-T-cells', 
        'Colon', 
        'Duodenum', 
        'Eosinophils', 
        'Esophagus', 
        'Monocytes', 
        'Neutrophils',          
        'NK-cells',
        'OAC', 
        'Pancreas', 
        'Stomach'
    ]
    model = train_and_eval(args.atlas_path, args.train_path+"/train",args.eval_path+"/eval",args.min_cpgs, args.num_threads, args.output_path, CELL_TYPES)
    eval_dilution(
        atlas_path=args.atlas_path, 
        eval_pat_dir=args.cd4_path, 
        cell_type="CD4-T-cells", 
        model=model, 
        output_dir=args.cd4_path,
        min_cpgs=4,
        cell_types=CELL_TYPES,
        threads=args.threads)
    eval_dilution(
        atlas_path=args.atlas_path, 
        eval_pat_dir=args.cd8_path, 
        cell_type="CD8-T-cells", 
        model=model, 
        output_dir=args.cd8_path,
        min_cpgs=4,
        cell_types=CELL_TYPES,
        threads=args.threads)
    
 



if __name__ == "__main__":    
    main()