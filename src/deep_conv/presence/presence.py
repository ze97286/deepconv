import argparse
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple 

from deep_conv.benchmark.benchmark_utils import *
from deep_conv.presence.model import *
from deep_conv.presence.train import *
from deep_conv.presence.evaluate import *

from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots



def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_validation_set(eval_pat_dir: str, atlas: pd.DataFrame,target_cell_type:int, names: set) -> Tuple[DataLoader, torch.Tensor]:
    """
    Reads marker coverage, methylation, and ground-truth label files from a validation set directory,
    filters them down to the set of markers in 'names', and returns a DataLoader plus normalized labels.

    Args:
        eval_pat_dir (str):
            Directory containing "marker_values.parquet", "coverage.parquet", and "ground_truth_y.parquet" 
            for the validation data.
        atlas (pd.DataFrame):
            A DataFrame that includes marker metadata and cell-type columns. 
            We use atlas.columns[8:] as the cell-type columns for the dataset constructor.
        names (set):
            The set of marker names (strings) to include, ensuring we only keep markers that appear
            in both 'atlas' and the parquet files.

    Returns:
        val_loader (DataLoader):
            A DataLoader wrapping the TissueDeconvolutionDataset for the validation set, 
            with batch_size=512.
        y_val (torch.Tensor):
            A [N, C] Tensor of ground-truth cell-type proportions, normalized so each row sums to 1.
    """
    # Load marker values, coverage, and labels from parquet
    X_val = pd.read_parquet(Path(eval_pat_dir) / "marker_values.parquet")
    coverage_val = pd.read_parquet(Path(eval_pat_dir) / "coverage.parquet")
    y_val = pd.read_parquet(Path(eval_pat_dir) / "ground_truth_y.parquet")

    # Filter to only include markers in 'names'
    X_val = X_val[X_val.name.isin(names)]
    coverage_val = coverage_val[coverage_val.name.isin(names)]
    
    # Drop name/direction columns and transpose => shape [samples, markers]
    X_val = X_val.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_val = coverage_val.drop(columns=["name", "direction"]).T.to_numpy()

    # Print some coverage stats for debug/monitoring
    print("median coverage", 
          np.median(coverage_val, axis=1), 
          "median of medians", 
          np.median(np.median(coverage_val, axis=1)), 
          "mean median", 
          np.median(coverage_val, axis=1).mean())
    
    y_val = y_val.to_numpy()
    
    val_dataset = BinaryCellTypeDataset(
        fraction=X_val,
        coverage=coverage_val,
        y=y_val,
        # target_ids=list(atlas.columns[8:]),
        target_cell_type=target_cell_type,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=4,
        persistent_workers=True,
        shuffle=False
    )
    
    # Convert y_val to a PyTorch tensor and normalize each row
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_val = y_val / y_val.sum(dim=1, keepdim=True)
    
    return val_loader, y_val


def load_training(base_dir: str, atlas: pd.DataFrame, names: set, target_cell_type: int, num_files: int = 4) -> DataLoader:
    """
    Loads and merges multiple parquet files containing training data (marker_values, coverage, ground_truth_y),
    filters them to only include the markers in 'names', and returns a DataLoader for training.

    Args:
        base_dir (str):
            Path prefix for the training parquet files. We expect files named like:
                base_dir + "1_marker_values.parquet",
                base_dir + "1_coverage.parquet",
                base_dir + "1_ground_truth_y.parquet",
                and so on up to num_files.
        atlas (pd.DataFrame):
            DataFrame with marker metadata plus columns for each cell type (atlas.columns[8:] are cell types).
        names (set):
            The set of marker names to keep (usually matches the set in the atlas).
        num_files (int):
            Number of parquet file batches to merge. Defaults to 4.

    Returns:
        DataLoader:
            A DataLoader over the merged training dataset with batch_size=256, shuffle=True, etc.
    """
    markers = []
    coverage = []
    y = []
    
    print("loading training from", base_dir)
    suffixes = [f"_batch{i}" for i in range(1, num_files + 1)]
    
    # Read multiple parquet files and accumulate marker values, coverage, and ground-truth
    for i in range(1, num_files + 1):
        markers.append(pd.read_parquet(base_dir + str(i) + "_marker_values.parquet"))
        coverage.append(pd.read_parquet(base_dir + str(i) + "_coverage.parquet"))
        y.append(pd.read_parquet(base_dir + str(i) + "_ground_truth_y.parquet"))
    
    # Merge all marker tables on ['name','direction']
    merged_markers = markers[0]
    for i, m in enumerate(markers[1:]):
        merged_markers = merged_markers.merge(
            m,
            on=['name', 'direction'],
            how='outer',
            suffixes=('', suffixes[i])
        )
    
    # Merge all coverage tables on ['name','direction']
    merged_coverage = coverage[0]
    for i, c in enumerate(coverage[1:]):
        merged_coverage = merged_coverage.merge(
            c,
            on=['name', 'direction'],
            how='outer',
            suffixes=('', suffixes[i])
        )
    
    # Concatenate all label DataFrames
    y = pd.concat(y, ignore_index=True).fillna(0)
    
    # Filter out any markers not in 'names'
    X_train = merged_markers[merged_markers.name.isin(names)]
    coverage_train = merged_coverage[merged_coverage.name.isin(names)]
    
    # Drop unnecessary columns and transpose => shape [samples, markers]
    X_train = X_train.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_train = coverage_train.drop(columns=["name", "direction"]).T.to_numpy()
    
    # Print coverage stats for debug
    print("median coverage", 
          np.median(coverage_train, axis=1), 
          "median of medians", 
          np.median(np.median(coverage_train, axis=1)), 
          "mean median", 
          np.median(coverage_train, axis=1).mean())
    
    # Convert the label DataFrame to numpy
    y_train = y.to_numpy()
    
    # Build a TissueDeconvolutionDataset
    train_dataset = BinaryCellTypeDataset(
        fraction=X_train,
        coverage=coverage_train,
        y=y_train,
        target_cell_type=target_cell_type,
        # target_ids=list(atlas.columns[8:]),
    )
    
    # Also create a normalized version of y for potential usage
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_train = y_train / y_train.sum(dim=1, keepdim=True)
    
    # Return a DataLoader for training
    return DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,  
        num_workers=4,
        persistent_workers=True
    )
   

def train_and_eval(
    atlas_path: str,
    train_pat_dir: str,
    eval_pat_dir: str,
    threads: int,
    output_path: str,  
) -> nn.Module:
    # Fix random seeds and threads for reproducibility
    set_seed()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    # 1) Read the atlas of markers and cell types
    atlas = pd.read_csv(atlas_path, sep="\t")
    cell_types = list(atlas.columns[8:])
    for target_cell_type_name in cell_types:
        print("training presence model for",target_cell_type_name)
        # The 'names' set ensures we only keep relevant markers
        names = set(atlas[atlas.target==target_cell_type_name].name.unique())
        target_cell_type=cell_types.index(target_cell_type_name)
        # 2) Build the training DataLoader from parquet files in train_pat_dir
        train_dl = load_training(train_pat_dir, atlas, names, target_cell_type=target_cell_type)
        # 3) Build DataLoaders for each validation subset
        tier1_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "tier1"), atlas, target_cell_type, names)
        tier2_dl = None
        if target_cell_type_name=="OAC":
            tier2_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "OAC"), atlas, target_cell_type, names)
        elif target_cell_type_name=="CD4-T-cells":
            tier2_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "CD4"), atlas, target_cell_type, names)
        elif target_cell_type_name=="CD8-T-cells":
            tier2_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "CD8"), atlas, target_cell_type, names)

        validation_dls = {
            "tier1": tier1_dl,        
        }
        if not tier2_dl is None:
            validation_dls['tier2'] = tier2_dl

        single_model = SingleCellTypePresenceModel(
            num_markers=len(atlas),       
        )

        # Train it
        trained_model = train_binary_classifier(
            model=single_model,
            dataloaders={"train":train_dl, "val": validation_dls},
            model_path=output_path,
            num_epochs=100,
            learning_rate=1e-3
            target_cell_type=target_cell_type_name
        )

        results_df = analyze_detection_by_concentration(trained_model, tier2_dl, output_path, target_cell_type_name)
        find_minimum_detection_concentration(results_df, target_cell_type_name, output_path)


def analyze_detection_by_concentration(model, dataloader, 
                                       output_path,
                                       target_cell_type,
                                      concentration_groups=None,
                                      threshold=0.5,
                                      device=None):
    """
    Analyze the model's detection performance across different concentration levels using Plotly visualizations.
    
    Args:
        model: Binary classifier model
        dataloader: DataLoader containing samples with concentration information
        concentration_groups: Dictionary mapping group names to concentration ranges
        threshold: Decision threshold for binary classification
        device: Device to run on
        
    Returns:
        results_df: DataFrame with detection results by sample
        group_stats: DataFrame with detection statistics by concentration group
        threshold_mapping: Dictionary mapping concentration ranges to optimal thresholds
    """
    if device is None:
        device = next(model.parameters()).device
    
    if concentration_groups is None:
        # Default concentration groups if not provided
        concentration_groups = {
            'very_high': (0.2, 1.0),    # 20% to 100%
            'high': (0.05, 0.2),        # 5% to 20%
            'medium': (0.01, 0.05),     # 1% to 5%
            'low': (0.001, 0.01),       # 0.1% to 1%
            'very_low': (0.0005, 0.001), # 0.05% to 0.1%
            'ultra_low': (0.0001, 0.0005) # 0.001% to 0.05%
        }
    
    model.eval()
    
    # Store results for each sample
    results = []
    
    # Evaluate
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing concentration detection"):
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            labels = batch['label'].to(device).view(-1, 1)
            
            # Get concentrations
            if 'concentration' in batch:
                concentrations = batch['concentration'].cpu().numpy()
            else:
                # If no concentration provided, use label as binary indicator
                concentrations = labels.cpu().numpy()
            
            # Forward pass
            logits, attention_weights = model(marker_values, coverage)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions = (probabilities >= threshold).astype(int)
            
            # Store results for each sample
            for i in range(len(concentrations)):
                results.append({
                    'concentration': concentrations[i],
                    'probability': probabilities[i][0],
                    'prediction': predictions[i][0],
                    'ground_truth': labels[i].item()
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add concentration group column
    def get_concentration_group(conc):
        for group, (min_conc, max_conc) in concentration_groups.items():
            if min_conc <= conc < max_conc:
                return group
        return 'other'
    
    results_df['concentration_group'] = results_df['concentration'].apply(get_concentration_group)
    
    # Calculate detection statistics by concentration group
    group_stats = results_df.groupby('concentration_group').agg({
        'prediction': 'mean',  # Detection rate
        'probability': ['mean', 'std', 'count'],
        'concentration': ['mean', 'min', 'max'],
        'ground_truth': 'mean'  # Actual rate of positives
    }).reset_index()
    
    # Flatten column names
    group_stats.columns = ['_'.join(col).strip('_') for col in group_stats.columns.values]
    
    # Rename columns
    group_stats = group_stats.rename(columns={
        'prediction_mean': 'detection_rate',
        'probability_mean': 'mean_probability',
        'probability_std': 'std_probability',
        'probability_count': 'sample_count',
        'concentration_mean': 'mean_concentration',
        'concentration_min': 'min_concentration',
        'concentration_max': 'max_concentration',
        'ground_truth_mean': 'true_positive_rate'
    })
    
    # Sort by mean concentration (descending)
    group_stats = group_stats.sort_values('mean_concentration', ascending=False)
    
    print("Detection rates by concentration group:")
    print(group_stats[['concentration_group', 'detection_rate', 'true_positive_rate', 
                       'mean_probability', 'sample_count', 'mean_concentration']])
    
    # Create Plotly subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Detection Rate by Concentration Group', 
            'Probability Distribution by Concentration Group',
            'Predicted Probability vs Concentration (Log Scale)', 
            'ROC Curves by Concentration Group'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Detection rate by concentration group
    fig.add_trace(
        go.Bar(
            x=group_stats['concentration_group'],
            y=group_stats['detection_rate'],
            name='Detection Rate',
            marker_color='skyblue'
        ),
        row=1, col=1
    )
    
    # 2. Distribution of probabilities by concentration group
    for group in group_stats['concentration_group']:
        group_data = results_df[results_df['concentration_group'] == group]
        
        fig.add_trace(
            go.Box(
                x=group_data['concentration_group'],
                y=group_data['probability'],
                name=group
            ),
            row=1, col=2
        )
    
    # Add threshold line to box plot
    fig.add_trace(
        go.Scatter(
            x=group_stats['concentration_group'],
            y=[threshold] * len(group_stats),
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Threshold ({threshold:.2f})'
        ),
        row=1, col=2
    )
    
    # 3. Scatter plot of probabilities vs concentration (log scale)
    fig.add_trace(
        go.Scatter(
            x=results_df['concentration'],
            y=results_df['probability'],
            mode='markers',
            marker=dict(
                color='blue',
                opacity=0.5,
                size=8
            ),
            name='Predictions'
        ),
        row=2, col=1
    )
    
    # Add threshold line to scatter plot
    fig.add_trace(
        go.Scatter(
            x=[results_df['concentration'].min(), results_df['concentration'].max()],
            y=[threshold, threshold],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Threshold ({threshold:.2f})'
        ),
        row=2, col=1
    )
    
    # 4. ROC curve for different concentration groups
    for group in concentration_groups.keys():
        group_data = results_df[results_df['concentration_group'] == group]
        
        # Only calculate ROC if there are enough samples with both classes
        if len(group_data) > 10 and len(group_data['ground_truth'].unique()) > 1:
            fpr, tpr, _ = roc_curve(group_data['ground_truth'], group_data['probability'])
            
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=group
                ),
                row=2, col=2
            )
    
    # Add diagonal line to ROC plot
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Random'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text='Cell Type Detection Performance by Concentration',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update x-axis for log scale on scatter plot
    fig.update_xaxes(type="log", row=2, col=1, title_text='Concentration (log scale)')
    fig.update_xaxes(title_text='Concentration Group', row=1, col=1)
    fig.update_xaxes(title_text='Concentration Group', row=1, col=2)
    fig.update_xaxes(title_text='False Positive Rate', row=2, col=2)
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Detection Rate', row=1, col=1)
    fig.update_yaxes(title_text='Predicted Probability', row=1, col=2)
    fig.update_yaxes(title_text='Predicted Probability', row=2, col=1)
    fig.update_yaxes(title_text='True Positive Rate', row=2, col=2)
    
    # Show the plot
    fig.write_html(output_path/f"{target_cell_type}_model_analysis.html")
    
    # Determine optimal thresholds for each concentration group
    opt_thresholds = {}
    
    for group in concentration_groups.keys():
        group_data = results_df[results_df['concentration_group'] == group]
        
        # Only calculate if there are enough samples with both classes
        if len(group_data) > 10 and len(group_data['ground_truth'].unique()) > 1:
            precision, recall, thresholds = precision_recall_curve(
                group_data['ground_truth'], group_data['probability'])
            
            # Calculate F1 score for each threshold
            f1_scores = []
            for p, r in zip(precision[:-1], recall[:-1]):  # precision_recall_curve returns one more precision than threshold
                if p + r > 0:
                    f1 = 2 * p * r / (p + r)
                else:
                    f1 = 0
                f1_scores.append(f1)
            
            # Find threshold with best F1 score
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            opt_thresholds[group] = {
                'threshold': best_threshold,
                'f1_score': best_f1,
                'precision': precision[best_idx],
                'recall': recall[best_idx]
            }
    
    # Print optimal thresholds
    print("\nOptimal thresholds by concentration group:")
    for group, stats in opt_thresholds.items():
        print(f"{group}: threshold={stats['threshold']:.4f}, F1={stats['f1_score']:.4f}, "
              f"precision={stats['precision']:.4f}, recall={stats['recall']:.4f}")
    
    # Create a mapping of concentration ranges to optimal thresholds
    threshold_mapping = {}
    for group, (min_conc, max_conc) in concentration_groups.items():
        if group in opt_thresholds:
            threshold_mapping[(min_conc, max_conc)] = opt_thresholds[group]['threshold']
    
    group_stats.to_csv(output_path/f"{target_cell_type}_group_stats.csv")
    print("threshold_mapping:",threshold_mapping)
    return results_df


def find_minimum_detection_concentration(results_df, output_path, target_cell_type, detection_rate_threshold=0.95):
    """
    Find the minimum concentration that can be reliably detected, with Plotly visualization.
    
    Args:
        results_df: DataFrame with detection results by sample
        detection_rate_threshold: Minimum detection rate to consider reliable
        
    Returns:
        min_reliable_conc: Minimum concentration with reliable detection
        bin_stats: DataFrame with detection statistics by concentration bin
    """
    # Create concentration bins (log scale)
    min_conc = results_df['concentration'].min()
    max_conc = results_df['concentration'].max()
    
    # Use log-spaced bins
    log_min = np.log10(max(min_conc, 1e-6))  # Avoid log of zero
    log_max = np.log10(max_conc)
    log_bins = np.linspace(log_min, log_max, 20)
    bins = 10 ** log_bins
    
    # Add bin labels
    results_df['conc_bin'] = pd.cut(results_df['concentration'], bins=bins)
    
    # Calculate detection rate by bin
    bin_stats = results_df.groupby('conc_bin').agg({
        'prediction': 'mean',
        'concentration': ['count', 'min', 'max'],
    }).reset_index()
    
    # Flatten column names
    bin_stats.columns = ['_'.join(col).strip('_') for col in bin_stats.columns.values]
    
    # Find minimum concentration with detection rate above threshold
    reliable_bins = bin_stats[bin_stats['prediction_mean'] >= detection_rate_threshold]
    
    if len(reliable_bins) > 0:
        min_reliable_conc = reliable_bins['concentration_min'].min()
        print(f"Minimum concentration with {detection_rate_threshold*100:.1f}% detection rate: {min_reliable_conc:.8f}")
    else:
        min_reliable_conc = None
        print(f"No concentration bin achieved {detection_rate_threshold*100:.1f}% detection rate")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add detection rate line
    fig.add_trace(
        go.Scatter(
            x=bin_stats['concentration_min'],
            y=bin_stats['prediction_mean'],
            mode='lines+markers',
            name='Detection Rate',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        )
    )
    
    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[bin_stats['concentration_min'].min(), bin_stats['concentration_min'].max()],
            y=[detection_rate_threshold, detection_rate_threshold],
            mode='lines',
            name=f'Target Rate ({detection_rate_threshold:.2f})',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    # Add marker for minimum reliable concentration
    if min_reliable_conc is not None:
        fig.add_trace(
            go.Scatter(
                x=[min_reliable_conc],
                y=[detection_rate_threshold],
                mode='markers',
                name=f'Min Reliable Conc: {min_reliable_conc:.8f}',
                marker=dict(
                    color='green',
                    size=15,
                    symbol='star'
                )
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Detection Rate by Concentration',
        xaxis=dict(
            title='Minimum Concentration (log scale)',
            type='log'
        ),
        yaxis=dict(
            title='Detection Rate',
            range=[0, 1.05]
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=900,
        height=600
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Show the plot
    fig.write_html(output_path/f"{target_cell_type}_minimum_detection_concentration.html")
    
    return min_reliable_conc, bin_stats


def visualize_attention_weights(model, data_sample, output_path, target_ids=None, marker_names=None, device=None):
    """
    Visualize the attention weights assigned to different markers by the model.
    
    Args:
        model: Binary classifier model with attention mechanism
        data_sample: Dictionary containing 'X' and 'coverage' tensors
        target_ids: Mapping of markers to cell types
        marker_names: Optional list of marker names for better visualization
        device: Device to run on
        
    Returns:
        fig: Plotly figure with attention visualization
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    with torch.no_grad():
        # Get data
        marker_values = data_sample['X'].to(device)
        coverage = data_sample['coverage'].to(device)
        
        # Add batch dimension if needed
        if marker_values.dim() == 1:
            marker_values = marker_values.unsqueeze(0)
            coverage = coverage.unsqueeze(0)
        
        # Forward pass to get attention weights
        _, attention_weights = model(marker_values, coverage)
        
        # Get attention for the first sample
        attention = attention_weights[0].cpu().numpy()
        
        # Get marker values and coverage
        marker_vals = marker_values[0].cpu().numpy()
        marker_cov = coverage[0].cpu().numpy()
        
        # Create mask for valid markers (coverage > 0)
        valid_mask = marker_cov > 0
    
    # Prepare data for plotting
    plot_data = []
    
    # Get indices of target cell type markers if target_ids is provided
    target_markers = None
    if target_ids is not None:
        target_ids_np = target_ids if isinstance(target_ids, np.ndarray) else target_ids.cpu().numpy()
        target_markers = target_ids_np == data_sample.get('target_cell_type', 0)
    
    # Create x-axis labels
    x_labels = marker_names if marker_names is not None else [f"M{i}" for i in range(len(attention))]
    
    # Sort markers by attention weight
    sort_indices = np.argsort(-attention)  # Descending order
    
    # Get top 50 markers with highest attention
    top_k = min(50, len(sort_indices))
    top_indices = sort_indices[:top_k]
    
    # Create marker info
    for i in top_indices:
        marker_info = {
            'marker_index': i,
            'marker_name': x_labels[i],
            'attention': attention[i],
            'value': marker_vals[i] if valid_mask[i] else np.nan,
            'coverage': marker_cov[i],
            'is_valid': valid_mask[i],
            'is_target': target_markers[i] if target_markers is not None else True
        }
        plot_data.append(marker_info)
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create Plotly figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Top Markers by Attention Weight', 'Marker Values and Coverage'),
        vertical_spacing=0.2,
        shared_xaxes=True,
        row_heights=[0.6, 0.4]
    )
    
    # Add attention bars
    colors = ['rgba(65, 105, 225, 0.8)' if row['is_target'] else 'rgba(220, 20, 60, 0.8)' for _, row in plot_df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=plot_df['marker_name'],
            y=plot_df['attention'],
            marker_color=colors,
            name='Attention Weight',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add marker values
    fig.add_trace(
        go.Scatter(
            x=plot_df['marker_name'],
            y=plot_df['value'],
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                symbol='circle'
            ),
            name='Marker Value'
        ),
        row=2, col=1
    )
    
    # Add coverage values on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=plot_df['marker_name'],
            y=plot_df['coverage'],
            mode='markers',
            marker=dict(
                size=8,
                color='green',
                symbol='diamond'
            ),
            name='Coverage',
            yaxis='y3'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title='Marker Importance Analysis',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update axes
    fig.update_xaxes(
        title_text='Marker',
        tickangle=45,
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text='Attention Weight',
        range=[0, max(plot_df['attention']) * 1.1],
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='Marker Value',
        row=2, col=1
    )
    
    # Add secondary y-axis for coverage
    fig.update_layout(
        yaxis3=dict(
            title='Coverage',
            anchor='x',
            overlaying='y2',
            side='right'
        )
    )
    
    # Show the plot
    fig.write_html(output_path+"attention_weights.html")
    
    return fig, plot_df


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
