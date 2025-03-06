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
   

def check_prediction_distributions(model, dataloader, device=None):
    """Analyze the raw prediction probabilities for positive and negative samples."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    pos_probs = []
    neg_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, _ = model(marker_values, coverage)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # Store probabilities by true label
            for i, (prob, label) in enumerate(zip(probs, labels)):
                if label > 0.5:  # True positive
                    pos_probs.append(prob)
                else:  # True negative
                    neg_probs.append(prob)
        
    # Print statistics
    print(f"Positive samples: {len(pos_probs)}")
    print(f"  - Mean probability: {np.mean(pos_probs):.4f}")
    print(f"  - Median probability: {np.median(pos_probs):.4f}")
    print(f"  - Min: {np.min(pos_probs):.4f}, Max: {np.max(pos_probs):.4f}")
    
    print(f"Negative samples: {len(neg_probs)}")
    print(f"  - Mean probability: {np.mean(neg_probs):.4f}")
    print(f"  - Median probability: {np.median(neg_probs):.4f}")
    print(f"  - Min: {np.min(neg_probs):.4f}, Max: {np.max(neg_probs):.4f}")
    
    # Check threshold effect
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        tp = sum(p >= threshold for p in pos_probs)
        fn = sum(p < threshold for p in pos_probs)
        fp = sum(p >= threshold for p in neg_probs)
        tn = sum(p < threshold for p in neg_probs)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Threshold {threshold:.1f}: Recall={recall:.4f}, Precision={precision:.4f}, "
              f"Specificity={specificity:.4f}, F1={f1:.4f}")
    
    return pos_probs, neg_probs


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
    for target_cell_type_name in ['CD4-T-cells', 'CD8-T-cells', 'OAC']:
        print("training presence model for",target_cell_type_name)
        # The 'names' set ensures we only keep relevant markers
        names = set(atlas[atlas.target==target_cell_type_name].name.unique())
        print("using", len(names), "markers for detection of presence of",target_cell_type_name)
        target_cell_type=cell_types.index(target_cell_type_name)
        # 2) Build the training DataLoader from parquet files in train_pat_dir
        train_dl = load_training(train_pat_dir, atlas, names, target_cell_type=target_cell_type)
        # 3) Build DataLoaders for each validation subset
        tier1_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "tier1"), atlas, target_cell_type, names)
        tier2_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "OAC"), atlas, target_cell_type, names)
        tier3_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "CD4"), atlas, target_cell_type, names)
        tier4_dl, _ = get_validation_set(str(Path(eval_pat_dir) / "CD8"), atlas, target_cell_type, names)

        validation_dls = {
            "tier1": tier1_dl,        
            "tier2": tier2_dl,        
            "tier3": tier3_dl,        
            "tier4": tier4_dl,        
        }
        
        single_model = SingleCellTypePresenceModel(
            num_markers=len(atlas),       
        )

        # Train it
        trained_model = train_binary_classifier(
            model=single_model,
            dataloaders={"train":train_dl, "val": validation_dls},
            model_path=output_path,
            num_epochs=100,
            learning_rate=1e-3,
            target_cell_type=target_cell_type_name,
        )

        val_dl = tier1_dl
        if target_cell_type_name=="OAC":
            check_prediction_distributions(trained_model, tier2_dl)    
            val_dl = tier2_dl
        if target_cell_type_name=="CD4-T-cells":
            check_prediction_distributions(trained_model, tier3_dl)
            val_dl = tier3_dl
        if target_cell_type_name=="CD8-T-cells":
            check_prediction_distributions(trained_model, tier4_dl)
            val_dl = tier4_dl

        results_df = analyze_detection_by_concentration(trained_model, val_dl, output_path, target_cell_type_name)
        find_minimum_detection_concentration(results_df, output_path, target_cell_type_name)


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
            'super_high': (0.25, 1.0),   # 24% to 100%
            'very_high': (0.10, 0.25),   # 10% to 25%
            'high': (0.05, 0.10),        # 5% to 20%
            'medium': (0.01, 0.05),      # 1% to 5%
            'low': (0.001, 0.01),        # 0.1% to 1%
            'very_low': (0.0005, 0.001), # 0.05% to 0.1%
            'ultra_low': (0.0001, 0.0005)# 0.001% to 0.05%
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


def find_minimum_detection_concentration(results_df, output_path,target_cell_type, detection_rate_threshold=0.95):
    """
    Find the minimum concentration that can be reliably detected with Plotly visualization.
    
    Args:
        results_df: DataFrame with detection results by sample
        detection_rate_threshold: Minimum detection rate to consider reliable
        save_path: Path to save the HTML visualization
        target_cell_type: Name of the target cell type (for file naming)
        
    Returns:
        min_reliable_conc: Minimum concentration with reliable detection
        bin_stats: DataFrame with detection statistics by concentration bin
    """
    # Create concentration bins (log scale)
    min_conc = results_df['true_concentration'].min()
    max_conc = results_df['true_concentration'].max()
    
    # Use log-spaced bins
    log_min = np.log10(max(min_conc, 1e-6))  # Avoid log of zero
    log_max = np.log10(max_conc)
    num_bins = 20
    
    if log_min < log_max:
        bin_edges = np.logspace(log_min, log_max, num_bins+1)
        # Create bin labels
        bin_labels = [f"{bin_edges[i]:.6f}-{bin_edges[i+1]:.6f}" for i in range(len(bin_edges)-1)]
        
        # Add bin column to results_df
        results_df['conc_bin'] = pd.cut(
            results_df['true_concentration'], 
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True
        )
    else:
        # If all concentrations are the same, use a single bin
        results_df['conc_bin'] = f"{min_conc:.6f}-{min_conc:.6f}"
        bin_labels = [f"{min_conc:.6f}-{min_conc:.6f}"]
        bin_edges = [min_conc, min_conc]
    
    # Print debug info to verify data
    print(f"Total samples: {len(results_df)}")
    print(f"Positive samples: {results_df['ground_truth'].sum()}")
    print(f"Predicted positive: {results_df['prediction'].sum()}")
    
    # For each bin, calculate:
    # 1. True positive rate (sensitivity/recall) - correctly identified positives
    # 2. Sample count
    # 3. Min/max concentration
    
    bin_stats = pd.DataFrame()
    bin_stats['conc_bin'] = pd.Series(bin_labels)
    bin_stats['min_concentration'] = [float(bin_name.split('-')[0]) for bin_name in bin_labels]
    bin_stats['max_concentration'] = [float(bin_name.split('-')[1]) for bin_name in bin_labels]
    
    # Calculate actual detection metrics for each bin
    bin_results = []
    
    for bin_label in bin_labels:
        bin_data = results_df[results_df['conc_bin'] == bin_label]
        
        if len(bin_data) == 0:
            # Skip empty bins
            bin_result = {
                'conc_bin': bin_label,
                'detection_rate': 0,
                'sample_count': 0,
                'positives': 0,
                'true_positives': 0,
                'false_negatives': 0
            }
        else:
            # Focus on samples where ground truth is positive (we care about detecting presence)
            positive_samples = bin_data[bin_data['ground_truth'] == 1]
            
            if len(positive_samples) == 0:
                detection_rate = 0  # No positive samples to detect
            else:
                # Detection rate = true positives / total positives (recall/sensitivity)
                true_positives = ((positive_samples['prediction'] == 1) & 
                                 (positive_samples['ground_truth'] == 1)).sum()
                detection_rate = true_positives / len(positive_samples)
            
            bin_result = {
                'conc_bin': bin_label,
                'detection_rate': detection_rate,
                'sample_count': len(bin_data),
                'positives': len(positive_samples),
                'true_positives': ((bin_data['prediction'] == 1) & 
                                  (bin_data['ground_truth'] == 1)).sum(),
                'false_negatives': ((bin_data['prediction'] == 0) & 
                                   (bin_data['ground_truth'] == 1)).sum()
            }
        
        bin_results.append(bin_result)
    
    # Convert to DataFrame
    detection_stats = pd.DataFrame(bin_results)
    
    # Merge detection stats with bin_stats
    bin_stats = pd.merge(bin_stats, detection_stats, on='conc_bin', how='left')
    bin_stats = bin_stats.fillna(0)  # Fill NaN values with 0
    
    # Sort by min_concentration (ascending)
    bin_stats = bin_stats.sort_values('min_concentration')
    
    # Print bin statistics for debugging
    print("\nBin statistics:")
    print(bin_stats[['conc_bin', 'detection_rate', 'sample_count', 'positives', 
                    'true_positives', 'false_negatives']].to_string(index=False))
    
    # Find minimum concentration with detection rate above threshold
    reliable_bins = bin_stats[bin_stats['detection_rate'] >= detection_rate_threshold]
    
    if len(reliable_bins) > 0:
        min_reliable_conc = reliable_bins['min_concentration'].min()
        print(f"Minimum concentration with {detection_rate_threshold*100:.1f}% detection rate: {min_reliable_conc:.8f}")
    else:
        min_reliable_conc = None
        print(f"No concentration bin achieved {detection_rate_threshold*100:.1f}% detection rate")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add detection rate line
    fig.add_trace(
        go.Scatter(
            x=bin_stats['min_concentration'],
            y=bin_stats['detection_rate'],
            mode='lines+markers',
            name='Detection Rate',
            line=dict(color='blue', width=3),
            marker=dict(size=10),
            hovertemplate='Concentration: %{x:.8f}<br>Detection Rate: %{y:.4f}<br>Samples: %{text}',
            text=bin_stats['positives']
        )
    )
    
    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[bin_stats['min_concentration'].min(), bin_stats['min_concentration'].max()],
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
    
    # Add annotation with minimum detection concentration
    if min_reliable_conc is not None:
        fig.add_annotation(
            x=min_reliable_conc,
            y=detection_rate_threshold + 0.05,
            text=f"Min Reliable Concentration: {min_reliable_conc:.8f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    # Add sample count information
    fig.add_trace(
        go.Bar(
            x=bin_stats['min_concentration'],
            y=bin_stats['positives'],
            name='Positive Samples',
            marker_color='lightblue',
            opacity=0.5,
            yaxis='y2'
        )
    )
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title=f'Detection Rate by Concentration for {target_cell_type}',
        xaxis=dict(
            title='Minimum Concentration (log scale)',
            type='log'
        ),
        yaxis=dict(
            title='Detection Rate',
            range=[0, 1.05]
        ),
        yaxis2=dict(
            title='Sample Count',
            overlaying='y',
            side='right',
            rangemode='nonnegative'
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
    
    # Save the visualization if path provided
    fig.write_html(f"{str(output_path)}/{target_cell_type}_minimum_detection_concentration.html")    
    return min_reliable_conc, bin_stats


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
