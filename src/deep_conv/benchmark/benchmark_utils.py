import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import logging
import os
import datetime
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score)



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> Dict[str, np.ndarray]:
    """
    Load the synthetic dataset from a .npz file.

    Args:
        dataset_path (str): Path to the .npz dataset file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing dataset components.
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    data = np.load(dataset_path, allow_pickle=True)
    logger.info(f"Loaded dataset from {dataset_path}")
    return {
        'X_train': data['X_train'],         # Shape: (n_train_samples, n_markers)
        'coverage_train': data['coverage_train'],  # Shape: (n_train_samples, n_markers)
        'y_train': data['y_train'],         # Shape: (n_train_samples, n_cell_types)
        'X_val': data['X_val'],             # Shape: (n_val_samples, n_markers)
        'coverage_val': data['coverage_val'],      # Shape: (n_val_samples, n_markers)
        'y_val': data['y_val'],             # Shape: (n_val_samples, n_cell_types)
        'cell_types': data['cell_types'].tolist(),   # List of cell type names
        'reference_profiles': data['reference_profiles']  # Shape: (n_cell_types, n_markers)
    }


def evaluate_performance(
    true_proportions: np.ndarray,
    estimated_proportions: np.ndarray,
    cell_types: list,
    analyse_proportion_ranges=None,
    min_range_samples: int = 5,
    zero_threshold: float = 0.001
):
    """
    Evaluate performance metrics between true and estimated proportions.

    Args:
        true_proportions (np.ndarray): True cell type proportions (n_samples x n_cell_types).
        estimated_proportions (np.ndarray): Estimated cell type proportions (n_samples x n_cell_types).
        cell_types (List[str]): List of cell type names.
        analyse_proportion_ranges (callable, optional): Function that returns a dict of range metrics.
        min_range_samples (int, optional): Minimum number of samples required to report bin stats.
        zero_threshold (float, optional): Threshold for zero vs. non-zero classification.

    Returns:
        Dict[str, Any]: Nested dictionary containing metrics per cell type, overall metrics, and range breakdown.
    """
    # ----------------------------------------------------------------------
    # Per-cell-type metrics
    # ----------------------------------------------------------------------
    metrics_per_cell = {}

    for i, cell_type in enumerate(cell_types):
        true = true_proportions[:, i]
        est = estimated_proportions[:, i]
        
        rmse = math.sqrt(mean_squared_error(true, est))
        mae = mean_absolute_error(true, est)
        
        # Check zero variance in true or est
        if np.std(true) < 1e-12:
            # If they're essentially identical:
            if np.allclose(true, est, atol=1e-12):
                corr = 1.0
                r2 = 1.0
            else:
                corr = 0.0
                r2  = 0.0
        elif np.std(est) < 1e-12:
            # est is constant but true is not
            corr = 0.0
            r2   = 0.0
        else:
            corr, _ = pearsonr(true, est)
            r2      = r2_score(true, est)
        
        # Zero vs. Non-Zero classification metrics
        # (Only useful if your domain cares about absent vs. present.)
        true_binary = (true >= zero_threshold).astype(int)
        est_binary  = (est >= zero_threshold).astype(int)
        
        zero_acc  = accuracy_score(true_binary, est_binary)
        zero_prec = precision_score(true_binary, est_binary, zero_division=0)
        zero_rec  = recall_score(true_binary, est_binary, zero_division=0)
        zero_f1   = f1_score(true_binary, est_binary, zero_division=0)
        
        metrics_per_cell[cell_type] = {
            "N_Samples": len(true),
            "RMSE": rmse,
            "MAE": mae,
            "Pearson Correlation": corr,
            "R²": r2,
            "Zero/NonZero Accuracy": zero_acc,
            "Zero/NonZero Precision": zero_prec,
            "Zero/NonZero Recall": zero_rec,
            "Zero/NonZero F1": zero_f1
        }
    
    # ----------------------------------------------------------------------
    # Overall metrics across ALL cell types (flattened errors)
    # ----------------------------------------------------------------------
    overall_rmse = math.sqrt(mean_squared_error(true_proportions, estimated_proportions))
    overall_mae  = mean_absolute_error(true_proportions, estimated_proportions)

    # Calculate average of per-cell-type R²
    r2_scores = [m["R²"] for m in metrics_per_cell.values()]
    average_r2 = np.mean(r2_scores)
    
    # "Flattened" global R² (treating all data points as one array)
    true_flat = true_proportions.ravel()
    est_flat  = estimated_proportions.ravel()
    if np.std(true_flat) < 1e-12:
        # all true are constant
        if np.allclose(true_flat, est_flat, atol=1e-12):
            global_r2 = 1.0
        else:
            global_r2 = 0.0
    else:
        global_r2 = r2_score(true_flat, est_flat)
    
    # Per-cell Pearson correlation can be averaged, but let's do the same approach
    # for correlation we do for R²:
    correlations = []
    for i, cell_type in enumerate(cell_types):
        c = metrics_per_cell[cell_type]["Pearson Correlation"]
        correlations.append(c)
    average_corr = np.mean(correlations)
    
    metrics_overall = {
        "Overall RMSE": overall_rmse,
        "Overall MAE": overall_mae,
        "Average R² (Per-Cell)": average_r2,
        "Global R² (Flattened)": global_r2,
        "Average Pearson Correlation": average_corr
    }
    
    # ----------------------------------------------------------------------
    # Range-based metrics
    # ----------------------------------------------------------------------
    range_metrics = {}
    if analyse_proportion_ranges is not None:
        # e.g. range_metrics = analyse_proportion_ranges(pred=estimated_proportions, 
        #                                               true=true_proportions, 
        #                                               cell_types=cell_types)
        range_metrics = analyse_proportion_ranges(pred=estimated_proportions, 
                                                  true=true_proportions, 
                                                  cell_types=cell_types)
    
    # Print a table of range stats, skipping bins with too few samples
    header = (f"{'Cell Type':<20} {'Range':<25} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'N':<8}")
    print(header)
    print("-" * len(header))
    
    for cell_type in cell_types:
        if cell_type in range_metrics:
            cell_range_metrics = range_metrics[cell_type]
            for range_name, stats_dict in cell_range_metrics.items():
                n_samples = stats_dict.get('n_samples', 0)
                # Skip if below threshold
                if n_samples < min_range_samples:
                    continue
                
                mae_ = stats_dict.get('mae', float('nan'))
                rmse_ = stats_dict.get('rmse', float('nan'))
                r2_ = stats_dict.get('r2', float('nan'))
                
                r2_str = '-' if np.isnan(r2_) else f"{r2_:7.4f}"
                print(f"{cell_type[:20]:<20} "
                      f"{range_name:<25} "
                      f"{mae_:7.4f} "
                      f"{rmse_:7.4f} "
                      f"{r2_str:>7} "
                      f"{n_samples:7d}")
    
    # ----------------------------------------------------------------------
    # Return a comprehensive dictionary
    # ----------------------------------------------------------------------
    results = {
        "Per_Cell_Type": metrics_per_cell,
        "Overall": metrics_overall,
        "Range_Breakdown": range_metrics
    }
    return results


def analyse_proportion_ranges(pred: np.ndarray, 
                              true: np.ndarray, 
                              cell_types: Optional[List[str]] = None,
                              custom_ranges: Optional[List[Tuple[float, float, str]]] = None
                             ) -> Dict[str, Dict[str, float]]:
    """
    Analyse prediction performance across different proportion ranges for each cell type,
    including optional uncertainty analysis.
    
    Args:
        pred (np.ndarray): Predicted proportions (n_samples x n_cell_types).
        true (np.ndarray): True proportions (n_samples x n_cell_types).
        cell_types (Optional[List[str]]): List of cell type names. If None, cell types are named generically.
        custom_ranges (Optional[List[Tuple[float, float, str]]]): 
            Custom proportion ranges as a list of tuples (min_val, max_val, range_name). 
            If None, predefined ranges are used.
    
    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with cell types as keys, each containing
            a dictionary of metrics per proportion range.
            Example:
            {
                'CellType_A': {
                    'Ultra-low (0, 1e-4)': {metrics},
                    'Very-low (1e-4, 1e-3)': {metrics},
                    ...
                },
                'CellType_B': {
                    ...
                },
                ...
            }
    """
    if custom_ranges is not None:
        ranges = custom_ranges
    else:
        ranges = [
            (0, 1e-4, "Ultra-low (0, 1e-4)"),
            (1e-4, 1e-3, "Very-low (1e-4, 1e-3)"),
            (1e-3, 1e-2, "Low (1e-3, 1e-2)"),
            (1e-2, 1e-1, "Medium (1e-2, 1e-1)"),
            (1e-1, 1.0, "High (1e-1, 1.0)")
        ]
    
    n_samples, n_cell_types = true.shape
    
    if cell_types is None:
        cell_types = [f'CellType_{i}' for i in range(n_cell_types)]
    
    range_metrics = {cell: {} for cell in cell_types}
    
    for i, cell in enumerate(cell_types):
        for min_val, max_val, range_name in ranges:
            # Create mask for the current cell type and proportion range
            mask = (true[:, i] >= min_val) & (true[:, i] < max_val)
            
            if np.any(mask):
                # Extract relevant predictions and true values
                true_masked = true[mask, i]
                pred_masked = pred[mask, i]
                
                # Calculate R² only if there are enough unique true values
                unique_true = np.unique(true_masked)
                if len(unique_true) > 1:
                    r2 = r2_score(true_masked, pred_masked)
                else:
                    r2 = np.nan  # R² is undefined with a single unique value
                
                # Basic Metrics
                mae = np.mean(np.abs(pred_masked - true_masked))
                rmse = np.sqrt(np.mean((pred_masked - true_masked)**2))
                mean_true = np.mean(true_masked)
                median_pred = np.median(pred_masked)
                mean_pred = np.mean(pred_masked)
                n_samples_in_range = np.sum(mask)
                
                metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mean_true': mean_true,
                    'mean_pred': mean_pred,
                    'median_pred': median_pred,
                    'n_samples': int(n_samples_in_range)
                }
                
                # Assign metrics to the appropriate range
                range_metrics[cell][range_name] = metrics
            else:
                # No samples in this range; assign NaN to all metrics
                metrics = {
                    'mae': np.nan,
                    'rmse': np.nan,
                    'r2': np.nan,
                    'mean_true': np.nan,
                    'mean_pred': np.nan,
                    'median_pred': np.nan,
                    'n_samples': 0
                }
                
                range_metrics[cell][range_name] = metrics
    
    return range_metrics



def log_metrics(metrics: Dict[str, Dict[str, float]]):
    """
    Log the performance metrics.

    Args:
        metrics (Dict[str, Dict[str, float]]): Nested dictionary containing performance metrics.
    """
    logger.info("Performance Metrics:")
    logger.info("\n--- Per Cell Type ---")
    for cell_type, metric in metrics['Per_Cell_Type'].items():
        logger.info(f"{cell_type}: RMSE={metric['RMSE']:.4f}, MAE={metric['MAE']:.4f}, Pearson Correlation={metric['Pearson Correlation']:.4f}, R²={metric['R²']:.4f}")
    
    logger.info("\n--- Overall Metrics ---")
    for metric_name, value in metrics['Overall'].items():
        logger.info(f"{metric_name}: {value:.4f}")


def save_estimated_proportions(
    estimated_train: np.ndarray,
    estimated_val: np.ndarray,
    save_path: str
):
    """
    Save the estimated proportions to a .npz file.

    Args:
        estimated_train (np.ndarray): Estimated training proportions.
        estimated_val (np.ndarray): Estimated validation proportions.
        save_path (str): Directory to save the estimated proportions.
    """
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"estimated_proportions_{timestamp}.npz"
    filepath = os.path.join(save_path, filename)
    np.savez_compressed(filepath, X_train=estimated_train, X_val=estimated_val)
    logger.info(f"Estimated proportions saved to {filepath}")



def columns():
    return [
        f"d1e-1_{i}" for i in range(1, 1001)
    ] + [
        f"d5e-2_{i}" for i in range(1, 1001)
    ] + [
        f"d1e-2_{i}" for i in range(1, 1001)
    ] + [
        f"d5e-3_{i}" for i in range(1, 1001)
    ] + [
        f"d1e-3_{i}" for i in range(1, 1001)
    ] + [
        f"d1e-4_{i}" for i in range(1, 1001)
    ] + [
        f"d1e-5_{i}" for i in range(1, 1001)
    ]


def dilutions():
    return [
        1e-1 for i in range(1, 1001)
    ] + [
        5e-2 for i in range(1, 1001)
    ] + [
        1e-2 for i in range(1, 1001)
    ] + [
        5e-3 for i in range(1, 1001)
    ] + [
        1e-3 for i in range(1, 1001)
    ] + [
        1e-4 for i in range(1, 1001)
    ] + [
        1e-5 for i in range(1, 1001)
    ] 


def process_atlas(atlas_path: str, fillna: str) -> Tuple[np.ndarray, List[str]]:
    """
    Process methylation atlas file to get reference profiles.

    Args:
        atlas_path (str): Path to the atlas file.

    Returns:
        Tuple[np.ndarray, List[str]]: Reference methylation profiles and cell type names.
    """
    logger.info(f"Processing atlas file at {atlas_path}")
    df = pd.read_csv(atlas_path, sep='\t')
    cell_type_cols = df.columns[8:]
    methylation_matrix = df[cell_type_cols].replace('NA', np.nan).astype(float)
    if fillna=="drop":
        methylation_matrix = methylation_matrix.dropna().drop_duplicates()
    elif fillna=="mean_cell_type":
        methylation_matrix = methylation_matrix.fillna(methylation_matrix.mean(axis=0)).drop_duplicates()
    elif fillna=="mean":
        methylation_matrix = methylation_matrix.fillna(methylation_matrix.mean())
    reference_profiles = methylation_matrix.to_numpy().T
    logger.info(f"Processed atlas with {reference_profiles.shape[0]} cell types and {reference_profiles.shape[1]} regions.")
    return reference_profiles, cell_type_cols.tolist()


def calculate_dilution_metrics(results_df):
    metrics = []
    overall_r2 = r2_score(
        results_df['dilution'],
        results_df['contribution'] / results_df['contribution'].max() * results_df['dilution'].max()
    )
    for dilution in sorted(results_df['dilution'].unique()):
        dilution_results = results_df[results_df['dilution'] == dilution]
        y_pred = dilution_results['contribution'].values
        mse = mean_squared_error(
            dilution * np.ones(len(y_pred)), 
            y_pred
        )
        rmse = np.sqrt(mse)
        rel_error = np.mean(np.abs(y_pred - dilution) / dilution)
        expected_ratio = dilution / results_df['dilution'].max()
        actual_ratio = np.mean(y_pred) / np.mean(results_df['contribution'].max())
        ratio_error = abs(expected_ratio - actual_ratio) / expected_ratio
        metrics.append({
            'dilution': dilution,
            'mse': mse,
            'rmse': rmse,
            'ratio_error': ratio_error,
            'relative_error': rel_error,
            'mean_pred': np.mean(y_pred),
            'median_pred': np.median(y_pred),
            'std_pred': np.std(y_pred),
            'n_samples': len(dilution_results)
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df['overall_r2'] = overall_r2
    return metrics_df


def plot_dilution_results(results_df, cell_type, all_predictions_df, output_path, show_metrics=True):
    # Calculate metrics
    metrics_df = calculate_dilution_metrics(results_df)
    # Calculate mean and std per dilution
    summary = results_df.groupby('dilution').agg({'contribution': ['mean', 'std']}).reset_index()
    summary.columns = ['dilution', 'mean', 'std']
    # Calculate the mean proportion for each cell type
    cell_type_means = all_predictions_df.mean(axis=0)
    # Sort the cell types by descending mean proportion (for red at top, blue at bottom)
    sorted_cell_types = cell_type_means.sort_values(ascending=False).index.tolist()  # Corrected to descending
    # Reorder the data accordingly
    sorted_heatmap_data = all_predictions_df[sorted_cell_types]
    # Create figure with subplots
    violin_titles = ['CD34-erythroblasts', 'CD34-megakaryocytes', 'Monocytes', 'Neutrophils']
    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Mean Contribution vs Dilution',    # 1
            'Performance Metrics',              # 2
            'Cell Type Proportions Heatmap',    # 3
            'CD34-erythroblasts',              # 7
            'CD34-megakaryocytes',             # 8
            'Monocytes',                       # 9
            'Neutrophils'                      # 10
        ),
        specs=[
            [{"type": "scatter"}, {"type": "table"}],        # row 1
            [{"type": "heatmap", "colspan": 2}, None],       # row 2
            [{"type": "violin"}, {"type": "violin"}],        # row 3
            [{"type": "violin"}, {"type": "violin"}]         # row 4
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        row_heights=[0.25, 0.35, 0.2, 0.2]
    )
    # Plot 1: Error bar plot
    fig.add_trace(
        go.Scatter(
            x=summary['dilution'],
            y=summary['mean'],
            error_y=dict(type='data', array=summary['std'], visible=True),
            mode='markers+lines',
            name=cell_type
        ),
        row=1, col=1
    )
    fig.update_xaxes(type="log", title="Dilution", row=1, col=1)
    fig.update_yaxes(type="log", title="Mean Fraction", row=1, col=1)
    # Plot 2: Metrics table
    if show_metrics:
        metrics_display = metrics_df.round(4)
        fig.add_trace(
            go.Table(
                header=dict(values=list(metrics_display.columns), align='left'),
                cells=dict(values=[metrics_display[col] for col in metrics_display.columns], align='left')
            ),
            row=1, col=2
        )
    # Plot 3: Heatmap with sorted cell types
    fig.add_trace(
        go.Heatmap(
            z=sorted_heatmap_data.T.values,  # Transposed sorted data
            x=sorted_heatmap_data.index,     # Sample indices
            y=sorted_cell_types,             # Sorted cell types (high to low)
            colorscale='RdBu_r',             # Red for high, blue for low
            colorbar=dict(title='Proportion'),
            zmin=0,
            zmax=0.4
        ),
        row=2, col=1
    )
    fig.update_xaxes(showticklabels=False, title="Samples", row=2, col=1)
    fig.update_yaxes(title="Cell Types", row=2, col=1)
    # Plot 4: Violin plots
    violin_data = all_predictions_df.copy()
    violin_data['dilution'] = results_df['dilution'].values  # Add dilution column
    violin_data = violin_data[violin_titles + ['dilution']]
    # Explicitly map violin titles to specific subplot locations
    violin_titles_mapping = {
        (3, 1): 'CD34-erythroblasts',
        (3, 2): 'CD34-megakaryocytes',
        (4, 1): 'Monocytes',
        (4, 2): 'Neutrophils'
    }
    # Ensure data matches titles
    print("Violin Data Columns:", violin_data.columns.tolist())
    print("Violin Titles Expected:", list(violin_titles_mapping.values()))
    violin_positions = [(3,1), (3,2), (4,1), (4,2)]
    cell_types_config = {
        'CD34-erythroblasts': {
            'color': 'rgba(147, 112, 219, 0.3)',
            'line_color': 'rgb(147, 112, 219)'
        },
        'CD34-megakaryocytes': {
            'color': 'rgba(255, 165, 0, 0.3)',
            'line_color': 'rgb(255, 165, 0)'
        },
        'Monocytes': {
            'color': 'rgba(0, 191, 255, 0.3)',
            'line_color': 'rgb(0, 191, 255)'
        },
        'Neutrophils': {
            'color': 'rgba(255, 192, 203, 0.3)',
            'line_color': 'rgb(255, 192, 203)'
        }
    }
    violin_cell_types = ['CD34-erythroblasts', 'CD34-megakaryocytes', 'Monocytes', 'Neutrophils']
    for (row, col), cell_type_name in zip(violin_positions, violin_cell_types):
        fig.add_trace(
            go.Violin(
                x=violin_data['dilution'].astype(str),  # Convert to string to make categorical
                y=violin_data[cell_type_name],
                name=cell_type_name,
                fillcolor=cell_types_config[cell_type_name]['color'],
                line_color=cell_types_config[cell_type_name]['line_color'],
                box=dict(
                    visible=True,
                    fillcolor='white',
                    line=dict(color='black', width=1)
                ),
                meanline=dict(visible=False),
                points=False,
                side='both',
                hoveron='violins',
                scalegroup='all',  # Keep consistent scaling across all plots
                scalemode='width'  # Scale by width rather than count
            ),
            row=row, col=col
        )
    fig.update_yaxes(range=[0, None])
    # Automatically set subplot titles
    fig.update_layout(
        title=dict(
            text=f"Violin Plot Subplots",
            x=0.5,  # Center the overall title
            xanchor="center"
        )
    )
    # Update x-axes for violin plots
    fig.update_xaxes(title="Dilution", row=3, col=1)
    fig.update_xaxes(title="Dilution", row=3, col=2)
    fig.update_xaxes(title="Dilution", row=4, col=1)
    fig.update_xaxes(title="Dilution", row=4, col=2)
    # Update y-axes for violin plots to avoid clipping
    for row in [3, 4]:
        for col in [1, 2]:
            fig.update_yaxes(title="Proportion", range=[0, 0.5], row=row, col=col)  # Adjusted to 0-0.5 to avoid clipping
    # Update overall layout
    fig.update_layout(
        height=1400,
        width=1600,
        showlegend=False,
        title=dict(
            text=f"Cell Type Analysis: {cell_type}",
            x=0.5,
            xanchor='center'
        )
    )
    # Save outputs
    metrics_df.to_csv(f"{output_path}{cell_type}_model_performance_metrics.csv", index=False)
    fig.write_html(f"{output_path}{cell_type}.html")
    fig.write_image(f"{output_path}{cell_type}.png", scale=2)

    fig = plot_concentrations_by_dilution(results_df)
    fig.write_html(f"{output_path}{cell_type}_expected_vs_actual.html")


def calculate_classification_metrics(y_true, y_pred, thresholds):
    """
    Calculate sensitivity, specificity and accuracy at different ground truth thresholds
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        thresholds: List of threshold values to evaluate at
    
    Returns:
        DataFrame with metrics for each threshold
    """
    metrics = []
    for threshold in thresholds:
        # Create binary classifications
        true_positive = (y_true >= threshold) & (y_pred >= threshold)
        true_negative = (y_true < threshold) & (y_pred < threshold)
        false_positive = (y_true < threshold) & (y_pred >= threshold)
        false_negative = (y_true >= threshold) & (y_pred < threshold)
        
        # Calculate metrics
        sensitivity = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_negative))
        specificity = np.sum(true_negative) / (np.sum(true_negative) + np.sum(false_positive))
        accuracy = (np.sum(true_positive) + np.sum(true_negative)) / len(y_true)
        precision = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive))
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        
        metrics.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'precision': precision,
            'f1': f1,
            'n_positive': np.sum(y_true >= threshold),
            'n_total': len(y_true)
        })
    
    return pd.DataFrame(metrics)


def calculate_roc_points(y_true, y_pred, thresholds):
    """
    Calculate ROC curve points manually for continuous values
    """
    fpr_list = []
    tpr_list = []
    
    for threshold in thresholds:
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    return np.array(fpr_list), np.array(tpr_list)


def calculate_classification_metrics(y_true, y_pred, thresholds):
    """
    Calculate sensitivity, specificity and accuracy at different ground truth thresholds
    """
    metrics = []
    for threshold in thresholds:
        # Create binary classifications
        y_true_binary = (y_true >= threshold)
        y_pred_binary = (y_pred >= threshold)
        tp = np.sum(y_true_binary & y_pred_binary)
        tn = np.sum(~y_true_binary & ~y_pred_binary)
        fp = np.sum(~y_true_binary & y_pred_binary)
        fn = np.sum(y_true_binary & ~y_pred_binary)
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        metrics.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'precision': precision,
            'f1': f1,
            'n_positive': np.sum(y_true >= threshold),
            'n_total': len(y_true)
        })
    return pd.DataFrame(metrics)


def calculate_performance_metrics(y_true, y_pred, rel_tolerances=[0.1, 0.2, 0.5]):
    """
    Calculate performance metrics using relative tolerances
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        rel_tolerances: List of relative tolerance values (e.g. 0.1 = 10% tolerance)
    """
    metrics = []
    
    for tolerance in rel_tolerances:
        # Consider prediction correct if within X% of true value
        relative_error = np.abs(y_pred - y_true) / np.maximum(y_true, 1e-10)
        correct_predictions = relative_error <= tolerance
        
        # Only consider cases where true value is significant
        significant_true = y_true > 0.001  # Minimum significant concentration
        
        # Calculate metrics
        tp = np.sum(correct_predictions & significant_true)
        fp = np.sum(~correct_predictions & significant_true)
        tn = np.sum(correct_predictions & ~significant_true)
        fn = np.sum(~correct_predictions & ~significant_true)
        
        # Calculate performance metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics.append({
            'tolerance': tolerance,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'precision': precision,
            'f1': f1,
            'mean_relative_error': np.mean(relative_error[significant_true]),
            'median_relative_error': np.median(relative_error[significant_true])
        })
    
    return pd.DataFrame(metrics)


def evaluate_cell_type_performance(y_true, predictions, intended_dilutions):
    """
    Evaluate model performance for a single cell type
    """
    metrics = []
    
    for dilution in sorted(intended_dilutions.unique()):
        # Get samples for this intended dilution
        dilution_mask = intended_dilutions == dilution
        
        if not np.any(dilution_mask):
            continue
            
        y_true_dilution = y_true[dilution_mask]
        pred_dilution = predictions[dilution_mask]
        
        # Calculate detection metrics using actual values
        true_positives = np.sum((pred_dilution >= y_true_dilution * 0.8) & (y_true_dilution > 0.001))
        false_positives = np.sum((pred_dilution >= y_true_dilution * 1.2) & (y_true_dilution > 0.001))
        true_negatives = np.sum((pred_dilution < 0.001) & (y_true_dilution <= 0.001))
        false_negatives = np.sum((pred_dilution < y_true_dilution * 0.8) & (y_true_dilution > 0.001))
        
        total_positive_cases = np.sum(y_true_dilution > 0.001)
        total_negative_cases = np.sum(y_true_dilution <= 0.001)
        
        # Calculate metrics
        sensitivity = true_positives / total_positive_cases if total_positive_cases > 0 else 0
        specificity = true_negatives / total_negative_cases if total_negative_cases > 0 else 0
        accuracy = (true_positives + true_negatives) / len(y_true_dilution)
        
        # Calculate relative errors for significant concentrations
        significant_mask = y_true_dilution > 0.001
        rel_errors = np.abs(pred_dilution[significant_mask] - y_true_dilution[significant_mask]) / y_true_dilution[significant_mask]
        
        metrics.append({
            'intended_dilution': dilution,
            'actual_mean': np.mean(y_true_dilution),
            'actual_std': np.std(y_true_dilution),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'median_rel_error': np.median(rel_errors) if len(rel_errors) > 0 else np.nan,
            'mean_rel_error': np.mean(rel_errors) if len(rel_errors) > 0 else np.nan,
            'n_samples': len(y_true_dilution)
        })
    
    return pd.DataFrame(metrics)


def plot_deconvolution_evaluation(y_true_df, predictions_df, intended_dilutions, output_path, samples_per_dilution=1000, alpha_threshold=1e-4):
    """
    Create comprehensive evaluation plots with advanced metrics for methylation deconvolution.
    
    Args:
        y_true_df: DataFrame with ground truth concentrations
        predictions_df: DataFrame with model predictions
        intended_dilutions: Series with intended dilution values for each sample
        output_path: Directory to save output files
        samples_per_dilution: Number of samples to use per dilution in heatmap
        alpha_threshold: Threshold below which predictions are set to 0 (default: 1e-4)
    """
    import os
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score, roc_curve, auc, precision_recall_curve, average_precision_score
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.stats import pearsonr, spearmanr
    
    os.makedirs(output_path, exist_ok=True)
    
    significant_cell_types = []
    for cell_type in y_true_df.columns:
        if np.any(y_true_df[cell_type] > 0.001):
            significant_cell_types.append(cell_type)
    
    # Sample for heatmap
    sampled_indices = []
    for dilution in sorted(intended_dilutions.unique()):
        dilution_indices = intended_dilutions[intended_dilutions == dilution].index
        if len(dilution_indices) > samples_per_dilution:
            sampled_indices.extend(np.random.choice(dilution_indices, samples_per_dilution, replace=False))
        else:
            sampled_indices.extend(dilution_indices)
    
    results = {}
    
    for cell_type in significant_cell_types:
        # Extract values
        y_true_vals = y_true_df[cell_type].values
        y_pred_vals = predictions_df[cell_type].values
        
        # Apply model's thresholding behavior (alpha < threshold set to 0)
        y_pred_vals_thresholded = y_pred_vals.copy()
        y_pred_vals_thresholded[y_pred_vals_thresholded < alpha_threshold] = 0
        
        # Calculate overall metrics
        r2 = r2_score(y_true_vals, y_pred_vals_thresholded)
        pearson_r = np.corrcoef(y_true_vals, y_pred_vals_thresholded)[0, 1]
        spearman_r = spearmanr(y_true_vals, y_pred_vals_thresholded)[0]
        mae = np.mean(np.abs(y_true_vals - y_pred_vals_thresholded))
        rmse = np.sqrt(np.mean(np.square(y_true_vals - y_pred_vals_thresholded)))
        
        # Calculate concentration-stratified MAE
        conc_strata = [
            (0, 0.001, 'Low (<0.1%)'),
            (0.001, 0.01, 'Med (0.1-1%)'),
            (0.01, 0.05, 'Med-High (1-5%)'),
            (0.05, 0.1, 'High (5-10%)'),
            (0.1, 1.0, 'Very High (≥10%)')
        ]
        
        cs_mae = {}
        for low, high, name in conc_strata:
            mask = (y_true_vals >= low) & (y_true_vals < high)
            if np.any(mask):
                cs_mae[name] = {
                    'mae': np.mean(np.abs(y_true_vals[mask] - y_pred_vals_thresholded[mask])),
                    'count': np.sum(mask)
                }
        
        # Calculate magnitude-sensitive metrics for different thresholds
        detection_thresholds = [0.001, 0.005, 0.01, 0.05, 0.10]
        
        def calculate_magnitude_metrics(y_true_vals, y_pred_vals, thresholds):
            magnitude_metrics = {}
            for threshold in thresholds:
                mask = (y_true_vals >= threshold * 0.5) & (y_true_vals <= threshold * 2)
                if np.sum(mask) == 0:
                    magnitude_metrics[f'threshold_{threshold:.3f}'] = {
                        'median_rel_error': np.nan,
                        'within_10pct': np.nan,
                        'n_samples': 0
                    }
                    continue
                y_true_subset = y_true_vals[mask]
                y_pred_subset = y_pred_vals[mask]
                rel_errors = 2 * (y_pred_subset - y_true_subset) / (y_pred_subset + y_true_subset + 1e-6)
                median_rel_error = np.median(rel_errors) * 100
                within_10pct = np.mean(np.abs(rel_errors) <= 0.1) * 100
                magnitude_metrics[f'threshold_{threshold:.3f}'] = {
                    'median_rel_error': median_rel_error,
                    'within_10pct': within_10pct,
                    'n_samples': np.sum(mask)
                }
            return magnitude_metrics
        
        magnitude_metrics = calculate_magnitude_metrics(y_true_vals, y_pred_vals_thresholded, detection_thresholds)
        
        # Create ROC curve for detection tasks
        detection_threshold = 0.005
        y_true_binary = y_true_vals >= detection_threshold
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_vals_thresholded)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_vals_thresholded)
        avg_precision = average_precision_score(y_true_binary, y_pred_vals_thresholded)
        
        # Create extended subplot layout with more space for the summary table
        fig = make_subplots(
            rows=6, cols=2,
            subplot_titles=[
                "Cell Type Proportions Heatmap", "Detection Performance Metrics",
                "Predicted vs Ground Truth", "Concentration-Stratified MAE",
                "Relative Error Rate", "ROC Curve",
                "Error by Concentration Range", "Precision-Recall Curve",
                "Calibration Plot", "Error Distribution",
                "Summary Metrics", None
            ],
            specs=[
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "histogram"}],
                [{"colspan": 2}, None],
            ],
            vertical_spacing=0.03,
            horizontal_spacing=0.08,
            row_heights=[0.2, 0.15, 0.15, 0.15, 0.15, 0.3]
        )
        
        # --------------------- Row 1, Col 1: Heatmap --------------------- #
        heatmap_data = predictions_df.loc[sampled_indices]
        cell_type_means = heatmap_data.mean(axis=0)
        sorted_cell_types = cell_type_means.sort_values(ascending=False).index.tolist()
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data[sorted_cell_types].T.values,
                x=np.arange(len(sampled_indices)),
                y=sorted_cell_types,
                colorscale='RdBu_r',
                colorbar=dict(
                    title='Proportion',
                    len=0.2,
                    y=0.9,
                    yanchor="middle"
                ),
                zmin=0,
                zmax=0.4
            ),
            row=1, col=1
        )
        
        # -------- Row 1, Col 2: Detection Performance Metrics --------- #
        threshold_values = []
        median_rel_error_values = []
        within_10pct_values = []
        
        for threshold in detection_thresholds:
            metrics = magnitude_metrics[f'threshold_{threshold:.3f}']
            threshold_values.append(f"{threshold*100:.1f}%")
            median_rel_error_values.append(metrics['median_rel_error'])
            within_10pct_values.append(metrics['within_10pct'])
        
        fig.add_trace(
            go.Bar(
                x=threshold_values,
                y=median_rel_error_values,
                name="Median Relative Error (%)",
                marker_color="blue",
                legendgroup="magnitude_metrics",
                legendgrouptitle_text="Magnitude Metrics"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=threshold_values,
                y=within_10pct_values,
                name="% Within ±10% Rel Error",
                marker_color="green",
                legendgroup="magnitude_metrics"
            ),
            row=1, col=2
        )
        
        # ----------------- Row 2, Col 1: Predicted vs. True ---------------- #
        fig.add_trace(
            go.Scatter(
                x=y_true_vals * 100,
                y=y_pred_vals_thresholded * 100,
                mode='markers',
                marker=dict(
                    color=np.log10(intended_dilutions),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title='log10(Intended Dilution)', 
                        len=0.2, 
                        y=0.65,
                        yanchor="top"
                    ), 
                    size=3,
                    opacity=0.6
                ),
                name="Predicted vs True",
                legendgroup="main",
                showlegend=False
            ),
            row=2, col=1
        )
        
        non_zero_vals = y_true_vals[y_true_vals > 0]
        min_val = max(non_zero_vals.min() * 100 if len(non_zero_vals) > 0 else 1e-6, 0.001)
        max_val = y_true_vals.max() * 100
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction',
                legendgroup="main"
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(
            title="True Value (%)", 
            type="log",
            row=2, col=1,
            ticktext=[f"{x:.3g}%" for x in [0.001, 0.01, 0.1, 1, 10]],
            tickvals=[0.001, 0.01, 0.1, 1, 10]
        )
        fig.update_yaxes(
            title="Predicted Value (%)", 
            type="log",
            row=2, col=1,
            ticktext=[f"{x:.3g}%" for x in [0.001, 0.01, 0.1, 1, 10]],
            tickvals=[0.001, 0.01, 0.1, 1, 10]
        )
        
        # ----------------- Row 2, Col 2: Concentration-Stratified MAE ---------------- #
        strata_names = []
        mae_values = []
        count_texts = []
        
        for name, data in cs_mae.items():
            strata_names.append(name)
            mae_values.append(data['mae'])
            count_texts.append(f"n={data['count']}")
        
        fig.add_trace(
            go.Bar(
                x=strata_names,
                y=mae_values,
                name='MAE by Concentration',
                text=count_texts,
                textposition='auto',
                legendgroup="main"
            ),
            row=2, col=2
        )
        
        # -------- Row 3, Col 1: Performance Metrics by Dilution --------- #
        metrics_df = calculate_metrics_by_dilution(y_true_vals, y_pred_vals_thresholded, intended_dilutions)
        metrics_df = metrics_df.sort_values('dilution', ascending=True)
        
        fig.add_trace(
            go.Scatter(
                x=metrics_df['dilution'] * 100,
                y=metrics_df['within_10pct'],
                mode='lines+markers',
                name='% Within ±10%',
                line=dict(color='blue'),
                legendgroup="error_metrics",
                legendgrouptitle_text="Error Metrics"
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=metrics_df['dilution'] * 100,
                y=metrics_df['median_rel_error'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=metrics_df['q75_rel_error'] - metrics_df['median_rel_error'],
                    arrayminus=metrics_df['median_rel_error'] - metrics_df['q25_rel_error']
                ),
                mode='lines+markers',
                name='Relative Error',
                line=dict(color='red'),
                legendgroup="error_metrics"
            ),
            row=3, col=1
        )
        
        # Dynamically set tickvals and ticktext based on intended_dilutions
        unique_dilutions = sorted(np.unique(intended_dilutions))
        tickvals = [dil * 100 for dil in unique_dilutions]
        ticktext = [f"{dil*100:.3g}%" for dil in unique_dilutions]
        
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="gray", row=3, col=1)
        fig.add_hline(y=-20, line_dash="dot", line_color="gray", row=3, col=1)
        
        # -------- Row 3, Col 2: ROC Curve --------- #
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC curve (AUC = {roc_auc:.3f})',
                line=dict(color='darkorange'),
                legendgroup="roc_pr",
                legendgrouptitle_text="ROC/PR Curves"
            ),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='navy', dash='dash'),
                showlegend=False
            ),
            row=3, col=2
        )
        
        # --------- Row 4, Col 1: Error by Concentration Range ---------- #
        concentration_ranges = [
            ('Low (<0.1%)', y_true_vals < 0.001),
            ('Med (0.1-1%)', (y_true_vals >= 0.001) & (y_true_vals < 0.01)),
            ('Med-High (1-5%)', (y_true_vals >= 0.01) & (y_true_vals < 0.05)),
            ('High (5-10%)', (y_true_vals >= 0.05) & (y_true_vals < 0.10)),
            ('Very High (≥10%)', y_true_vals >= 0.10)
        ]
        
        for conc_type, mask in concentration_ranges:
            if np.any(mask):
                fig.add_trace(
                    go.Box(
                        y=2 * (y_pred_vals_thresholded[mask] - y_true_vals[mask]) / (y_pred_vals_thresholded[mask] + y_true_vals[mask] + 1e-6) * 100,
                        name=conc_type,
                        legendgroup="concentration",
                        legendgrouptitle_text="Concentration"
                    ),
                    row=4, col=1
                )
        
        # --------- Row 4, Col 2: Precision-Recall Curve ---------- #
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name=f'PR curve (AP = {avg_precision:.3f})',
                line=dict(color='green'),
                legendgroup="roc_pr"
            ),
            row=4, col=2
        )
        
        # --------- Row 5, Col 1: Bias-Variance Heatmap ---------- #
        conc_bins = np.logspace(-4, -1, 20)
        error_matrix = np.zeros((len(conc_bins)-1, 3))
        
        for i in range(len(conc_bins)-1):
            mask = (y_true_vals >= conc_bins[i]) & (y_true_vals < conc_bins[i+1])
            if np.sum(mask) > 0:
                errors = y_pred_vals_thresholded[mask] - y_true_vals[mask]
                error_matrix[i, 0] = np.mean(errors)
                error_matrix[i, 1] = np.mean(np.abs(errors))
                error_matrix[i, 2] = np.var(errors)
        
        for j in range(3):
            if np.std(error_matrix[:, j]) > 0:
                error_matrix[:, j] = (error_matrix[:, j] - np.mean(error_matrix[:, j])) / np.std(error_matrix[:, j])
            
        fig.add_trace(
            go.Heatmap(
                z=error_matrix,
                x=['Bias', 'MAE', 'Variance'],
                y=[f"{conc_bins[i]:.4f}-{conc_bins[i+1]:.4f}" for i in range(len(conc_bins)-1)],
                colorscale='RdBu_r',
                colorbar=dict(
                    title='Z-score',
                    len=0.2,
                    y=0.225,
                    yanchor="middle",
                    title_side="right"
                ),
                zmin=-2,
                zmax=2
            ),
            row=5, col=1
        )
        
        # --------- Row 5, Col 2: Error Distribution Histogram ---------- #
        errors = y_pred_vals_thresholded - y_true_vals
        
        fig.add_trace(
            go.Histogram(
                x=errors,
                nbinsx=50,
                name='Error Distribution',
                marker_color='blue'
            ),
            row=5, col=2
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=5, col=2)
        
        # --------- Row 6: Clinical Relevance Score and Summary Metrics ---------- #
        if "T-cells" in cell_type:
            weights = {
                '0.1-1%': 3.0,
                '1-5%': 1.5,
                '5-10%': 1.0,
                '>10%': 0.5,
                'within_10pct': 2.0
            }
        else:
            weights = {
                '0.1-1%': 2.0,
                '1-5%': 1.5,
                '5-10%': 1.0,
                '>10%': 0.8,
                'within_10pct': 2.0
            }
        
        crs_components = []
        
        for range_name, data in cs_mae.items():
            if range_name in weights:
                norm_score = 1.0 / (1.0 + 10*data['mae'])
                crs_components.append(weights[range_name] * norm_score)
        
        within_10pct = magnitude_metrics['threshold_0.005']['within_10pct'] / 100
        crs_components.append(weights['within_10pct'] * within_10pct)
        
        crs = 100 * sum(crs_components) / sum(weights.values())
        
        summary_text = [
            f"<b>Summary Metrics for {cell_type}</b>",
            f"R²: {r2:.3f} | Pearson r: {pearson_r:.3f} | Spearman r: {spearman_r:.3f}",
            f"MAE: {mae:.5f} | RMSE: {rmse:.5f}",
            f"ROC-AUC: {roc_auc:.3f} | PR-AUC: {avg_precision:.3f}"
        ]
        
        for threshold in detection_thresholds:
            metrics = magnitude_metrics[f'threshold_{threshold:.3f}']
            summary_text.append(
                f"Error at {threshold*100:.1f}%: {metrics['median_rel_error']:.2f}% rel err, {metrics['within_10pct']:.1f}% within ±10%"
            )
        
        dilution_metrics = calculate_metrics_by_dilution(y_true_vals, y_pred_vals_thresholded, intended_dilutions)
        key_dilutions = [0.0001, 0.001, 0.01, 0.1]
        for dilution in key_dilutions:
            dil_metrics = dilution_metrics[dilution_metrics['dilution'] == dilution]
            if not dil_metrics.empty:
                mae_at_dilution = dil_metrics['mae'].iloc[0]
                median_rel_error = dil_metrics['median_rel_error'].iloc[0]
                summary_text.append(
                    f"MAE at {dilution*100:.2f}%: {mae_at_dilution:.5f} | Median Rel Error: {median_rel_error:.2f}%"
                )
        
        summary_text.append(f"<b>Clinical Relevance Score: {crs:.1f}/100</b>")
        
        fig.add_annotation(
            x=0.5,
            y=0.1,
            text="<br>".join(summary_text),
            showarrow=False,
            font=dict(size=10),
            align="center",
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
            bgcolor="white",
            opacity=0.8,
            xref="x6",
            yref="y6"
        )
        
        # --------------------- Axis Updates ---------------------- #
        fig.update_xaxes(showticklabels=False, title="Samples", row=1, col=1)
        fig.update_yaxes(title="Cell Types", row=1, col=1)
        fig.update_xaxes(title="Detection Threshold", row=1, col=2)
        fig.update_yaxes(title="Error (%) / Proportion (%)", row=1, col=2)
        
        fig.update_xaxes(title="Concentration Range", row=2, col=2)
        fig.update_yaxes(title="Mean Absolute Error", row=2, col=2)
        
        fig.update_xaxes(
            title="Intended Dilution (%)", 
            type="log", 
            row=3, col=1,
            tickvals=tickvals,
            ticktext=ticktext
        )
        fig.update_yaxes(title="Percentage / Error", row=3, col=1)
        fig.update_xaxes(title="False Positive Rate", range=[0, 1], row=3, col=2)
        fig.update_yaxes(title="True Positive Rate", range=[0, 1], row=3, col=2)
        
        fig.update_xaxes(title="Concentration Range", row=4, col=1)
        fig.update_yaxes(title="Relative Error (%)", row=4, col=1)
        fig.update_xaxes(title="Recall", range=[0, 1], row=4, col=2)
        fig.update_yaxes(title="Precision", range=[0, 1], row=4, col=2)
        
        fig.update_xaxes(title="Error Component", row=5, col=1)
        fig.update_yaxes(title="Concentration Range", row=5, col=1)
        fig.update_xaxes(title="Prediction Error", row=5, col=2)
        fig.update_yaxes(title="Count", row=5, col=2)
        
        fig.update_xaxes(title="", showticklabels=False, row=6, col=1)
        fig.update_yaxes(title="", showticklabels=False, row=6, col=1)
        fig.update_xaxes(title="", showticklabels=False, row=6, col=2)
        fig.update_yaxes(title="", showticklabels=False, row=6, col=2)
        
        # --------------------- Layout -------------------- #
        fig.update_layout(
            height=3000,
            width=1600,
            title=f"Cell Type Analysis: {cell_type} (R²={r2:.3f}, Pearson r={pearson_r:.3f})",
            legend=dict(
                y=0.99, x=1.15,
                title="Main Legend",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="Gray",
                borderwidth=1
            ),
            legend2=dict(
                y=0.72, x=1.15,
                title="Magnitude Metrics",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="Gray",
                borderwidth=1
            ),
            legend3=dict(
                y=0.48, x=1.15,
                title="Error Metrics",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="Gray",
                borderwidth=1
            ),
            legend4=dict(
                y=0.24, x=1.15,
                title="AUROC/PR Metrics",
                bgcolor="rgba(255,255,255,0.8)", 
                bordercolor="Gray",
                borderwidth=1
            )
        )
        
        # Save results
        html_file = os.path.join(output_path, f"{cell_type}_evaluation.html")
        csv_file = os.path.join(output_path, f"{cell_type}_metrics.csv")
        print(f"Saving {html_file}")
        fig.write_html(html_file)
        metrics_df.to_csv(csv_file, index=False)
        
        # Save detailed metrics dictionary
        results[cell_type] = {
            'overall_metrics': {
                'r2': r2,
                'pearson_r': pearson_r,
                'spearman_r': spearman_r,
                'mae': mae,
                'rmse': rmse,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'clinical_relevance_score': crs
            },
            'cs_mae': cs_mae,
            'magnitude_metrics': magnitude_metrics,
            'metrics_by_dilution': metrics_df
        }
    
    # Create summary CSV with key metrics for all cell types
    summary_rows = []
    for cell_type, metrics in results.items():
        row = {
            'cell_type': cell_type,
            'r2': metrics['overall_metrics']['r2'],
            'pearson_r': metrics['overall_metrics']['pearson_r'],
            'mae': metrics['overall_metrics']['mae'],
            'roc_auc': metrics['overall_metrics']['roc_auc'],
            'within_10pct_0.5pct': metrics['magnitude_metrics']['threshold_0.005']['within_10pct'],
            'median_rel_error_0.5pct': metrics['magnitude_metrics']['threshold_0.005']['median_rel_error'],
            'clinical_relevance_score': metrics['overall_metrics']['clinical_relevance_score']
        }
        
        for range_name in ['Low (<0.1%)', 'Med (0.1-1%)', 'Med-High (1-5%)', 'High (5-10%)', 'Very High (≥10%)']:
            if range_name in metrics['cs_mae']:
                row[f'mae_{range_name.split()[0].lower()}'] = metrics['cs_mae'][range_name]['mae']
            else:
                row[f'mae_{range_name.split()[0].lower()}'] = None
                
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_path, "all_cell_types_summary.csv"), index=False)
    
    return results


def calculate_metrics_by_dilution(y_true, y_pred, intended_dilutions):
    """Calculate metrics for each dilution level"""
    metrics = []
    for dilution in sorted(np.unique(intended_dilutions)):
        mask = intended_dilutions == dilution
        y_true_dil = y_true[mask]
        y_pred_dil = y_pred[mask]
        
        # Use symmetric relative error to handle small values better
        rel_errors = 2 * (y_pred_dil - y_true_dil) / (y_pred_dil + y_true_dil + 1e-6)
        
        # Calculate percentage within 10% of true value
        within_10pct = np.mean(np.abs(rel_errors) <= 0.1) * 100
        
        # Calculate absolute errors
        abs_errors = np.abs(y_true_dil - y_pred_dil)
        
        # Calculate additional metrics
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(np.square(y_true_dil - y_pred_dil)))
        
        # Calculate bias (mean error)
        bias = np.mean(y_pred_dil - y_true_dil)
        
        # Calculate error variance
        error_variance = np.var(y_pred_dil - y_true_dil)
        
        metrics.append({
            'dilution': dilution,
            'within_10pct': within_10pct,
            'median_rel_error': np.median(rel_errors) * 100,  # Convert to percentage
            'q25_rel_error': np.percentile(rel_errors, 25) * 100,
            'q75_rel_error': np.percentile(rel_errors, 75) * 100,
            'mean_rel_error': np.mean(rel_errors) * 100,
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'error_variance': error_variance,
            'n_samples': len(y_true_dil)
        })
    
    return pd.DataFrame(metrics)

def calculate_metrics_per_dilution(y_true, y_pred, intended_dilution):
    """
    Calculate comprehensive metrics for a specific dilution level
    """
    # R-squared for this dilution
    r2 = r2_score(y_true, y_pred)
    
    # Mean absolute percentage error (excluding very small values)
    significant_mask = y_true > 0.001
    mape = np.mean(np.abs((y_true[significant_mask] - y_pred[significant_mask]) / y_true[significant_mask]))
    
    # Calculate detection metrics
    # True positive: Predicted high when actually high
    # False positive: Predicted high when actually low
    # Using median of true values at this dilution as threshold
    threshold = np.median(y_true)
    
    tp = np.sum((y_pred >= threshold) & (y_true >= threshold))
    fp = np.sum((y_pred >= threshold) & (y_true < threshold))
    tn = np.sum((y_pred < threshold) & (y_true < threshold))
    fn = np.sum((y_pred < threshold) & (y_true >= threshold))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    
    # Correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    return {
        'intended_dilution': intended_dilution,
        'r2': r2,
        'mape': mape,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'correlation': correlation,
        'mean_true': np.mean(y_true),
        'mean_pred': np.mean(y_pred),
        'std_true': np.std(y_true),
        'std_pred': np.std(y_pred),
        'n_samples': len(y_true)
    }


def calculate_performance_metrics(y_true, y_pred):
    """Calculate performance metrics for continuous values"""
    # Overall R² score
    r2 = r2_score(y_true, y_pred)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Relative errors
    rel_errors = (y_pred - y_true) / np.maximum(y_true, 1e-10)
    
    # Percentage within ±20% of true value
    within_10pct = np.mean(np.abs(rel_errors) <= 0.1)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'within_10pct': within_10pct,
        'median_rel_error': np.median(rel_errors),
        'mean_rel_error': np.mean(rel_errors)
    }

def plot_concentrations_by_dilution(df):
    """
    Create a violin plot of contributions by dilution using Plotly
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'dilution' and 'contribution' columns
    """
    
    # Create the violin plot
    fig = go.Figure()
    
    # Add violin plot for each dilution
    for dilution in sorted(df['dilution'].unique()):
        subset = df[df['dilution'] == dilution]['contribution']
        
        fig.add_trace(go.Violin(
            x=df[df['dilution'] == dilution]['dilution'],
            y=subset,
            name=str(dilution),
            box_visible=True,
            meanline_visible=True,
            points='outliers'
        ))
    
    # Update layout
    fig.update_layout(
        title='Distribution of Contributions by Dilution',
        xaxis_title='Dilution',
        yaxis_title='Contribution',
        yaxis_type='log',  # Set y-axis to log scale
        xaxis={'type': 'category', 'categoryorder': 'array', 'categoryarray': sorted(df['dilution'].unique())},  # Set x-axis as categorical
        violinmode='group',
        width=1600,
        height=900,
        showlegend=False
    )
    
    # Update y-axis for better log scale visualization
    fig.update_yaxes(
        type='log',
        exponentformat='power'
    )
    
    return fig