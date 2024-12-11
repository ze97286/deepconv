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
    cell_types: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate performance metrics between true and estimated proportions.

    Args:
        true_proportions (np.ndarray): True cell type proportions (n_samples x n_cell_types).
        estimated_proportions (np.ndarray): Estimated cell type proportions (n_samples x n_cell_types).
        cell_types (List[str]): List of cell type names.

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary containing metrics per cell type and overall.
    """
    metrics_per_cell = {}
    for i, cell_type in enumerate(cell_types):
        true = true_proportions[:, i]
        est = estimated_proportions[:, i]
        rmse = np.sqrt(mean_squared_error(true, est))
        mae = mean_absolute_error(true, est)
        # Handle cases where the variance is zero to avoid NaN in correlation
        if np.std(est) == 0 or np.std(true) == 0:
            corr = 0.0
            r2 = 0.0
            logger.warning(f"Zero variance detected for cell type '{cell_type}'. Setting Correlation and R² to 0.")
        else:
            corr, _ = pearsonr(true, est)
            r2 = r2_score(true, est)
        metrics_per_cell[cell_type] = {
            'RMSE': rmse,
            'MAE': mae,
            'Pearson Correlation': corr,
            'R²': r2
        }
    
    # Overall metrics
    overall_rmse = np.sqrt(mean_squared_error(true_proportions, estimated_proportions))
    overall_mae = mean_absolute_error(true_proportions, estimated_proportions)
    
    # Compute R² for each cell type and average
    r2_scores = []
    for i in range(len(cell_types)):
        true = true_proportions[:, i]
        est = estimated_proportions[:, i]
        if np.std(est) == 0 or np.std(true) == 0:
            r2 = 0.0
        else:
            r2 = r2_score(true, est)
        r2_scores.append(r2)
    overall_r2 = np.mean(r2_scores)
    
    # Compute Pearson correlations for each cell type and average
    correlations = []
    for i in range(len(cell_types)):
        true = true_proportions[:, i]
        est = estimated_proportions[:, i]
        if np.std(est) == 0 or np.std(true) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(true, est)
        correlations.append(corr)
    overall_corr = np.mean(correlations)
    
    metrics_overall = {
        'Overall RMSE': overall_rmse,
        'Overall MAE': overall_mae,
        'Overall R²': overall_r2,
        'Average Pearson Correlation': overall_corr
    }
    
    range_metrics = analyse_proportion_ranges(pred=estimated_proportions, true=true_proportions, cell_types=cell_types)
    header = (f"{'Cell Type':<20} {'Range':<25} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'N':<8}")
    print(header)
    print("-" * len(header))
    
    for cell_type in cell_types:
        if cell_type in range_metrics:
            cell_range_metrics = range_metrics[cell_type]
            for range_name, range_m in cell_range_metrics.items():
                mae = range_m.get('mae', np.nan)
                rmse = range_m.get('rmse', np.nan)
                r2 = range_m.get('r2', np.nan)
                n = range_m.get('n_samples', 0)
                r2_str = '-'.rjust(7) if np.isnan(r2) else f"{r2:7.4f}"
                print(f"{cell_type[:20]:<20} "
                        f"{range_name:<25} "
                        f"{mae:7.4f} "
                        f"{rmse:7.4f} "
                        f"{r2_str} "
                        f"{n:7}")
                    
    return {'Per_Cell_Type': metrics_per_cell, 'Overall': metrics_overall}


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
        f"d0_{i}" for i in range(1, 101)
    ] + [
        f"d1e-04_{i}" for i in range(1, 101)
    ] + [
        f"d0.001_{i}" for i in range(1, 101)
    ] + [
        f"d0.005_{i}" for i in range(1, 101)
    ] + [
        f"d0.01_{i}" for i in range(1, 101)
    ] + [
        f"d0.05_{i}" for i in range(1, 101)
    ] + [
        f"d0.1_{i}" for i in range(1, 101)
    ]

def dilutions():
    return [
        0 for i in range(1, 101)
    ] + [
        1e-4 for i in range(1, 101)
    ] + [
        1e-3 for i in range(1, 101)
    ] + [
        0.005 for i in range(1, 101)
    ] + [
        0.01 for i in range(1, 101)
    ] + [
        0.05 for i in range(1, 101)
    ] + [
        0.1 for i in range(1, 101)
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
    # Ensure all cell types are included
    all_cell_types = all_predictions_df.columns.tolist()
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
    return fig, metrics_df



    
    

    
    