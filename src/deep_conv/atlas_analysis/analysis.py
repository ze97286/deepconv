import argparse
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from deep_conv.benchmark.benchmark_utils import process_atlas


def generate_realistic_dilution_mixtures(H, diluted_type_idx, n_samples=100, 
                                       dilution=0.01, mean_coverage=100,
                                       zero_coverage_rate=0.1):
    """
    Generate synthetic mixtures with realistic noise and coverage patterns
    
    Args:
        H: Reference profiles matrix (n_components x n_markers)
        diluted_type_idx: Index of cell type to dilute
        n_samples: Number of samples to generate
        dilution: Proportion for diluted cell type
        mean_coverage: Average coverage to simulate
        zero_coverage_rate: Proportion of markers with zero coverage
    """
    n_components, n_markers = H.shape
    
    # Generate base proportions (W)
    W = np.zeros((n_samples, n_components))
    W[:, diluted_type_idx] = dilution
    
    # Distribute remaining proportion
    remaining_proportion = 1 - dilution
    for i in range(n_samples):
        other_indices = [j for j in range(n_components) if j != diluted_type_idx]
        other_props = np.random.uniform(0.25, 0.35, len(other_indices))
        other_props = other_props / other_props.sum() * remaining_proportion
        W[i, other_indices] = other_props
    
    # Generate ideal methylation patterns
    X_ideal = np.dot(W, H)
    
    # Generate coverage values
    coverage = np.random.negative_binomial(n=5, p=5/(5+mean_coverage), 
                                         size=(n_samples, n_markers))
    
    # Add zero coverage spots
    zero_mask = np.random.random((n_samples, n_markers)) < zero_coverage_rate
    coverage[zero_mask] = 0
    
    # Generate observed counts based on coverage and methylation rate
    methylated_counts = np.random.binomial(coverage, X_ideal)
    
    # Calculate observed methylation proportions
    X_observed = np.zeros_like(X_ideal)
    nonzero_coverage = coverage > 0
    X_observed[nonzero_coverage] = methylated_counts[nonzero_coverage] / coverage[nonzero_coverage]
    X_observed[~nonzero_coverage] = np.nan
    
    return X_observed, W, coverage


def evaluate_cell_type_detection(H, solver, cell_types, dilution, n_repeats=5):
    """
    Evaluate detection with proper R² calculation
    """
    n_components = H.shape[0]
    all_results = []
    
    for idx in range(n_components):
        cell_results = []
        
        all_true_props = []  # Collect all true proportions
        all_pred_props = []  # Collect all predictions
        
        for _ in range(n_repeats):
            X, W_true, coverage = generate_realistic_dilution_mixtures(H, idx, dilution=dilution)
            W_pred = solver(X, coverage)
            
            true_prop = W_true[:, idx]
            pred_prop = W_pred[:, idx]
            
            # Collect all values
            all_true_props.extend(true_prop)
            all_pred_props.extend(pred_prop)
            
            result = {
                'cell_type': cell_types[idx],
                'mse': np.mean((true_prop - pred_prop)**2),
                'ratio_error': np.mean(pred_prop[true_prop > 0] / true_prop[true_prop > 0]),
                'relative_error': np.mean(np.abs(pred_prop - true_prop) / true_prop),
                'mean_pred': np.mean(pred_prop),
                'std_pred': np.std(pred_prop)
            }
            cell_results.append(result)
        
        # Calculate R² using all predictions across repeats
        all_true_props = np.array(all_true_props)
        all_pred_props = np.array(all_pred_props)
        
        # Use 1 - MSE/variance of predictions as R² measure
        # This gives us a measure of prediction quality relative to prediction variance
        mse = np.mean((all_true_props - all_pred_props)**2)
        pred_var = np.var(all_pred_props)
        r2 = 1 - mse/pred_var if pred_var > 0 else 0
        
        # Average results across repeats
        avg_result = {
            'cell_type_idx': idx,
            'cell_type': cell_types[idx],
            'mse': np.mean([r['mse'] for r in cell_results]),
            'ratio_error': np.mean([r['ratio_error'] for r in cell_results]),
            'relative_error': np.mean([r['relative_error'] for r in cell_results]),
            'mean_pred': np.mean([r['mean_pred'] for r in cell_results]),
            'std_pred': np.mean([r['std_pred'] for r in cell_results]),
            'r2': r2
        }
        all_results.append(avg_result)
    
    return pd.DataFrame(all_results)


def plot_detection_analysis(H, results, cell_types):
    """
    Create visualization with 2x2 layout (removed Correlation vs Error)
    """
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(H)
    
    # Create figure with 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Metrics by Cell Type', 
                    'Prediction Statistics',
                    'Cell Type Correlation Matrix',
                    'Error Analysis'),
        vertical_spacing=0.25,    
        column_widths=[0.5, 0.5], 
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}]
        ],
        column_width=[0.5, 0.5],  
        horizontal_spacing=0.2    
    )
    
    # Sort data
    metrics_data = results.sort_values('relative_error')
    
    # 1. Performance Metrics
    fig.add_trace(
        go.Bar(name='Relative Error', 
               x=metrics_data['cell_type'],
               y=metrics_data['relative_error'],
               marker_color='rgb(158,202,225)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Ratio Error',
               x=metrics_data['cell_type'],
               y=metrics_data['ratio_error'],
               marker_color='rgb(94,158,217)'),
        row=1, col=1
    )
    
    # 2. Prediction Statistics
    fig.add_trace(
        go.Bar(name='Mean Prediction',
               x=metrics_data['cell_type'],
               y=metrics_data['mean_pred'],
               marker_color='rgb(222,165,164)'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name='Std Prediction',
               x=metrics_data['cell_type'],
               y=metrics_data['std_pred'],
               marker_color='rgb(212,105,104)'),
        row=1, col=2
    )
    
    # 3. Correlation Heatmap with centered colorbar
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix,
            x=cell_types,
            y=cell_types,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title='Correlation',
                len=0.5,          
                y=0.25,           
                yanchor='middle',
                x=-0.15          
            )
        ),
        row=2, col=1
    )
    
    # 4. Error Analysis (Relative Error vs Ratio Error)
    fig.add_trace(
        go.Scatter(
            x=results['relative_error'],
            y=results['ratio_error'],
            mode='markers+text',
            text=results['cell_type'],
            textposition="top center",
            marker=dict(
                size=10,
                color=results['mean_pred'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Mean Prediction')
            ),
            name='Error Types'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1000, 
        width=1750,  
        title_text="Cell Type Detection Analysis",
        title_x=0.5,            
        showlegend=True,
        template='plotly_white',
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=0,
        ),
        margin=dict(           
            l=200,              # Reduced from 350
            r=50,               # Reduced from 100
            t=100,              # Reduced from 150
            b=50                # Reduced from 100
        )
    )
    
    # Update all axes for better spacing
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(
                row=row, 
                col=col,
                tickangle=45,
                tickfont=dict(size=11),
                title_font=dict(size=13)
            )
            fig.update_yaxes(
                row=row,
                col=col,
                tickfont=dict(size=11),
                title_font=dict(size=13)
            )
    
    # Update specific axis labels
    fig.update_xaxes(title_text="Cell Type", row=1, col=1)
    fig.update_xaxes(title_text="Cell Type", row=1, col=2)
    fig.update_xaxes(title_text="Cell Type", row=2, col=1)
    fig.update_xaxes(title_text="Relative Error", row=2, col=2)
    
    fig.update_yaxes(title_text="Error Value", row=1, col=1)
    fig.update_yaxes(title_text="Prediction Value", row=1, col=2)
    fig.update_yaxes(title_text="Cell Type", row=2, col=1)
    fig.update_yaxes(title_text="Ratio Error", row=2, col=2)
    
    return fig


def nnls_solver(X, coverage):
    """NNLS solver that handles missing values"""
    results = []
    for x, c in zip(X, coverage):
        # Replace NaNs with 0 where coverage is 0
        x_clean = np.where(c > 0, x, 0)
        w, _ = nnls(H.T, x_clean)
        w = w / np.sum(w)
        results.append(w)
    return np.array(results)


def generate_dilution_predictions(H, solver, cell_type_idx, dilution, n_samples=100):
    """Get predictions from a single batch"""
    X, W_true, coverage = generate_realistic_dilution_mixtures(H, cell_type_idx, dilution=dilution,n_samples=n_samples)
    W_pred = solver(X, coverage)
    return W_pred[:, cell_type_idx]


def plot_prediction_distributions(H, solver, cell_types, dilution):
    """
    Create distribution plot for any dilution level
    
    Args:
        H: Reference profiles matrix
        solver: Solver function to use
        cell_types: List of cell type names
        dilution: Dilution level to analyze (e.g., 0.01 for 1%)
    """
    # Generate predictions for each cell type
    predictions_data = []
    for i, cell_type in enumerate(cell_types):
        pred = generate_dilution_predictions(H, solver, i, dilution)
        predictions_data.append(go.Box(
            y=pred,
            name=cell_type,
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8,
            marker_color='rgb(93, 164, 214)',
            showlegend=False
        ))
    
    # Calculate appropriate y-axis range based on dilution
    y_max = max(dilution * 3, dilution + np.std([p.y for p in predictions_data]) * 3)
    y_min = max(0, dilution - np.std([p.y for p in predictions_data]) * 2)
    
    # Create figure
    fig = go.Figure(data=predictions_data)
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        title_text=f"Prediction Distributions by Cell Type (Dilution: {dilution:.1%})",
        template='plotly_white',
        margin=dict(t=100, l=100, r=100, b=150),
        yaxis=dict(
            title="Predicted Proportion",
            range=[y_min, y_max],
            tickformat='.1%',  # Format as percentage
            tickfont=dict(size=12)
        ),
        xaxis=dict(
            title="Cell Type",
            tickangle=45,
            tickfont=dict(size=12)
        ),
        shapes=[
            dict(
                type='line',
                yref='y',
                y0=dilution,
                y1=dilution,
                xref='paper',
                x0=0,
                x1=1,
                line=dict(
                    color='red',
                    dash='dash',
                    width=2
                )
            )
        ],
        annotations=[
            dict(
                text=f"True Proportion ({dilution:.1%})",
                x=1.02,
                y=dilution,
                xref='paper',
                yref='y',
                showarrow=False,
                textangle=0
            )
        ]
    )
    
    return fig


def visualise_cell_type_detection(H, results):
    # Select top 3 and bottom 3 based on MSE
    sorted_results = results.sort_values('mse')
    top_3 = sorted_results.head(3)['cell_type'].values
    bottom_3 = sorted_results.tail(3)['cell_type'].values
    top_3_idx = sorted_results.head(3)['cell_type_idx'].values
    bottom_3_idx = sorted_results.tail(3)['cell_type_idx'].values
    
    # Create figure with subplots
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Methylation Profiles', 'Detection Performance'),
                       vertical_spacing=0.2,
                       row_heights=[0.6, 0.4])
    
    # Plot methylation profiles
    for idx, name in zip(top_3_idx, top_3):
        fig.add_trace(
            go.Scatter(y=H[idx], name=f'Type {name} (Easy)',
                      line=dict(width=2)),
            row=1, col=1
        )
    
    for idx, name in zip(bottom_3_idx, bottom_3):
        fig.add_trace(
            go.Scatter(y=H[idx], name=f'Type {name} (Hard)',
                      line=dict(dash='dash')),
            row=1, col=1
        )
    
    # Performance metrics bar plot
    metrics = ['relative_error', 'ratio_error']
    cell_types = np.concatenate([top_3, bottom_3])
    x_positions = np.arange(len(cell_types))
    
    for metric in metrics:
        values = [results[results['cell_type'] == ct][metric].values[0] 
                 for ct in cell_types]
        
        fig.add_trace(
            go.Bar(x=[f'Type {ct}' for ct in cell_types],
                  y=values,
                  name=metric,
                  visible='legendonly' if metric != 'relative_error' else True),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        title='Cell Type Detection Analysis',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    fig.update_xaxes(title_text="Marker Index", row=1, col=1)
    fig.update_yaxes(title_text="Methylation Level", row=1, col=1)
    fig.update_xaxes(title_text="Cell Type", row=2, col=1)
    fig.update_yaxes(title_text="Metric Value", row=2, col=1)
    
    # Add a horizontal line at y=1 for ratio error reference
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment: Weighted NNLS optimisation with R² Metric for Cell Type Deconvolution")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to the altas")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the save figures")
    args = parser.parse_args()

    H,cell_types = process_atlas(args.atlas_path)
    for dilution in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]:
        results = evaluate_cell_type_detection(H, nnls_solver, cell_types, dilution)
        fig = visualise_cell_type_detection(H, results)
        fig.write_html(args.output_path+f"atlas_{dilution}_analysis_by_cell_type.html")

        fig = plot_detection_analysis(H, results, cell_types)
        fig.write_html(args.output_path+f"atlas_{dilution}_detection_analysis_by_cell_type.html")

        fig = plot_prediction_distributions(H, nnls_solver, cell_types, dilution)
        fig.write_html(args.output_path+f"atlas_{dilution}_distribution_by_cell_type.html")


if __name__ == "__main__":
    main()