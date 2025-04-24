import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os
import json

def create_visualisations(predictions, ground_truth, output_dir, threshold=0.01, model_name=None):
    """
    Create comprehensive visualisations for model performance analysis with consistent metrics
    
    Args:
        predictions: Array of predicted cell type concentrations
        ground_truth: Array of true cell type concentrations
        output_dir: Directory to save visualisations
        threshold: Default classification threshold for binary metrics (default 0.01 or 1%)
        model_name: Optional name of the model for plot titles
    
    Returns:
        Dictionary of calculated metrics
    """
    # Convert to numpy arrays if not already
    y_pred = np.array(predictions).flatten()
    y_true = np.array(ground_truth).flatten()
    
    # Create dataframe for easier manipulation
    df = pd.DataFrame({
        'true_value': y_true,
        'predicted_value': y_pred,
        'error': y_pred - y_true,
        'abs_error': np.abs(y_pred - y_true),
        'rel_error': np.where(y_true > 0, np.abs(y_pred - y_true) / y_true * 100, np.nan)
    })
    
    # Define concentration ranges
    ranges = [
        ('Low', 0, 0.001),
        ('Medium-Low', 0.001, 0.01),
        ('Medium', 0.01, 0.05),
        ('Medium-High', 0.05, 0.1),
        ('High', 0.1, 0.5),
        ('Very High', 0.5, 1.0)
    ]
    
    # Add range column to dataframe
    df['range'] = 'Unknown'
    for name, lower, upper in ranges:
        mask = (df['true_value'] >= lower) & (df['true_value'] < upper)
        df.loc[mask, 'range'] = name
    
    # Calculate basic metrics
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
    spearman_r = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    # Calculate percentage within error bounds
    within_5pct = np.mean(np.abs(y_pred - y_true) <= 0.05 * np.maximum(y_true, 1e-6)) * 100
    within_10pct = np.mean(np.abs(y_pred - y_true) <= 0.10 * np.maximum(y_true, 1e-6)) * 100
    within_20pct = np.mean(np.abs(y_pred - y_true) <= 0.20 * np.maximum(y_true, 1e-6)) * 100
    
    # Calculate detection metrics with consistent thresholds
    detection_thresholds = [0.001, 0.01, 0.05, 0.1]
    detection_metrics = {}
    
    for thresh in detection_thresholds:
        y_true_binary = (y_true >= thresh).astype(int)
        
        # Skip if no positive examples
        if sum(y_true_binary) == 0:
            continue
            
        # ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Find sensitivity at 95% specificity (5% FPR)
        idx_95spec = np.argmin(np.abs(fpr - 0.05))
        sens_at_95spec = tpr[idx_95spec]
        
        # Precision-recall curve and average precision
        precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, y_pred)
        ap = average_precision_score(y_true_binary, y_pred)
        
        detection_metrics[thresh] = {
            'auc': float(roc_auc),
            'sensitivity_at_95spec': float(sens_at_95spec),
            'average_precision': float(ap),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
    
    # Generate standard visualizations
    create_scatter_plot(df, output_dir)
    create_stratified_mae(df, ranges, output_dir)
    create_relative_error_plot(df, ranges, output_dir) 
    
    # Generate enhanced metrics and plots
    create_enhanced_roc_curve(detection_metrics, output_dir, 
                             title_prefix=model_name if model_name else "")
    create_magnitude_aware_metrics(df, detection_thresholds, output_dir)
    create_error_distribution_plot(df, output_dir)
    
    # Compile metrics dictionary for return
    metrics = {
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mae': float(mae),
        'rmse': float(rmse),
        'within_5pct': float(within_5pct),
        'within_10pct': float(within_10pct),
        'within_20pct': float(within_20pct),
        'detection_metrics': detection_metrics
    }
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'visualization_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def create_scatter_plot(df, output_dir):
    """
    Create scatter plot of predicted vs true values with log scales
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    df_nonzero = df.copy()
    df_nonzero['true_value'] = df_nonzero['true_value'] + epsilon
    df_nonzero['predicted_value'] = df_nonzero['predicted_value'] + epsilon
    
    # Create scatter plot
    fig = px.scatter(
        df_nonzero, 
        x='true_value', 
        y='predicted_value',
        color='range',
        log_x=True, 
        log_y=True,
        opacity=0.7,
        hover_data=['abs_error', 'rel_error']
    )
    
    # Add identity line
    fig.add_trace(
        go.Scatter(
            x=[epsilon, 1],
            y=[epsilon, 1],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect prediction'
        )
    )
    
    # Calculate R² for annotation
    r2 = np.corrcoef(df['true_value'], df['predicted_value'])[0, 1]**2
    
    # Add annotation
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"R² = {r2:.4f}",
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Customise layout
    fig.update_layout(
        title='Predicted vs Ground Truth (Log Scale)',
        xaxis_title='True Value (%)',
        yaxis_title='Predicted Value (%)',
        legend_title='Concentration Range',
        template='plotly_white',
        autosize=False,
        width=900,
        height=700
    )
    
    # Format axes as percentages
    fig.update_xaxes(tickformat='.2%')
    fig.update_yaxes(tickformat='.2%')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'scatter_plot_log.html'))
    fig.write_image(os.path.join(output_dir, 'scatter_plot_log.png'), scale=2)


def create_stratified_mae(df, ranges, output_dir):
    """
    Create bar chart of MAE stratified by concentration range
    """
    # Calculate MAE for each range
    range_stats = []
    for name, lower, upper in ranges:
        mask = (df['true_value'] >= lower) & (df['true_value'] < upper)
        range_df = df[mask]
        if len(range_df) > 0:
            mae = range_df['abs_error'].mean()
            count = len(range_df)
            range_stats.append({
                'range': name,
                'mae': mae,
                'count': count,
                'lower': lower,
                'upper': upper
            })
    
    # Convert to dataframe
    range_df = pd.DataFrame(range_stats)
    
    # Skip if no data
    if len(range_df) == 0:
        return
    
    # Create bar chart
    fig = px.bar(
        range_df,
        x='range',
        y='mae',
        text=range_df['count'].apply(lambda x: f"n={x}"),
        color='mae',
        color_continuous_scale='Viridis',
        labels={'mae': 'Mean Absolute Error', 'range': 'Concentration Range'}
    )
    
    # Customise layout
    fig.update_layout(
        title='Concentration-Stratified MAE',
        xaxis_title='Concentration Range',
        yaxis_title='Mean Absolute Error',
        template='plotly_white',
        autosize=False,
        width=900,
        height=600
    )
    
    # Update text position
    fig.update_traces(textposition='inside')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'stratified_mae.html'))
    fig.write_image(os.path.join(output_dir, 'stratified_mae.png'), scale=2)


def create_relative_error_plot(df, ranges, output_dir):
    """
    Create line plot of relative error by concentration range
    """
    # Calculate relative error metrics for each range
    range_stats = []
    for name, lower, upper in ranges:
        mask = (df['true_value'] >= lower) & (df['true_value'] < upper)
        range_df = df[mask]
        if len(range_df) > 0:
            # Calculate relative error statistics
            median_rel_error = np.nanmedian(range_df['rel_error'])
            mean_rel_error = np.nanmean(range_df['rel_error']) 
            std_rel_error = np.nanstd(range_df['rel_error'])
            within_10pct = np.mean(range_df['rel_error'] <= 10) * 100
            within_20pct = np.mean(range_df['rel_error'] <= 20) * 100
            
            range_stats.append({
                'range': name,
                'median_rel_error': median_rel_error,
                'mean_rel_error': mean_rel_error,
                'std_rel_error': std_rel_error,
                'within_10pct': within_10pct,
                'within_20pct': within_20pct,
                'count': len(range_df),
                'midpoint': (lower + upper) / 2
            })
    
    # Convert to dataframe
    range_df = pd.DataFrame(range_stats)
    
    # Skip if no data
    if len(range_df) == 0:
        return
    
    # Create figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add relative error line
    fig.add_trace(
        go.Scatter(
            x=range_df['midpoint'],
            y=range_df['mean_rel_error'],
            error_y=dict(
                type='data',
                array=range_df['std_rel_error'] / np.sqrt(range_df['count']),
                visible=True
            ),
            mode='lines+markers',
            name='Mean Relative Error (%)',
            line=dict(color='red', width=2)
        ),
        secondary_y=False
    )
    
    # Add within 10% line
    fig.add_trace(
        go.Scatter(
            x=range_df['midpoint'],
            y=range_df['within_10pct'],
            mode='lines+markers',
            name='Within 10% Error',
            line=dict(color='blue', width=2)
        ),
        secondary_y=True
    )
    
    # Add reference lines
    fig.add_hline(y=0, line=dict(color='black', dash='dot'), secondary_y=False)
    fig.add_hline(y=20, line=dict(color='gray', dash='dash'), secondary_y=False)
    fig.add_hline(y=-20, line=dict(color='gray', dash='dash'), secondary_y=False)
    
    # Customise layout
    fig.update_layout(
        title='Relative Error by Concentration Range',
        xaxis=dict(
            title='Concentration (%)',
            type='log',
            tickformat='.4%'
        ),
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        autosize=False,
        width=900,
        height=600
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Relative Error (%)", secondary_y=False)
    fig.update_yaxes(title_text="Percentage Within Tolerance", secondary_y=True)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'relative_error_by_range.html'))
    fig.write_image(os.path.join(output_dir, 'relative_error_by_range.png'), scale=2)


def create_enhanced_roc_curve(detection_metrics, output_dir, title_prefix=""):
    """
    Create enhanced ROC curve with consistent metrics from detection_metrics 
    """
    # Create figure
    fig = go.Figure()
    
    # Add diagonal reference line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', dash='dash'),
            showlegend=False
        )
    )
    
    # Add ROC curves for each threshold from the pre-calculated metrics
    for threshold, metrics in sorted(detection_metrics.items()):
        if 'fpr' in metrics and 'tpr' in metrics:
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            auc_value = metrics['auc']
            
            # Add to plot
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'≥{threshold:.1%}, AUC={auc_value:.3f}',
                    line=dict(width=2)
                )
            )
    
    # Create summary text with metrics
    primary_threshold = 0.01  # Default to 1% as primary threshold
    
    # Use the provided metrics if available, otherwise use a default message
    if primary_threshold in detection_metrics:
        metrics = detection_metrics[primary_threshold]
        
        # Create annotation text
        annotation_text = (
            f"<b>ROC Analysis Summary</b><br>"
            f"AUC at {primary_threshold:.1%}: {metrics['auc']:.3f}<br>"
            f"Sensitivity at 95% specificity: {metrics['sensitivity_at_95spec']:.3f}<br>"
        )
        
        # Add metrics for each threshold
        for thresh, thresh_metrics in sorted(detection_metrics.items()):
            annotation_text += (
                f"For ≥{thresh:.1%}: "
                f"AUC = {thresh_metrics['auc']:.3f}, "
                f"Sens@95%Spec = {thresh_metrics['sensitivity_at_95spec']:.3f}<br>"
            )
    else:
        annotation_text = "Metrics not available for standard thresholds"
    
    # Add annotation to figure
    fig.add_annotation(
        x=0.5,
        y=0.1,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )
    
    # Customise layout
    title = "ROC Curve by Concentration Threshold"
    if title_prefix:
        title = f"{title_prefix} - {title}"
        
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        autosize=False,
        width=900,
        height=700,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'enhanced_roc_curve.html'))
    fig.write_image(os.path.join(output_dir, 'enhanced_roc_curve.png'), scale=2)


def create_magnitude_aware_metrics(df, thresholds, output_dir):
    """
    Create metrics and plot for magnitude-aware classification performance.
    This considers both binary detection and the magnitude of errors.
    """
    # Define error tolerance levels (as fraction of true value)
    error_tolerances = [0.1, 0.2, 0.5, 1.0, 2.0]  # 10%, 20%, 50%, 100%, 200%
    
    # Initialize results dictionary
    results = {}
    
    # For each concentration threshold
    for threshold in thresholds:
        tolerance_results = {}
        
        # For each error tolerance
        for tolerance in error_tolerances:
            # True positive: predicted ≥ threshold when true ≥ threshold AND within tolerance
            tp_mask = (df['true_value'] >= threshold) & (df['predicted_value'] >= threshold) & \
                     (df['rel_error'] <= tolerance * 100)
            
            # False positive: predicted ≥ threshold when true < threshold OR exceeds tolerance
            fp_mask = ((df['true_value'] < threshold) & (df['predicted_value'] >= threshold)) | \
                     ((df['true_value'] >= threshold) & (df['predicted_value'] >= threshold) & \
                      (df['rel_error'] > tolerance * 100))
            
            # True negative: predicted < threshold when true < threshold
            tn_mask = (df['true_value'] < threshold) & (df['predicted_value'] < threshold)
            
            # False negative: predicted < threshold when true ≥ threshold
            fn_mask = (df['true_value'] >= threshold) & (df['predicted_value'] < threshold)
            
            # Calculate counts
            tp = tp_mask.sum()
            fp = fp_mask.sum()
            tn = tn_mask.sum()
            fn = fn_mask.sum()
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # Store metrics
            tolerance_results[tolerance] = {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'f1_score': float(f1_score),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
        
        results[threshold] = tolerance_results
    
    # Create plot for key threshold (1%)
    create_magnitude_aware_plot(results, output_dir)
    
    # Save results to file
    magnitude_metrics_file = os.path.join(output_dir, 'magnitude_aware_metrics.json')
    with open(magnitude_metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_magnitude_aware_plot(results, output_dir):
    """
    Create plot for magnitude-aware classification metrics
    """
    # Select a primary threshold for visualization (typically 1%)
    primary_threshold = 0.01
    if primary_threshold not in results:
        # Use the first available threshold
        primary_threshold = list(results.keys())[0]
    
    # Prepare data for plotting
    tolerance_metrics = results[primary_threshold]
    tolerances = sorted(float(t) for t in tolerance_metrics.keys())
    
    sensitivity = [tolerance_metrics[t]['sensitivity'] for t in tolerances]
    specificity = [tolerance_metrics[t]['specificity'] for t in tolerances]
    precision = [tolerance_metrics[t]['precision'] for t in tolerances]
    f1_score = [tolerance_metrics[t]['f1_score'] for t in tolerances]
    
    # Convert tolerances to percentage labels for x-axis
    tolerance_labels = [f"{t*100:.0f}%" for t in tolerances]
    
    # Create figure
    fig = go.Figure()
    
    # Add metric lines
    fig.add_trace(
        go.Scatter(
            x=tolerance_labels,
            y=sensitivity,
            mode='lines+markers',
            name='Sensitivity',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=tolerance_labels,
            y=specificity,
            mode='lines+markers',
            name='Specificity',
            line=dict(color='red', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=tolerance_labels,
            y=precision,
            mode='lines+markers',
            name='Precision',
            line=dict(color='green', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=tolerance_labels,
            y=f1_score,
            mode='lines+markers',
            name='F1 Score',
            line=dict(color='purple', width=2)
        )
    )
    
    # Customize layout
    fig.update_layout(
        title=f'Magnitude-Aware Classification Metrics at {primary_threshold:.1%} Threshold',
        xaxis_title='Error Tolerance',
        yaxis_title='Metric Value',
        template='plotly_white',
        autosize=False,
        width=900,
        height=600,
        yaxis=dict(range=[0, 1]),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'magnitude_aware_metrics.html'))
    fig.write_image(os.path.join(output_dir, 'magnitude_aware_metrics.png'), scale=2)


def create_error_distribution_plot(df, output_dir):
    """
    Create error distribution plot to show both absolute and relative errors
    """
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Absolute Error Distribution', 'Relative Error Distribution'))
    
    # Add absolute error histogram
    fig.add_trace(
        go.Histogram(
            x=df['abs_error'],
            nbinsx=50,
            name='Absolute Error',
            marker_color='blue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add vertical line at median and mean
    median_abs_error = df['abs_error'].median()
    mean_abs_error = df['abs_error'].mean()
    
    fig.add_vline(x=median_abs_error, line=dict(color="red", dash="dash"), 
                 annotation_text=f"Median: {median_abs_error:.6f}", row=1, col=1)
    fig.add_vline(x=mean_abs_error, line=dict(color="green", dash="dash"), 
                 annotation_text=f"Mean: {mean_abs_error:.6f}", row=1, col=1)
    
    # Add relative error histogram (cap extreme values)
    rel_error_capped = df['rel_error'].clip(-1000, 1000)  # Cap at ±1000%
    rel_error_valid = rel_error_capped.dropna()
    
    fig.add_trace(
        go.Histogram(
            x=rel_error_valid,
            nbinsx=50,
            name='Relative Error',
            marker_color='orange',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add vertical line at median and mean
    median_rel_error = rel_error_valid.median()
    mean_rel_error = rel_error_valid.mean()
    
    fig.add_vline(x=median_rel_error, line=dict(color="red", dash="dash"), 
                 annotation_text=f"Median: {median_rel_error:.1f}%", row=2, col=1)
    fig.add_vline(x=mean_rel_error, line=dict(color="green", dash="dash"), 
                 annotation_text=f"Mean: {mean_rel_error:.1f}%", row=2, col=1)
    
    # Add reference lines at ±20%
    fig.add_vline(x=20, line=dict(color="gray", dash="dot"), row=2, col=1)
    fig.add_vline(x=-20, line=dict(color="gray", dash="dot"), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title='Error Distribution Analysis',
        template='plotly_white',
        height=800,
        width=900,
        showlegend=False
    )
    
    # Update x-axis ranges
    fig.update_xaxes(title_text="Absolute Error", row=1, col=1)
    fig.update_xaxes(title_text="Relative Error (%)", row=2, col=1, range=[-100, 100])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'error_distribution.html'))
    fig.write_image(os.path.join(output_dir, 'error_distribution.png'), scale=2)


def visualize_test_results(results_file, output_dir):
    """
    Visualize test results from a results.json file
    
    Args:
        results_file: Path to test_results.json file
        output_dir: Directory to save visualizations
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    
    # Create visualizations
    return create_visualisations(predictions, targets, output_dir)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Create enhanced visualizations for model evaluation')
    parser.add_argument('--results_file', type=str, help='Path to test_results.json file')
    parser.add_argument('--output_dir', type=str, default='enhanced_visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    if args.results_file:
        metrics = visualize_test_results(args.results_file, args.output_dir)
        print("Enhanced visualizations saved to", args.output_dir)
        print(f"Key metrics: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.6f}")
    else:
        print("Please provide a results file with --results_file")