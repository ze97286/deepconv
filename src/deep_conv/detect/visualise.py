import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def create_visualisations(predictions, ground_truth, output_dir, threshold=0.5):
    """
    Create comprehensive visualisations for model performance analysis
    
    Args:
        predictions: Array of predicted cell type concentrations
        ground_truth: Array of true cell type concentrations
        output_dir: Directory to save visualisations
        threshold: Classification threshold for binary metrics
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
        'rel_error': np.where(y_true > 0, (y_pred - y_true) / y_true * 100, np.nan)
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
    
    # Calculate metrics
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
    spearman_r = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100
    
    # Calculate percentage within error bounds
    within_5pct = np.mean(np.abs(y_pred - y_true) <= 0.05 * np.maximum(y_true, 1e-6)) * 100
    within_10pct = np.mean(np.abs(y_pred - y_true) <= 0.10 * np.maximum(y_true, 1e-6)) * 100
    within_20pct = np.mean(np.abs(y_pred - y_true) <= 0.20 * np.maximum(y_true, 1e-6)) * 100
    
    # Create visualisations
    create_scatter_plot(df, output_dir)
    create_stratified_mae(df, ranges, output_dir)
    create_relative_error_plot(df, ranges, output_dir)
    create_roc_curve(y_true, y_pred, threshold, output_dir)
    create_error_by_range_boxplot(df, output_dir)
    create_precision_recall_curve(y_true, y_pred, threshold, output_dir)
    
    # Print summary statistics
    print(f"Performance Summary:")
    print(f"R² Score: {r2:.4f}")
    print(f"Pearson correlation: {pearson_r:.4f}")
    print(f"Spearman correlation: {spearman_r:.4f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Within 5% error: {within_5pct:.2f}%")
    print(f"Within 10% error: {within_10pct:.2f}%")
    print(f"Within 20% error: {within_20pct:.2f}%")
    
    # Save metrics to file
    metrics = {
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'within_5pct': float(within_5pct),
        'within_10pct': float(within_10pct),
        'within_20pct': float(within_20pct)
    }
    
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
    fig.write_html(f"{output_dir}/scatter_plot_log.html")
    fig.write_image(f"{output_dir}/scatter_plot_log.png", scale=2)


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
    fig.write_html(f"{output_dir}/stratified_mae.html")
    fig.write_image(f"{output_dir}/stratified_mae.png", scale=2)


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
            median_rel_error = range_df['rel_error'].median()
            mean_rel_error = range_df['rel_error'].mean()
            std_rel_error = range_df['rel_error'].std()
            within_10pct = np.mean(np.abs(range_df['rel_error']) <= 10) * 100
            within_20pct = np.mean(np.abs(range_df['rel_error']) <= 20) * 100
            
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
            name='Relative Error (%)',
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
        title='Relative Error Rate',
        xaxis=dict(
            title='Intended Dilution (%)',
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
    fig.update_yaxes(title_text="Percentage / Error", secondary_y=False)
    fig.update_yaxes(title_text="Percentage Within Tolerance", secondary_y=True)
    
    # Save figure
    fig.write_html(f"{output_dir}/relative_error_by_range.html")
    fig.write_image(f"{output_dir}/relative_error_by_range.png", scale=2)


def create_roc_curve(y_true, y_pred, threshold, output_dir):
    """
    Create ROC curve for various concentration thresholds
    """
    # Create classification targets at different thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    # Create figure
    fig = go.Figure()
    
    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Chance',
            line=dict(color='navy', dash='dash'),
            showlegend=False
        )
    )
    
    # Add ROC curves for each threshold
    results = []
    for thresh in thresholds:
        y_true_binary = (y_true >= thresh).astype(int)
        if sum(y_true_binary) > 0:  # Only calculate if we have positive examples
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
            roc_auc = auc(fpr, tpr)
            
            # Add to results
            results.append({
                'threshold': thresh,
                'roc_auc': roc_auc,
                # Calculate sensitivity and specificity at the detection threshold
                'sensitivity': tpr[np.argmax(fpr >= 0.05)],
                'specificity': 1 - fpr[np.argmax(fpr >= 0.05)]
            })
            
            # Add to plot
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'≥{thresh:.1%}, AUC={roc_auc:.3f}',
                    line=dict(width=2)
                )
            )
    
    # Create annotation with summary metrics
    primary_thresh = 0.01  # 1% concentration as primary threshold
    primary_results = next((r for r in results if r['threshold'] == primary_thresh), results[0])
    
    # Calculate overall metrics
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
    spearman_r = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    # Create annotation text
    annotation_text = (
        f"<b>Summary Metrics for T-cells</b><br>"
        f"R²: {r2:.3f} | Pearson r: {pearson_r:.3f} | Spearman r: {spearman_r:.3f}<br>"
        f"MAE: {mae:.5f} | RMSE: {rmse:.5f}<br>"
        f"ROC-AUC: {primary_results['roc_auc']:.3f}<br>"
    )
    
    # Add detection metrics for each threshold
    for result in results:
        annotation_text += (
            f"Detection at {result['threshold']:.1%}: "
            f"{result['sensitivity']:.3f} sens, {result['specificity']:.3f} spec<br>"
        )
    
    # Add a clinical relevance score (simplified example)
    clinical_score = primary_results['roc_auc'] * 30 + (1 - mae) * 70
    annotation_text += f"<b>Clinical Relevance Score: {clinical_score:.1f}/100</b>"
    
    # Add annotation to figure
    fig.add_annotation(
        x=0.5,
        y=0.25,
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
    fig.update_layout(
        title='Concentration Range<br>ROC Curve',
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
    fig.write_html(f"{output_dir}/roc_curve.html")
    fig.write_image(f"{output_dir}/roc_curve.png", scale=2)


def create_error_by_range_boxplot(df, output_dir):
    """
    Create boxplot of relative error by concentration range
    """
    # Create figure
    fig = go.Figure()
    
    # Add boxplots for each range
    for range_name in df['range'].unique():
        range_df = df[df['range'] == range_name]
        if len(range_df) > 0:
            fig.add_trace(
                go.Box(
                    y=range_df['rel_error'],
                    name=range_name,
                    boxmean=True  # Show mean as a dashed line
                )
            )
    
    # Add reference line at 0
    fig.add_hline(y=0, line=dict(color='black', dash='dot'))
    
    # Customise layout
    fig.update_layout(
        title='Error by Concentration Range',
        xaxis_title='Concentration Range',
        yaxis_title='Relative Error (%)',
        template='plotly_white',
        autosize=False,
        width=900,
        height=600,
        yaxis=dict(
            range=[-200, 200]  # Limit y range for better visualisation
        )
    )
    
    # Add second y-axis for precision
    fig.update_layout(
        yaxis2=dict(
            title="Precision",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            anchor="x",
            overlaying="y",
            side="right",
            position=1.0
        )
    )
    
    # Save figure
    fig.write_html(f"{output_dir}/error_by_range_boxplot.html")
    fig.write_image(f"{output_dir}/error_by_range_boxplot.png", scale=2)


def create_precision_recall_curve(y_true, y_pred, threshold, output_dir):
    """
    Create precision-recall curve for various concentration thresholds
    """
    # Create classification targets at different thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    # Create figure
    fig = go.Figure()
    
    # Add precision-recall curves for each threshold
    for thresh in thresholds:
        y_true_binary = (y_true >= thresh).astype(int)
        if sum(y_true_binary) > 0:  # Only calculate if we have positive examples
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred)
            ap = average_precision_score(y_true_binary, y_pred)
            
            # Add to plot
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f'≥{thresh:.1%}, AP={ap:.3f}',
                    line=dict(width=2)
                )
            )
    
    # Add reference line for random classifier
    baseline = sum(y_true >= threshold) / len(y_true)
    fig.add_hline(y=baseline, line=dict(color='gray', dash='dash'))
    
    # Customise layout
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_white',
        autosize=False,
        width=900,
        height=600,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save figure
    fig.write_html(f"{output_dir}/precision_recall_curve.html")
    fig.write_image(f"{output_dir}/precision_recall_curve.png", scale=2)


# Example usage
if __name__ == "__main__":
    import argparse
    import os
    import json
    
    parser = argparse.ArgumentParser(description='Create visualisations for model evaluation')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions JSON or CSV')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth JSON or CSV')
    parser.add_argument('--output_dir', type=str, default='visualisations', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions and ground truth
    if args.predictions.endswith('.json'):
        with open(args.predictions, 'r') as f:
            predictions_data = json.load(f)
        
        if isinstance(predictions_data, dict) and 'predictions' in predictions_data:
            predictions = predictions_data['predictions']
        else:
            predictions = predictions_data
    else:
        predictions = pd.read_csv(args.predictions).values.flatten()
    
    if args.ground_truth.endswith('.json'):
        with open(args.ground_truth, 'r') as f:
            ground_truth_data = json.load(f)
        
        if isinstance(ground_truth_data, dict) and 'ground_truth' in ground_truth_data:
            ground_truth = ground_truth_data['ground_truth']
        else:
            ground_truth = ground_truth_data
    else:
        ground_truth = pd.read_csv(args.ground_truth).values.flatten()
    
    # Create visualisations
    metrics = create_visualisations(predictions, ground_truth, args.output_dir)
    
    # Save metrics
    with open(f"{args.output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)