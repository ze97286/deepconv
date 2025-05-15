import os
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
from deep_conv.detect.preprocess import *
from deep_conv.detect.predict import load_model

def analyze_control_samples(model, control_loader, output_dir, device='cpu'):
    """
    Analyze prediction patterns on control samples
    
    Args:
        model: Trained model
        control_loader: DataLoader with control samples
        output_dir: Directory to save results
        device: Device to run inference on
        
    Returns:
        DataFrame with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    all_predictions = []
    all_attentions = []
    all_marker_values = []
    all_coverage = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(control_loader, desc="Analyzing controls"):
            if len(batch) == 4:
                marker_values, coverage, _, sample_ids = batch
            else:
                marker_values, coverage, _ = batch
                sample_ids = [f"sample_{i}" for i in range(len(marker_values))]
            
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            
            # Get predictions and attention weights
            mu, phi, _, attention_weights = model(marker_values, coverage)
            
            all_predictions.append(mu.cpu().numpy())
            all_attentions.append(attention_weights.cpu().numpy())
            all_marker_values.append(marker_values.cpu().numpy())
            all_coverage.append(coverage.cpu().numpy())
            all_sample_ids.extend(sample_ids)
    
    # Concatenate results
    all_predictions = np.concatenate(all_predictions).flatten()
    all_attentions = np.concatenate(all_attentions)
    all_marker_values = np.concatenate(all_marker_values)
    all_coverage = np.concatenate(all_coverage)
    
    # Sort by prediction value (highest first)
    sorted_indices = np.argsort(-all_predictions)
    
    # Create analysis results
    results = {
        'sample_id': [],
        'prediction': [],
        'top_5_markers': [],
        'top_5_values': [],
        'top_5_coverage': [],
        'top_5_attention': []
    }
    
    # Analyze each sample
    for idx in sorted_indices:
        sample_id = all_sample_ids[idx]
        prediction = float(all_predictions[idx])
        attention = all_attentions[idx]
        marker_values = all_marker_values[idx]
        coverage_values = all_coverage[idx]
        
        # Find top 5 markers by attention
        top_marker_indices = np.argsort(-attention)[:5]
        
        # Store results
        results['sample_id'].append(sample_id)
        results['prediction'].append(prediction)
        results['top_5_markers'].append(top_marker_indices.tolist())
        results['top_5_values'].append(marker_values[top_marker_indices].tolist())
        results['top_5_coverage'].append(coverage_values[top_marker_indices].tolist())
        results['top_5_attention'].append(attention[top_marker_indices].tolist())
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'control_analysis.csv')
    df.to_csv(output_file, index=False)
    
    # Generate visualizations
    plot_control_analysis(df, output_dir)
    compare_high_low_predictions(df, output_dir)
    analyze_marker_frequencies(df, output_dir)
    
    # Print summary
    print(f"Control analysis complete. Results saved to {output_dir}")
    
    # Summary statistics
    print(f"Number of control samples: {len(df)}")
    print(f"Average prediction: {df['prediction'].mean():.6f}")
    print(f"Median prediction: {df['prediction'].median():.6f}")
    print(f"Max prediction: {df['prediction'].max():.6f}")
    print(f"Min prediction: {df['prediction'].min():.6f}")
    
    # Print top and bottom 5 predictions
    print("\nTop 5 predictions:")
    for _, row in df.head(5).iterrows():
        print(f"Sample {row['sample_id']}: {row['prediction']:.6f}")
        print(f"  Top markers: {row['top_5_markers']}")
        print(f"  Values: {row['top_5_values']}")
        print(f"  Coverage: {row['top_5_coverage']}")
    
    print("\nBottom 5 predictions:")
    for _, row in df.tail(5).iterrows():
        print(f"Sample {row['sample_id']}: {row['prediction']:.6f}")
    
    return df

def plot_control_analysis(df, output_dir):
    """
    Create visualizations for control analysis using Plotly
    
    Args:
        df: DataFrame with control analysis results
        output_dir: Directory to save visualizations
    """
    # Plot 1: Prediction distribution
    fig1 = px.histogram(
        df, x='prediction', 
        marginal='box', 
        title='Distribution of Predictions in Control Samples',
        labels={'prediction': 'Prediction Value'},
        opacity=0.7,
        color_discrete_sequence=['royalblue']
    )
    fig1.update_layout(
        xaxis_title='Prediction Value',
        yaxis_title='Count',
        template='plotly_white'
    )
    fig1.write_html(os.path.join(output_dir, 'prediction_distribution.html'))
    
    # Plot 2: Top 10 samples by prediction
    top_df = df.nlargest(10, 'prediction')
    fig2 = px.bar(
        top_df, 
        x='sample_id', 
        y='prediction',
        title='Top 10 Control Samples by Prediction',
        labels={'prediction': 'Prediction Value', 'sample_id': 'Sample ID'},
        color='prediction',
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(
        xaxis_title='Sample ID',
        yaxis_title='Prediction Value',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    fig2.write_html(os.path.join(output_dir, 'top_samples.html'))
    
    # Plot 3: Attention vs Value for top samples
    fig3 = make_subplots(rows=2, cols=1, 
                         subplot_titles=('Marker Values vs Attention (Top 3 Samples)',
                                         'Coverage vs Attention (Top 3 Samples)'))
    
    # Get top 3 samples
    top3_samples = df.nlargest(3, 'prediction')
    colors = ['red', 'green', 'blue']
    
    for i, (_, row) in enumerate(top3_samples.iterrows()):
        # Extract data
        markers = row['top_5_markers']
        values = row['top_5_values']
        coverage = row['top_5_coverage']
        attention = row['top_5_attention']
        sample_id = row['sample_id']
        
        # Values vs Attention
        fig3.add_trace(
            go.Scatter(
                x=values, 
                y=attention, 
                mode='markers+text',
                marker=dict(size=12, color=colors[i]),
                text=[f"M{m}" for m in markers],
                name=f"{sample_id} (pred={row['prediction']:.4f})"
            ),
            row=1, col=1
        )
        
        # Coverage vs Attention
        fig3.add_trace(
            go.Scatter(
                x=coverage, 
                y=attention, 
                mode='markers+text',
                marker=dict(size=12, color=colors[i]),
                text=[f"M{m}" for m in markers],
                name=f"{sample_id} (pred={row['prediction']:.4f})"
            ),
            row=2, col=1
        )
    
    # Update layout
    fig3.update_layout(
        height=800,
        width=800,
        template='plotly_white',
        title='Marker Analysis for Top Samples'
    )
    fig3.update_xaxes(title_text='Marker Value', row=1, col=1)
    fig3.update_yaxes(title_text='Attention Weight', row=1, col=1)
    fig3.update_xaxes(title_text='Coverage', row=2, col=1)
    fig3.update_yaxes(title_text='Attention Weight', row=2, col=1)
    
    fig3.write_html(os.path.join(output_dir, 'marker_analysis.html'))

def compare_high_low_predictions(df, output_dir):
    """
    Compare high-prediction vs low-prediction controls
    
    Args:
        df: DataFrame with control analysis results
        output_dir: Directory to save visualizations
    """
    # Split controls into high and low predictions
    median_pred = df['prediction'].median()
    high_pred = df[df['prediction'] > median_pred].copy()
    low_pred = df[df['prediction'] <= median_pred].copy()
    
    high_pred['group'] = 'High Prediction'
    low_pred['group'] = 'Low Prediction'
    
    # Combine for plotting
    compare_df = pd.concat([high_pred, low_pred])
    
    # Collect marker data
    marker_data = []
    
    for _, row in compare_df.iterrows():
        for i in range(5):  # Top 5 markers
            marker_data.append({
                'sample_id': row['sample_id'],
                'prediction': row['prediction'],
                'group': row['group'],
                'marker_index': row['top_5_markers'][i],
                'marker_value': row['top_5_values'][i],
                'marker_coverage': row['top_5_coverage'][i],
                'marker_attention': row['top_5_attention'][i]
            })
    
    marker_df = pd.DataFrame(marker_data)
    
    # Plot 1: Marker value distribution by group
    fig1 = px.box(
        marker_df, 
        x='group', 
        y='marker_value',
        color='group',
        title='Distribution of Marker Values in High vs Low Prediction Controls',
        points='all'
    )
    fig1.update_layout(
        xaxis_title='Prediction Group',
        yaxis_title='Marker Value',
        template='plotly_white'
    )
    fig1.write_html(os.path.join(output_dir, 'value_comparison.html'))
    
    # Plot 2: Attention distribution by group
    fig2 = px.box(
        marker_df, 
        x='group', 
        y='marker_attention',
        color='group',
        title='Distribution of Attention Weights in High vs Low Prediction Controls',
        points='all'
    )
    fig2.update_layout(
        xaxis_title='Prediction Group',
        yaxis_title='Attention Weight',
        template='plotly_white'
    )
    fig2.write_html(os.path.join(output_dir, 'attention_comparison.html'))
    
    # Plot 3: Coverage distribution by group
    fig3 = px.box(
        marker_df, 
        x='group', 
        y='marker_coverage',
        color='group',
        title='Distribution of Coverage in High vs Low Prediction Controls',
        points='all'
    )
    fig3.update_layout(
        xaxis_title='Prediction Group',
        yaxis_title='Coverage',
        template='plotly_white'
    )
    fig3.write_html(os.path.join(output_dir, 'coverage_comparison.html'))

def analyze_marker_frequencies(df, output_dir):
    """
    Analyze which markers appear most frequently in control samples
    
    Args:
        df: DataFrame with control analysis results
        output_dir: Directory to save visualizations
    """
    # Identify common top markers in high-prediction controls
    high_pred_df = df[df['prediction'] > df['prediction'].median()]
    
    # Collect all markers from high prediction samples
    high_pred_markers = []
    for markers in high_pred_df['top_5_markers']:
        high_pred_markers.extend(markers)
    
    # Count frequency
    marker_counts = Counter(high_pred_markers)
    
    # Create DataFrame for plotting
    freq_df = pd.DataFrame({
        'marker_index': list(marker_counts.keys()),
        'frequency': list(marker_counts.values())
    })
    freq_df = freq_df.sort_values('frequency', ascending=False).head(20)
    
    # Plot marker frequency
    fig = px.bar(
        freq_df, 
        x='marker_index', 
        y='frequency',
        title='Most Common Markers in High-Prediction Control Samples',
        labels={'marker_index': 'Marker Index', 'frequency': 'Frequency'},
        color='frequency',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_title='Marker Index',
        yaxis_title='Frequency',
        template='plotly_white',
        xaxis_tickangle=0
    )
    fig.write_html(os.path.join(output_dir, 'marker_frequency.html'))
    
    # Create more detailed analysis of top 10 markers
    top_markers = freq_df.head(10)['marker_index'].tolist()
    
    # Collect data for these markers across all samples
    top_marker_data = []
    
    for _, row in df.iterrows():
        for marker in top_markers:
            if marker in row['top_5_markers']:
                idx = row['top_5_markers'].index(marker)
                top_marker_data.append({
                    'sample_id': row['sample_id'],
                    'prediction': row['prediction'],
                    'marker_index': marker,
                    'marker_value': row['top_5_values'][idx],
                    'marker_coverage': row['top_5_coverage'][idx],
                    'marker_attention': row['top_5_attention'][idx]
                })
    
    top_marker_df = pd.DataFrame(top_marker_data)
    
    # Plot relationship between marker value and prediction
    fig2 = px.scatter(
        top_marker_df, 
        x='marker_value', 
        y='prediction',
        color='marker_index',
        facet_col='marker_index',
        facet_col_wrap=5,
        title='Relationship Between Marker Value and Prediction for Top Markers',
        trendline='ols'
    )
    fig2.update_layout(
        height=800,
        template='plotly_white'
    )
    fig2.write_html(os.path.join(output_dir, 'marker_prediction_relationship.html'))
    
    # Save data for further analysis
    top_marker_df.to_csv(os.path.join(output_dir, 'top_markers_analysis.csv'), index=False)

def analyse(model_path, control_dir, atlas_path, out_dir):
    model,_ = load_model(model_path)
    control_val_marker_values, control_val_coverage, _ = load_control_data(control_dir, atlas_path, "OAC")
    control_dataset = cfDNAMethylationDataset(
        control_val_marker_values.numpy(),
        control_val_coverage.numpy(),
        np.zeros(len(control_val_marker_values)),
        np.ones(len(control_val_marker_values), dtype=bool)
    )
    control_loader = DataLoader(control_dataset, batch_size=32, shuffle=False)
    analyze_control_samples(model, control_loader, out_dir)

analyse(
    model_path="/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/CpGenie_OAC/",
    atlas_path="/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed",
    control_dir="/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/",
    out_dir='/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/CpGenie_OAC/control_analysis'
)


# Example usage:
# df = analyze_control_samples(model, control_loader, '/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/CpGenie_OAC/control_analysis')
