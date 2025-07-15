#!/usr/bin/env python3
"""
Marker Overlap Analysis Script

Analyzes spatial overlaps between methylation markers to understand:
1. How many markers overlap with each other
2. Distribution of overlap sizes
3. Impact on model architecture (ScalableMarkerAggregator)
4. Potential redundancy in the 10k marker set

Usage:
    python analyze_marker_overlaps.py --atlas_path path/to/atlas.tsv --output_dir ./overlap_analysis
"""

import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from intervaltree import Interval, IntervalTree
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze marker overlaps in methylation atlas')
    
    parser.add_argument('--atlas_path', type=str, required=True, 
                       help='Path to atlas TSV file with marker regions')
    parser.add_argument('--output_dir', type=str, default='./overlap_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size for analysis (default: use all markers)')
    parser.add_argument('--min_overlap_bp', type=int, default=1,
                       help='Minimum overlap size in base pairs to consider')
    
    return parser.parse_args()

def load_atlas_markers(atlas_path, sample_size=None):
    """
    Load marker regions from atlas file
    
    Expected columns: chr, startCpG, endCpG, name, direction
    """
    print(f"Loading atlas markers from {atlas_path}...")
    
    # Read the atlas file
    try:
        markers_df = pd.read_csv(atlas_path, sep='\t')
        print(f"Loaded {len(markers_df)} markers")
    except Exception as e:
        print(f"Error loading atlas: {e}")
        return None
    
    # Check required columns
    required_cols = ['chr', 'startCpG', 'endCpG', 'name', 'direction']
    missing_cols = [col for col in required_cols if col not in markers_df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {list(markers_df.columns)}")
        return None
    
    # Sample if requested
    if sample_size and sample_size < len(markers_df):
        markers_df = markers_df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} markers for analysis")
    
    # Sort by chromosome and position
    markers_df = markers_df.sort_values(['chr', 'startCpG']).reset_index(drop=True)
    
    # Add marker length
    markers_df['length_bp'] = markers_df['endCpG'] - markers_df['startCpG']
    
    print(f"Marker length statistics:")
    print(f"  Mean: {markers_df['length_bp'].mean():.1f} bp")
    print(f"  Median: {markers_df['length_bp'].median():.1f} bp")
    print(f"  Min: {markers_df['length_bp'].min()} bp")
    print(f"  Max: {markers_df['length_bp'].max()} bp")
    
    return markers_df

def build_interval_trees(markers_df):
    """Build interval trees for efficient overlap detection"""
    print("Building interval trees for overlap detection...")
    
    trees = {}
    for chrom in markers_df['chr'].unique():
        chrom_markers = markers_df[markers_df['chr'] == chrom]
        tree = IntervalTree()
        
        for idx, row in chrom_markers.iterrows():
            # Store original dataframe index in the interval
            tree[row['startCpG']:row['endCpG']] = idx
        
        trees[chrom] = tree
        print(f"  {chrom}: {len(chrom_markers)} markers")
    
    return trees

def analyze_overlaps(markers_df, trees, min_overlap_bp=1):
    """
    Analyze overlaps between markers using interval trees
    """
    print(f"Analyzing overlaps (min overlap: {min_overlap_bp} bp)...")
    
    overlap_data = []
    overlap_matrix = defaultdict(set)
    
    for idx, marker in tqdm(markers_df.iterrows(), total=len(markers_df), desc="Finding overlaps"):
        chrom = marker['chr']
        start = marker['startCpG']
        end = marker['endCpG']
        
        if chrom not in trees:
            continue
        
        # Find overlapping intervals
        overlaps = trees[chrom][start:end]
        
        for interval in overlaps:
            other_idx = interval.data
            
            # Skip self-overlap
            if other_idx == idx:
                continue
            
            other_marker = markers_df.iloc[other_idx]
            
            # Calculate overlap size
            overlap_start = max(start, other_marker['startCpG'])
            overlap_end = min(end, other_marker['endCpG'])
            overlap_size = overlap_end - overlap_start
            
            if overlap_size >= min_overlap_bp:
                # Calculate overlap fractions
                overlap_frac_1 = overlap_size / (end - start)
                overlap_frac_2 = overlap_size / (other_marker['endCpG'] - other_marker['startCpG'])
                
                overlap_data.append({
                    'marker1_idx': idx,
                    'marker2_idx': other_idx,
                    'marker1_name': marker['name'],
                    'marker2_name': other_marker['name'],
                    'chr': chrom,
                    'overlap_size_bp': overlap_size,
                    'overlap_frac_1': overlap_frac_1,
                    'overlap_frac_2': overlap_frac_2,
                    'max_overlap_frac': max(overlap_frac_1, overlap_frac_2),
                    'min_overlap_frac': min(overlap_frac_1, overlap_frac_2),
                    'marker1_length': end - start,
                    'marker2_length': other_marker['endCpG'] - other_marker['startCpG']
                })
                
                # Track overlap relationships (symmetric)
                overlap_matrix[idx].add(other_idx)
                overlap_matrix[other_idx].add(idx)
    
    overlap_df = pd.DataFrame(overlap_data)
    print(f"Found {len(overlap_df)} overlap pairs")
    
    return overlap_df, overlap_matrix

def calculate_overlap_statistics(markers_df, overlap_df, overlap_matrix):
    """Calculate comprehensive overlap statistics"""
    print("Calculating overlap statistics...")
    
    stats = {}
    
    # Basic counts
    total_markers = len(markers_df)
    markers_with_overlaps = len([idx for idx in overlap_matrix if len(overlap_matrix[idx]) > 0])
    
    stats['total_markers'] = total_markers
    stats['markers_with_overlaps'] = markers_with_overlaps
    stats['markers_without_overlaps'] = total_markers - markers_with_overlaps
    stats['overlap_percentage'] = (markers_with_overlaps / total_markers) * 100
    
    # Overlap degree distribution
    overlap_degrees = [len(overlap_matrix[idx]) for idx in range(total_markers)]
    stats['overlap_degrees'] = overlap_degrees
    stats['mean_overlaps_per_marker'] = np.mean(overlap_degrees)
    stats['median_overlaps_per_marker'] = np.median(overlap_degrees)
    stats['max_overlaps_per_marker'] = np.max(overlap_degrees)
    
    if len(overlap_df) > 0:
        # Overlap size statistics
        stats['overlap_sizes'] = {
            'mean_bp': overlap_df['overlap_size_bp'].mean(),
            'median_bp': overlap_df['overlap_size_bp'].median(),
            'min_bp': overlap_df['overlap_size_bp'].min(),
            'max_bp': overlap_df['overlap_size_bp'].max(),
            'std_bp': overlap_df['overlap_size_bp'].std()
        }
        
        # Overlap fraction statistics
        stats['overlap_fractions'] = {
            'mean_max_frac': overlap_df['max_overlap_frac'].mean(),
            'median_max_frac': overlap_df['max_overlap_frac'].median(),
            'mean_min_frac': overlap_df['min_overlap_frac'].mean(),
            'median_min_frac': overlap_df['min_overlap_frac'].median()
        }
        
        # High overlap analysis (>50% overlap)
        high_overlap = overlap_df[overlap_df['max_overlap_frac'] > 0.5]
        stats['high_overlap_pairs'] = len(high_overlap)
        stats['high_overlap_percentage'] = (len(high_overlap) / len(overlap_df)) * 100
    
    # Chromosome-specific statistics
    chrom_stats = {}
    for chrom in markers_df['chr'].unique():
        chrom_markers = markers_df[markers_df['chr'] == chrom]
        chrom_overlaps = overlap_df[overlap_df['chr'] == chrom] if len(overlap_df) > 0 else pd.DataFrame()
        
        chrom_stats[chrom] = {
            'total_markers': len(chrom_markers),
            'overlap_pairs': len(chrom_overlaps),
            'density': len(chrom_overlaps) / len(chrom_markers) if len(chrom_markers) > 0 else 0
        }
    
    stats['chromosome_stats'] = chrom_stats
    
    return stats

def create_visualizations(markers_df, overlap_df, overlap_matrix, stats, output_dir):
    """Create comprehensive visualizations"""
    print("Creating visualizations...")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Overlap degree distribution
    create_overlap_degree_plot(stats, plots_dir)
    
    # 2. Overlap size distribution
    if len(overlap_df) > 0:
        create_overlap_size_plots(overlap_df, plots_dir)
        
        # 3. Overlap fraction distribution
        create_overlap_fraction_plots(overlap_df, plots_dir)
        
        # 4. Chromosome-specific analysis
        create_chromosome_plots(markers_df, overlap_df, stats, plots_dir)
        
        # 5. High overlap analysis
        create_high_overlap_analysis(overlap_df, plots_dir)
        
        # 6. Model implications plot
        create_model_implications_plot(stats, plots_dir)

def create_overlap_degree_plot(stats, plots_dir):
    """Plot distribution of overlap degrees"""
    fig = go.Figure()
    
    # Histogram of overlap degrees
    fig.add_trace(go.Histogram(
        x=stats['overlap_degrees'],
        nbinsx=min(50, max(stats['overlap_degrees']) + 1),
        name='Overlap Degree',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    # Add vertical lines for mean and median
    fig.add_vline(x=stats['mean_overlaps_per_marker'], 
                  line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {stats['mean_overlaps_per_marker']:.1f}")
    fig.add_vline(x=stats['median_overlaps_per_marker'], 
                  line_dash="dash", line_color="green",
                  annotation_text=f"Median: {stats['median_overlaps_per_marker']:.1f}")
    
    fig.update_layout(
        title="Distribution of Marker Overlap Degrees",
        xaxis_title="Number of Overlapping Markers",
        yaxis_title="Count",
        template='plotly_white',
        width=800,
        height=600
    )
    
    # Add text annotation with key stats
    fig.add_annotation(
        x=0.7, y=0.8,
        xref="paper", yref="paper",
        text=f"Total markers: {stats['total_markers']}<br>"
             f"Markers with overlaps: {stats['markers_with_overlaps']}<br>"
             f"Overlap percentage: {stats['overlap_percentage']:.1f}%<br>"
             f"Max overlaps: {stats['max_overlaps_per_marker']}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.write_html(os.path.join(plots_dir, 'overlap_degree_distribution.html'))
    fig.write_image(os.path.join(plots_dir, 'overlap_degree_distribution.png'), scale=2)

def create_overlap_size_plots(overlap_df, plots_dir):
    """Plot overlap size distributions"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Overlap Size (bp)', 'Overlap Size (log scale)')
    )
    
    # Linear scale
    fig.add_trace(
        go.Histogram(
            x=overlap_df['overlap_size_bp'],
            nbinsx=50,
            name='Overlap Size',
            marker_color='lightblue',
            opacity=0.7,
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Log scale
    fig.add_trace(
        go.Histogram(
            x=overlap_df['overlap_size_bp'],
            nbinsx=50,
            name='Overlap Size (log)',
            marker_color='lightcoral',
            opacity=0.7,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update x-axis for log scale
    fig.update_xaxes(type="log", row=1, col=2)
    
    fig.update_layout(
        title="Overlap Size Distributions",
        template='plotly_white',
        width=1000,
        height=500
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Overlap Size (bp)", row=1, col=1)
    fig.update_xaxes(title_text="Overlap Size (bp, log scale)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.write_html(os.path.join(plots_dir, 'overlap_size_distribution.html'))
    fig.write_image(os.path.join(plots_dir, 'overlap_size_distribution.png'), scale=2)

def create_overlap_fraction_plots(overlap_df, plots_dir):
    """Plot overlap fraction distributions"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Max Overlap Fraction', 'Min Overlap Fraction', 
                       'Overlap Fraction Scatter', 'High Overlap Pairs')
    )
    
    # Max overlap fraction histogram
    fig.add_trace(
        go.Histogram(
            x=overlap_df['max_overlap_frac'],
            nbinsx=50,
            name='Max Overlap Fraction',
            marker_color='blue',
            opacity=0.7,
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Min overlap fraction histogram
    fig.add_trace(
        go.Histogram(
            x=overlap_df['min_overlap_frac'],
            nbinsx=50,
            name='Min Overlap Fraction',
            marker_color='green',
            opacity=0.7,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Scatter plot of overlap fractions
    fig.add_trace(
        go.Scatter(
            x=overlap_df['overlap_frac_1'],
            y=overlap_df['overlap_frac_2'],
            mode='markers',
            marker=dict(size=3, opacity=0.6, color='purple'),
            name='Overlap Pairs',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # High overlap pairs (>50%)
    high_overlap = overlap_df[overlap_df['max_overlap_frac'] > 0.5]
    if len(high_overlap) > 0:
        fig.add_trace(
            go.Histogram(
                x=high_overlap['max_overlap_frac'],
                nbinsx=20,
                name='High Overlap',
                marker_color='red',
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Overlap Fraction Analysis",
        template='plotly_white',
        width=1200,
        height=800
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Max Overlap Fraction", row=1, col=1)
    fig.update_xaxes(title_text="Min Overlap Fraction", row=1, col=2)
    fig.update_xaxes(title_text="Marker 1 Overlap Fraction", row=2, col=1)
    fig.update_yaxes(title_text="Marker 2 Overlap Fraction", row=2, col=1)
    fig.update_xaxes(title_text="High Overlap Fraction", row=2, col=2)
    
    fig.write_html(os.path.join(plots_dir, 'overlap_fraction_analysis.html'))
    fig.write_image(os.path.join(plots_dir, 'overlap_fraction_analysis.png'), scale=2)

def create_chromosome_plots(markers_df, overlap_df, stats, plots_dir):
    """Create chromosome-specific analysis plots"""
    chrom_data = []
    for chrom, data in stats['chromosome_stats'].items():
        chrom_data.append({
            'chromosome': chrom,
            'total_markers': data['total_markers'],
            'overlap_pairs': data['overlap_pairs'],
            'density': data['density']
        })
    
    chrom_df = pd.DataFrame(chrom_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Markers per Chromosome', 'Overlap Pairs per Chromosome',
                       'Overlap Density', 'Overlap vs Marker Count')
    )
    
    # Markers per chromosome
    fig.add_trace(
        go.Bar(
            x=chrom_df['chromosome'],
            y=chrom_df['total_markers'],
            name='Markers',
            marker_color='lightblue',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Overlap pairs per chromosome
    fig.add_trace(
        go.Bar(
            x=chrom_df['chromosome'],
            y=chrom_df['overlap_pairs'],
            name='Overlap Pairs',
            marker_color='lightcoral',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Overlap density (pairs per marker)
    fig.add_trace(
        go.Bar(
            x=chrom_df['chromosome'],
            y=chrom_df['density'],
            name='Density',
            marker_color='lightgreen',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Scatter: overlap pairs vs marker count
    fig.add_trace(
        go.Scatter(
            x=chrom_df['total_markers'],
            y=chrom_df['overlap_pairs'],
            mode='markers',
            marker=dict(size=8, color='purple'),
            text=chrom_df['chromosome'],
            textposition="top center",
            name='Chromosomes',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Chromosome-Specific Overlap Analysis",
        template='plotly_white',
        width=1200,
        height=800
    )
    
    # Update axis labels
    fig.update_yaxes(title_text="Marker Count", row=1, col=1)
    fig.update_yaxes(title_text="Overlap Pairs", row=1, col=2)
    fig.update_yaxes(title_text="Overlap Density", row=2, col=1)
    fig.update_xaxes(title_text="Marker Count", row=2, col=2)
    fig.update_yaxes(title_text="Overlap Pairs", row=2, col=2)
    
    fig.write_html(os.path.join(plots_dir, 'chromosome_analysis.html'))
    fig.write_image(os.path.join(plots_dir, 'chromosome_analysis.png'), scale=2)

def create_high_overlap_analysis(overlap_df, plots_dir):
    """Analyze pairs with high overlap (>50%)"""
    high_overlap = overlap_df[overlap_df['max_overlap_frac'] > 0.5]
    
    if len(high_overlap) == 0:
        print("No high overlap pairs found (>50%)")
        return
    
    # Create analysis of high overlap pairs
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('High Overlap Size Distribution', 'High Overlap by Chromosome',
                       'Marker Length vs Overlap', 'High Overlap Fractions')
    )
    
    # High overlap size distribution
    fig.add_trace(
        go.Histogram(
            x=high_overlap['overlap_size_bp'],
            nbinsx=30,
            name='High Overlap Size',
            marker_color='red',
            opacity=0.7,
            showlegend=False
        ),
        row=1, col=1
    )
    
    # High overlap by chromosome
    chrom_counts = high_overlap['chr'].value_counts()
    fig.add_trace(
        go.Bar(
            x=chrom_counts.index,
            y=chrom_counts.values,
            name='High Overlap by Chr',
            marker_color='orange',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Marker length vs overlap size
    fig.add_trace(
        go.Scatter(
            x=high_overlap['marker1_length'],
            y=high_overlap['overlap_size_bp'],
            mode='markers',
            marker=dict(size=4, opacity=0.6, color='blue'),
            name='Length vs Overlap',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # High overlap fraction distribution
    fig.add_trace(
        go.Box(
            y=high_overlap['max_overlap_frac'],
            name='Max Overlap Fraction',
            marker_color='purple',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"High Overlap Analysis (n={len(high_overlap)} pairs >50% overlap)",
        template='plotly_white',
        width=1200,
        height=800
    )
    
    fig.write_html(os.path.join(plots_dir, 'high_overlap_analysis.html'))
    fig.write_image(os.path.join(plots_dir, 'high_overlap_analysis.png'), scale=2)

def create_model_implications_plot(stats, plots_dir):
    """Create visualization showing implications for model architecture"""
    
    # Calculate model-relevant metrics
    overlap_degrees = stats['overlap_degrees']
    
    # Group analysis for ScalableMarkerAggregator
    group_sizes = [64, 128, 256, 512]  # Different possible group sizes
    
    implications_data = []
    for group_size in group_sizes:
        markers_per_group = stats['total_markers'] / group_size
        avg_overlaps_per_group = stats['mean_overlaps_per_marker'] * markers_per_group
        
        implications_data.append({
            'group_size': group_size,
            'markers_per_group': markers_per_group,
            'avg_overlaps_per_group': avg_overlaps_per_group,
            'potential_redundancy': min(avg_overlaps_per_group / markers_per_group, 1.0)
        })
    
    impl_df = pd.DataFrame(implications_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Markers per Group', 'Overlaps per Group',
                       'Potential Redundancy', 'Overlap Degree Distribution')
    )
    
    # Markers per group
    fig.add_trace(
        go.Bar(
            x=impl_df['group_size'],
            y=impl_df['markers_per_group'],
            name='Markers per Group',
            marker_color='lightblue',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Average overlaps per group
    fig.add_trace(
        go.Bar(
            x=impl_df['group_size'],
            y=impl_df['avg_overlaps_per_group'],
            name='Overlaps per Group',
            marker_color='lightcoral',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Potential redundancy
    fig.add_trace(
        go.Scatter(
            x=impl_df['group_size'],
            y=impl_df['potential_redundancy'],
            mode='lines+markers',
            name='Redundancy',
            line=dict(color='red', width=3),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Overlap degree distribution (box plot)
    fig.add_trace(
        go.Box(
            y=overlap_degrees,
            name='Overlap Degrees',
            marker_color='green',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Model Architecture Implications",
        template='plotly_white',
        width=1200,
        height=800
    )
    
    # Add text annotation with recommendations
    recommendation_text = f"""
    Recommendations for ScalableMarkerAggregator:
    
    • Current: 64 groups → {stats['total_markers']/64:.0f} markers/group
    • Overlap rate: {stats['overlap_percentage']:.1f}% of markers have overlaps
    • Avg overlaps: {stats['mean_overlaps_per_marker']:.1f} per marker
    
    • Soft grouping may naturally cluster overlapping markers
    • Cross-group attention can handle spatial relationships
    • No major architecture changes needed
    """
    
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=recommendation_text,
        showarrow=False,
        font=dict(size=10),
        bgcolor="lightyellow",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )
    
    fig.write_html(os.path.join(plots_dir, 'model_implications.html'))
    fig.write_image(os.path.join(plots_dir, 'model_implications.png'), scale=2)

def generate_summary_report(markers_df, overlap_df, stats, output_dir):
    """Generate a comprehensive summary report"""
    print("Generating summary report...")
    
    report_path = os.path.join(output_dir, 'overlap_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Marker Overlap Analysis Report\n\n")
        
        # Basic statistics
        f.write("## Basic Statistics\n\n")
        f.write(f"- **Total markers**: {stats['total_markers']:,}\n")
        f.write(f"- **Markers with overlaps**: {stats['markers_with_overlaps']:,} ({stats['overlap_percentage']:.1f}%)\n")
        f.write(f"- **Markers without overlaps**: {stats['markers_without_overlaps']:,}\n")
        f.write(f"- **Average overlaps per marker**: {stats['mean_overlaps_per_marker']:.2f}\n")
        f.write(f"- **Maximum overlaps for single marker**: {stats['max_overlaps_per_marker']}\n\n")
        
        if len(overlap_df) > 0:
            # Overlap characteristics
            f.write("## Overlap Characteristics\n\n")
            f.write(f"- **Total overlap pairs**: {len(overlap_df):,}\n")
            f.write(f"- **Mean overlap size**: {stats['overlap_sizes']['mean_bp']:.1f} bp\n")
            f.write(f"- **Median overlap size**: {stats['overlap_sizes']['median_bp']:.1f} bp\n")
            f.write(f"- **High overlap pairs** (>50%): {stats['high_overlap_pairs']:,} ({stats['high_overlap_percentage']:.1f}%)\n\n")
            
            # Overlap fractions
            f.write("## Overlap Fractions\n\n")
            f.write(f"- **Mean maximum overlap fraction**: {stats['overlap_fractions']['mean_max_frac']:.3f}\n")
            f.write(f"- **Median maximum overlap fraction**: {stats['overlap_fractions']['median_max_frac']:.3f}\n")
            f.write(f"- **Mean minimum overlap fraction**: {stats['overlap_fractions']['mean_min_frac']:.3f}\n")
            f.write(f"- **Median minimum overlap fraction**: {stats['overlap_fractions']['median_min_frac']:.3f}\n\n")
        
        # Chromosome breakdown
        f.write("## Chromosome Breakdown\n\n")
        f.write("| Chromosome | Markers | Overlap Pairs | Density |\n")
        f.write("|------------|---------|---------------|----------|\n")
        for chrom, data in sorted(stats['chromosome_stats'].items()):
            f.write(f"| {chrom} | {data['total_markers']:,} | {data['overlap_pairs']:,} | {data['density']:.3f} |\n")
        f.write("\n")
        
        # Model implications
        f.write("## Implications for ScalableMarkerAggregator\n\n")
        f.write("### Current Architecture (64 groups):\n")
        f.write(f"- **Markers per group**: {stats['total_markers']/64:.0f}\n")
        f.write(f"- **Expected overlaps per group**: {stats['mean_overlaps_per_marker'] * stats['total_markers']/64:.1f}\n\n")
        
        f.write("### Recommendations:\n\n")
        if stats['overlap_percentage'] < 30:
            f.write("✅ **Low overlap rate** - Current architecture should work well\n\n")
        elif stats['overlap_percentage'] < 60:
            f.write("⚠️ **Moderate overlap rate** - Monitor grouping effectiveness\n\n")
        else:
            f.write("❌ **High overlap rate** - Consider overlap-aware grouping\n\n")
        
        f.write("### Architecture Notes:\n")
        f.write("- Soft grouping in ScalableMarkerAggregator may naturally cluster overlapping markers\n")
        f.write("- Cross-group attention can capture spatial relationships\n")
        f.write("- Marker identity embeddings provide position information\n")
        f.write("- No immediate architecture changes required\n\n")
        
        # Files generated
        f.write("## Generated Files\n\n")
        f.write("- `overlap_analysis.json` - Raw overlap data\n")
        f.write("- `overlap_statistics.json` - Summary statistics\n")
        f.write("- `plots/` - Visualization plots\n")
        f.write("  - `overlap_degree_distribution.html/png`\n")
        f.write("  - `overlap_size_distribution.html/png`\n")
        f.write("  - `overlap_fraction_analysis.html/png`\n")
        f.write("  - `chromosome_analysis.html/png`\n")
        f.write("  - `high_overlap_analysis.html/png`\n")
        f.write("  - `model_implications.html/png`\n")
    
    print(f"Summary report saved to: {report_path}")

def main():
    """Main analysis function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load atlas markers
    markers_df = load_atlas_markers(args.atlas_path, args.sample_size)
    if markers_df is None:
        return
    
    # Build interval trees for efficient overlap detection
    trees = build_interval_trees(markers_df)
    
    # Analyze overlaps
    overlap_df, overlap_matrix = analyze_overlaps(markers_df, trees, args.min_overlap_bp)
    
    # Calculate statistics
    stats = calculate_overlap_statistics(markers_df, overlap_df, overlap_matrix)
    
    # Create visualizations
    create_visualizations(markers_df, overlap_df, overlap_matrix, stats, args.output_dir)
    
    # Save raw data
    if len(overlap_df) > 0:
        overlap_df.to_csv(os.path.join(args.output_dir, 'overlap_analysis.csv'), index=False)
    
    # Save statistics
    with open(os.path.join(args.output_dir, 'overlap_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Generate summary report
    generate_summary_report(markers_df, overlap_df, stats, args.output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}")
    print(f"📊 Key findings:")
    print(f"   - {stats['overlap_percentage']:.1f}% of markers have overlaps")
    print(f"   - Average {stats['mean_overlaps_per_marker']:.1f} overlaps per marker")
    print(f"   - Maximum {stats['max_overlaps_per_marker']} overlaps for single marker")
    if len(overlap_df) > 0:
        print(f"   - {stats['high_overlap_pairs']} pairs with >50% overlap")

if __name__ == "__main__":
    main()