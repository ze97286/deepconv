import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def analyze_detection_limits(atlas_df, cell_types, coverage_levels, confidence_level=0.95):
    """
    Analyze detection limits for cell types based on marker pattern frequencies.
    
    Parameters:
        atlas_df: DataFrame with marker information
            Should include a column 'target' and columns for each cell type
        cell_types: List of all cell types to analyze (column names in atlas_df)
        coverage_levels: List of coverage values to analyze
        confidence_level: Statistical confidence for detection (default: 0.95)
    
    Returns:
        DataFrame with detection limits by marker, cell type, and coverage
    """
    z_score = stats.norm.ppf(confidence_level)
    results = []
    
    # First, filter the atlas to only include the specified cell types
    cell_type_columns = [col for col in atlas_df.columns if col in cell_types]
    
    for target_cell_type in cell_types:
        # For each marker assigned to this cell type
        markers = atlas_df[atlas_df['target'] == target_cell_type]
        
        # Background cell types are all other cell types
        background_cell_types = [ct for ct in cell_type_columns if ct != target_cell_type]
        
        for _, marker in markers.iterrows():
            try:
                # Get pattern frequency in target cell type
                target_freq = float(marker[target_cell_type])
                
                # Get pattern frequencies for all background cell types
                background_freqs = []
                valid_bg_types = []
                
                for bg in background_cell_types:
                    try:
                        bg_freq = float(marker[bg])
                        background_freqs.append(bg_freq)
                        valid_bg_types.append(bg)
                    except (ValueError, TypeError):
                        continue
                
                if not background_freqs:
                    continue  # Skip if no valid background frequencies
                
                # Find the most similar background cell type (hardest to distinguish)
                closest_idx = np.argmin([abs(x - target_freq) for x in background_freqs])
                closest_bg_freq = background_freqs[closest_idx]
                closest_bg_type = valid_bg_types[closest_idx]
                
                # Calculate pattern difference
                pattern_diff = abs(target_freq - closest_bg_freq)
                
                # Skip markers with negligible separation
                if pattern_diff < 0.01:
                    continue
                    
                for coverage in coverage_levels:
                    # Sampling error calculations
                    target_sampling_error = np.sqrt((target_freq * (1 - target_freq)) / coverage)
                    bg_sampling_error = np.sqrt((closest_bg_freq * (1 - closest_bg_freq)) / coverage)
                    
                    # Combined error
                    combined_error = np.sqrt(target_sampling_error**2 + bg_sampling_error**2)
                    
                    # Minimum detectable concentration
                    min_conc = (z_score * combined_error) / pattern_diff
                    min_conc = min(min_conc, 1.0)  # Cap at 100%
                    
                    results.append({
                        'marker_id': marker.name if hasattr(marker, 'name') else marker.get('name', 'Unknown'),
                        'target_cell_type': target_cell_type,
                        'closest_background': closest_bg_type,
                        'coverage': coverage,
                        'target_frequency': target_freq,
                        'background_frequency': closest_bg_freq,
                        'pattern_difference': pattern_diff,
                        'min_detectable_concentration': min_conc
                    })
            except Exception as e:
                print(f"Error processing marker: {e}")
                continue
    
    return pd.DataFrame(results)


def visualize_detection_limits_plotly(results_df):
    figures = {}
    # 1. Plot detection limits vs coverage for each cell type
    fig_by_celltype = go.Figure()
    for cell_type, group in results_df.groupby('target_cell_type'):
        # Calculate median detection limit across markers
        agg_data = group.groupby('coverage')['min_detectable_concentration'].median().reset_index()
        fig_by_celltype.add_trace(go.Scatter(
            x=agg_data['coverage'],
            y=agg_data['min_detectable_concentration'],
            mode='lines+markers',
            name=cell_type,
            hovertemplate='Coverage: %{x}<br>Min Concentration: %{y:.4f}'
        ))
    fig_by_celltype.update_layout(
        title='Detection Limits by Cell Type and Coverage',
        xaxis=dict(
            title='Coverage (reads)',
            type='log',
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Minimum Detectable Concentration',
            type='log',
            gridcolor='lightgray'
        ),
        legend_title='Cell Type',
        hovermode='closest',
        template='plotly_white'
    )
    figures['by_cell_type'] = fig_by_celltype
    # 2. Heatmaps of detection limits by marker and coverage for each cell type
    for cell_type in results_df['target_cell_type'].unique():
        cell_data = results_df[results_df['target_cell_type'] == cell_type]
        # Get top 20 markers with lowest detection limits at lowest coverage
        min_cov = min(results_df['coverage'])
        top_markers = cell_data[cell_data['coverage'] == min_cov].nsmallest(
            20, 'min_detectable_concentration')['marker_id'].unique()
        # Filter data for these markers
        plot_data = cell_data[cell_data['marker_id'].isin(top_markers)]
        # Create pivot table
        pivot_data = plot_data.pivot(
            index='marker_id', 
            columns='coverage', 
            values='min_detectable_concentration'
        )
        # Create heatmap
        fig_heatmap = px.imshow(
            pivot_data,
            labels=dict(x="Coverage", y="Marker", color="Min Detectable Concentration"),
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale="YlGnBu_r",
            aspect="auto"
        )
        # Add text annotations
        for i, y in enumerate(pivot_data.index):
            for j, x in enumerate(pivot_data.columns):
                value = pivot_data.iloc[i, j]
                fig_heatmap.add_annotation(
                    x=x,
                    y=y,
                    text=f"{value:.3f}",
                    showarrow=False,
                    font=dict(color="black" if value > 0.5 else "white")
                )
        fig_heatmap.update_layout(
            title=f'Detection Limits by Marker and Coverage: {cell_type}',
            height=600,
            width=800
        )
        figures[f'heatmap_{cell_type}'] = fig_heatmap
    # 3. Create interactive coverage requirement table
    detection_thresholds = [0.1, 0.05, 0.01, 0.005, 0.001]
    summary_data = []
    for cell_type, group in results_df.groupby('target_cell_type'):
        for threshold in detection_thresholds:
            # For each coverage, get median detection limit across markers
            cov_limits = {}
            for coverage, cov_group in group.groupby('coverage'):
                cov_limits[coverage] = cov_group['min_detectable_concentration'].median()
            # Find minimum coverage that meets threshold
            min_coverage = None
            for coverage, limit in sorted(cov_limits.items()):
                if limit <= threshold:
                    min_coverage = coverage
                    break
            summary_data.append({
                'Cell Type': cell_type,
                'Detection Threshold': threshold,
                'Minimum Coverage Required': min_coverage if min_coverage else "Not achievable"
            })
    summary_df = pd.DataFrame(summary_data)
    # Create table visualization
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_df.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[summary_df[col] for col in summary_df.columns],
            fill_color='lavender',
            align='left'
        )
    )])
    fig_table.update_layout(
        title='Coverage Requirements for Detection Thresholds'
    )
    figures['coverage_requirements'] = fig_table
    # 4. Box plot of detection limits by cell type (at lowest coverage)
    min_cov = min(results_df['coverage'])
    low_cov_data = results_df[results_df['coverage'] == min_cov]
    fig_boxplot = px.box(
        low_cov_data, 
        x='target_cell_type', 
        y='min_detectable_concentration',
        title=f'Distribution of Detection Limits by Cell Type (Coverage = {min_cov})',
        labels={
            'target_cell_type': 'Cell Type',
            'min_detectable_concentration': 'Minimum Detectable Concentration'
        }
    )
    fig_boxplot.update_layout(
        yaxis=dict(type='log'),
        template='plotly_white'
    )
    figures['boxplot'] = fig_boxplot
    return figures, summary_df


def analyse_atlas(atlas_path, out_dir):
    df = pd.read_csv(atlas_path, sep="\t")
    cell_types = df.columns[8:]
    covs = [5,10,15,20,25,30,35,40]
    results = analyze_detection_limits(df, cell_types,covs)
    figs, summary = visualize_detection_limits_plotly(results)
    for k, fig in figs.items():
        fig.write_html(f"{out_dir}/{k}.html")
        summary.to_csv(f"{out_dir}/results_summary.csv",index=False)