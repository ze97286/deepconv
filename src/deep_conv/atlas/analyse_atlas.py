import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

t_cell_name = "T-cells"
t_cell_name = "CD4-T-cells"

def analyze_atlas(df):
    # Get cell type columns (after 'direction')
    cell_types = df.columns[df.columns.get_loc('direction')+1:].tolist()
    # 1. Basic Statistics
    stats_dict = {
        'markers_per_cell_type': {},
        'non_zero_markers': {},
        'mean_values': {},
        'variance': {}
    }
    for cell_type in cell_types:
        stats_dict['markers_per_cell_type'][cell_type] = len(df[df['target'] == cell_type])
        stats_dict['non_zero_markers'][cell_type] = (df[cell_type] > 0).sum()
        stats_dict['mean_values'][cell_type] = df[cell_type].mean()
        stats_dict['variance'][cell_type] = df[cell_type].var()
    # 2. CD4 Specific Analysis
    cd4_markers = df[df['target'] == t_cell_name]
    cd4_stats = {
        'total_markers': len(cd4_markers),
        'zero_value_markers': (cd4_markers[t_cell_name] == 0).sum(),
        'mean_value': cd4_markers[t_cell_name].mean(),
        'variance': cd4_markers[t_cell_name].var(),
        'discriminatory_power': []
    }
    # Calculate discriminatory power for different concentrations
    concentrations = [0.1, 0.01, 0.005, 0.001]  # 10%, 1%, 0.5%, 0.1%
    for conc in concentrations:
        signal = cd4_markers[t_cell_name] * conc
        noise = cd4_markers[cell_types].drop(t_cell_name, axis=1).mean(axis=1) * (1-conc)
        snr = signal / (noise + 1e-10)
        cd4_stats['discriminatory_power'].append({
            'concentration': conc,
            'mean_snr': snr.mean(),
            'median_snr': snr.median(),
            'detectable_markers': (snr > 1).sum()
        })
    # 3. Correlation Analysis
    corr_matrix = df[cell_types].corr()
    return {
        'general_stats': stats_dict,
        t_cell_name+'_analysis': cd4_stats,
        'correlation': corr_matrix
    }

def simulate_cd4_detection(df, concentrations):
    cd4_markers = df[df['target'] == t_cell_name]
    cell_types = df.columns[df.columns.get_loc('direction')+1:].tolist()
    other_cells = [ct for ct in cell_types if ct != t_cell_name]
    results = []
    for conc in concentrations:
        other_conc = (1 - conc) / len(other_cells)
        mixture = cd4_markers[t_cell_name] * conc
        for cell in other_cells:
            mixture += cd4_markers[cell] * other_conc
        signal = cd4_markers[t_cell_name] * conc
        background = mixture - signal
        snr = signal / (background + 1e-10)
        results.append({
            'concentration': conc,
            'mean_signal': signal.mean(),
            'mean_background': background.mean(),
            'snr': snr.mean(),
            'detectable_markers': (snr > 1).sum()
        })
    return pd.DataFrame(results)

def create_visualizations(df, output_file='atlas_analysis.html'):
    # Get cell type columns
    cell_types = df.columns[df.columns.get_loc('direction')+1:].tolist()
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Markers per Cell Type',
            t_cell_name+' Marker Value Distribution',
            t_cell_name+' Detection Analysis',
            'Cell Type Correlations',
            t_cell_name+' vs Other Cell Types',
            'Signal-to-Noise Ratio by Concentration'
        )
    )
    # 1. Markers per Cell Type
    markers_per_type = pd.Series({ct: len(df[df['target'] == ct]) for ct in cell_types})
    fig.add_trace(
        go.Bar(x=markers_per_type.index, y=markers_per_type.values, name='Markers'),
        row=1, col=1
    )
    # 2. CD4 Value Distribution
    cd4_markers = df[df['target'] == t_cell_name]
    fig.add_trace(
        go.Histogram(x=cd4_markers[t_cell_name], name=t_cell_name+' Values'),
        row=1, col=2
    )
    # 3. Detection Analysis
    concentrations = [0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.001, 0.0005, 0.0001]
    detection_results = simulate_cd4_detection(df, concentrations)
    fig.add_trace(
        go.Scatter(
            x=detection_results['concentration'],
            y=detection_results['snr'],
            name='Signal-to-Noise',
            mode='lines+markers'
        ),
        row=2, col=1
    )
    # 4. Correlation Heatmap
    corr_matrix = df[cell_types].corr()
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=cell_types,
            y=cell_types,
            colorscale='RdBu',
            name='Correlations'
        ),
        row=2, col=2
    )
    # 5. CD4 vs Other Types Scatter
    other_cell = cell_types[0] if cell_types[0] != t_cell_name else cell_types[1]
    fig.add_trace(
        go.Scatter(
            x=df[t_cell_name],
            y=df[other_cell],
            mode='markers',
            name=f'{t_cell_name} vs {other_cell}',
            opacity=0.6
        ),
        row=3, col=1
    )
    # 6. Detectable Markers by Concentration
    fig.add_trace(
        go.Scatter(
            x=detection_results['concentration'],
            y=detection_results['detectable_markers'],
            name='Detectable Markers',
            mode='lines+markers'
        ),
        row=3, col=2
    )
    # Update layout
    fig.update_layout(
        height=1200,
        width=1600,
        showlegend=True,
        title_text="Atlas Analysis Dashboard"
    )
    # Update x-axes to log scale for concentration plots
    fig.update_xaxes(type="log", row=2, col=1)
    fig.update_xaxes(type="log", row=3, col=2)
    # Save to file
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")

def run_analysis(input_file, output_file='atlas_analysis.html'):
    # Read the data
    df = pd.read_csv(input_file, sep='\t')
    
    # Create visualizations
    create_visualizations(df, output_file)
    
    # Return basic statistics
    analysis_results = analyze_atlas(df)
    return analysis_results


def analyze_cd4_markers(df, output_file='cd4_detailed_analysis.html'):
    # Get blood/immune cell types
    blood_cells = ['B-cells', 'CD4-T-cells', 'CD8-T-cells', 'NK-cells', 'CD34-erythroblasts', 'CD34-megakaryocytes', 'Monocytes', 'Neutrophils', 'Eosinophils']
    # Get CD4 markers
    cd4_markers = df[df['target'] == 'CD4-T-cells'].copy()
    # Calculate marker quality metrics
    for idx, row in cd4_markers.iterrows():
        # Signal strength: CD4 value
        cd4_markers.loc[idx, 'signal_strength'] = row['CD4-T-cells']
        # Background noise: mean value in other blood cells
        other_blood = [c for c in blood_cells if c != 'CD4-T-cells']
        cd4_markers.loc[idx, 'blood_background'] = row[other_blood].mean()
        # Signal-to-background ratio
        cd4_markers.loc[idx, 'baseline_snr'] = (row['CD4-T-cells'] + 1e-10) / (row[other_blood].mean() + 1e-10)
        # Variance in background
        cd4_markers.loc[idx, 'background_var'] = row[other_blood].var()
    # Analyze marker counts performance
    marker_counts = np.linspace(100, len(cd4_markers), 20, dtype=int)  # 20 evenly spaced points
    detection_by_count = []
    # Sort markers by baseline SNR
    sorted_markers = cd4_markers.sort_values('baseline_snr', ascending=False)
    for count in marker_counts:
        top_markers = sorted_markers.head(count)
        for conc in [0.01, 0.005, 0.001]:
            # Calculate mixture for all markers at once
            cd4_signal = top_markers['CD4-T-cells'] * conc
            other_blood = [c for c in blood_cells if c != 'CD4-T-cells']
            other_conc = (1 - conc) / len(other_blood)
            background = sum(top_markers[cell] * other_conc for cell in other_blood)
            # Calculate SNR
            snr = cd4_signal / (background + 1e-10)
            mean_snr = snr.mean()
            detectable = (snr > 1).sum()
            detection_by_count.append({
                'marker_count': count,
                'concentration': conc,
                'mean_snr': mean_snr,
                'detectable_markers': detectable,
                'concentration_label': f'{conc*100}%'
            })
    # Simulate detection at low concentrations for all markers
    concentrations = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    detection_results = []
    for conc in concentrations:
        cd4_signal = cd4_markers['CD4-T-cells'] * conc
        other_blood = [c for c in blood_cells if c != 'CD4-T-cells']
        other_conc = (1 - conc) / len(other_blood)
        background = sum(cd4_markers[cell] * other_conc for cell in other_blood)
        snr = cd4_signal / (background + 1e-10)
        markers_above_threshold = (snr > 1).sum()
        detection_results.append({
            'concentration': conc,
            'detectable_markers': markers_above_threshold,
            'mean_snr': snr.mean(),
            'median_snr': snr.median(),
            'top_10_snr': np.mean(sorted(snr, reverse=True)[:10])
        })
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Marker Quality Distribution',
            'Detection Performance by Marker Count',
            'Top Markers Performance',
            'Background Noise Distribution'
        )
    )
    # 1. Marker Quality Distribution
    fig.add_trace(
        go.Scatter(
            x=cd4_markers['signal_strength'],
            y=cd4_markers['blood_background'],
            mode='markers',
            name='Markers',
            marker=dict(
                color=cd4_markers['baseline_snr'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Baseline SNR')
            )
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text='CD4 Signal', row=1, col=1)
    fig.update_yaxes(title_text='Blood Background', row=1, col=1)
    # 2. Detection Performance by Marker Count
    df_by_count = pd.DataFrame(detection_by_count)
    for conc in df_by_count['concentration'].unique():
        data = df_by_count[df_by_count['concentration'] == conc]
        fig.add_trace(
            go.Scatter(
                x=data['marker_count'],
                y=data['mean_snr'],
                name=f'{conc*100}%',
                mode='lines+markers',
                hovertemplate="Count: %{x}<br>SNR: %{y:.3f}<br>Detectable: %{text}",
                text=data['detectable_markers']
            ),
            row=1, col=2
        )
    fig.update_xaxes(
        title_text='Number of Top Markers', 
        type='log',
        dtick=0.30103,  # log10(2), gives nice round numbers
        row=1, col=2
    )
    fig.update_yaxes(title_text='Mean SNR', row=1, col=2)
    # 3. Top Markers Performance
    df_detection = pd.DataFrame(detection_results)
    fig.add_trace(
        go.Scatter(
            x=df_detection['concentration'],
            y=df_detection['top_10_snr'],
            name='Top 10 Markers',
            mode='lines+markers'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_detection['concentration'],
            y=df_detection['mean_snr'],
            name='All Markers',
            mode='lines+markers'
        ),
        row=2, col=1
    )
    fig.update_xaxes(type="log", title_text="Concentration", row=2, col=1)
    fig.update_yaxes(title_text="SNR", row=2, col=1)
    # 4. Background Noise Distribution
    fig.add_histogram(
        x=cd4_markers['blood_background'],
        name='Background Distribution',
        row=2, col=2
    )
    fig.update_xaxes(title_text="Background Level", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        showlegend=True,
        title_text="CD4 Marker Detailed Analysis",
        legend=dict(
            x=0.4,      # Position legend at 30% from the left
            y=0.6,     # Position at 55% from the bottom
            xanchor='left',   # Anchor point on the left side of the legend
            yanchor='top',    # Anchor point at the top of the legend
            bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent white background
            bordercolor='rgba(0,0,0,0.2)',    # Light border
            borderwidth=1
        )
    )
    # Save to file
    fig.write_html(output_file)
    # Return analysis results
    return {
        'top_markers': sorted_markers.head(100).index.tolist(),
        'detection_results': detection_results,
        'detection_by_count': detection_by_count
    }


def analyse_marker_quality(df):
    blood_cells = ['B-cells', 'CD4-T-cells', 'CD8-T-cells', 'NK-cells', 'CD34-erythroblasts', 'CD34-megakaryocytes', 'Monocytes', 'Neutrophils', 'Eosinophils']
    # Get CD4 markers
    cd4_markers = df[df['target'] == 'CD4-T-cells'].copy()
    # Calculate marker quality metrics
    other_blood = [c for c in blood_cells if c != 'CD4-T-cells']
    cd4_markers['blood_background_max'] = cd4_markers[other_blood].max(axis=1)
    cd4_markers['snr'] = (cd4_markers['CD4-T-cells'] + 1e-10) / (cd4_markers['blood_background_max'] + 1e-10)
    # Create subplots for marker quality analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Marker Quality Distribution',
            'Background by Cell Type',
            'Top 20 Markers Profile',
            'Background Distribution by Cell Type'
        )
    )
    # 1. Marker Quality Distribution
    fig.add_trace(
        go.Scatter(
            x=cd4_markers['CD4-T-cells'],
            y=cd4_markers['blood_background_max'],
            mode='markers',
            name='Markers',
            marker=dict(
                color=cd4_markers['snr'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Signal-to-Noise Ratio<br>(CD4/max background)')
            ),
            hovertemplate="CD4: %{x:.3f}<br>Max Background: %{y:.3f}<br>SNR: %{marker.color:.2f}"
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text='CD4 Signal (proportion of reads)', row=1, col=1)
    fig.update_yaxes(title_text='Maximum Blood Background', row=1, col=1)
    # 2. Background by Cell Type
    for cell in other_blood:
        fig.add_trace(
            go.Box(
                y=cd4_markers[cell],
                name=cell,
                boxpoints='outliers'
            ),
            row=1, col=2
        )
    fig.update_yaxes(title_text='Background Signal', row=1, col=2)
    # 3. Top 20 Markers Profile
    top_20 = cd4_markers.nlargest(20, 'snr')
    x_vals = list(range(20))  # Convert range to list
    for cell in blood_cells:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=top_20[cell].values,  # Ensure we have numpy array or list
                name=cell,
                mode='lines+markers'
            ),
            row=2, col=1
        )
    fig.update_xaxes(title_text='Marker Rank', row=2, col=1)
    fig.update_yaxes(title_text='Signal', row=2, col=1)
    # 4. Background Distribution by Cell Type
    for cell in other_blood:
        fig.add_trace(
            go.Histogram(
                x=cd4_markers[cell],
                name=cell,
                opacity=0.7,
                nbinsx=30
            ),
            row=2, col=2
        )
    fig.update_xaxes(title_text='Signal Value', row=2, col=2)
    fig.update_yaxes(title_text='Count', row=2, col=2)
    # Update layout
    fig.update_layout(
        height=1200,  # Increase height slightly to accommodate legend on top
        width=1200,
        showlegend=True,
        title_text="CD4 Marker Quality Analysis",
        legend=dict(
            orientation="h",    # Horizontal legend
            yanchor="bottom",
            y=1.02,            # Place it above the plots
            xanchor="center",
            x=0.7,             # Center it
                bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    fig.write_html("marker_quality.html")
    return cd4_markers



def analyze_detection_limit(df):
    blood_cells = ['B-cells', 'CD4-T-cells', 'CD8-T-cells', 'NK-cells', 'CD34-erythroblasts', 'CD34-megakaryocytes', 'Monocytes', 'Neutrophils', 'Eosinophils']
    cd4_markers = df[df['target'] == 'CD4-T-cells'].copy()
    other_blood = [c for c in blood_cells if c != 'CD4-T-cells']
    cd4_markers['blood_background_max'] = cd4_markers[other_blood].max(axis=1)
    cd4_markers['snr'] = (cd4_markers['CD4-T-cells'] + 1e-10) / (cd4_markers['blood_background_max'] + 1e-10)
    # Get top 20 markers by SNR
    top_20 = cd4_markers.nlargest(20, 'snr')
    # Create simulation of signal vs background at different concentrations
    concentrations = [0.1, 0.01, 0.005, 0.001]  # 10%, 1%, 0.5%, 0.1%
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'CD4 Signal vs Background at Different Concentrations (Top 20 Markers)',
            'Signal Distribution at 1% CD4',
            'Distribution of Signal Ratios at Different Concentrations',
            'Signal vs Background Noise per Marker at 1% CD4'
        )
    )
    # 1. Signal vs Background line plot
    for marker_idx in range(20):
        marker = top_20.iloc[marker_idx]
        cd4_signals = []
        bg_signals = []
        for conc in concentrations:
            # CD4 signal at this concentration
            cd4_signal = marker['CD4-T-cells'] * conc
            # Background signal (other cells equally distributed in remaining proportion)
            other_conc = (1 - conc) / len(other_blood)
            bg_signal = sum(marker[cell] * other_conc for cell in other_blood)
            cd4_signals.append(cd4_signal)
            bg_signals.append(bg_signal)
        # Plot signal and background lines
        fig.add_trace(
            go.Scatter(
                x=concentrations,
                y=cd4_signals,
                name=f'CD4 Signal Marker {marker_idx+1}',
                line=dict(color='blue', width=1),
                opacity=0.3,
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=concentrations,
                y=bg_signals,
                name=f'Background Marker {marker_idx+1}',
                line=dict(color='red', width=1),
                opacity=0.3,
                showlegend=False
            ),
            row=1, col=1
        )
    # Add mean lines
    cd4_means = []
    bg_means = []
    for conc in concentrations:
        cd4_signal = top_20['CD4-T-cells'] * conc
        other_conc = (1 - conc) / len(other_blood)
        bg_signal = sum(top_20[cell] * other_conc for cell in other_blood)
        cd4_means.append(cd4_signal.mean())
        bg_means.append(bg_signal.mean())
    fig.add_trace(
        go.Scatter(
            x=concentrations,
            y=cd4_means,
            name='Mean CD4 Signal',
            line=dict(color='blue', width=3)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=concentrations,
            y=bg_means,
            name='Mean Background',
            line=dict(color='red', width=3)
        ),
        row=1, col=1
    )
    # 2. Signal Distribution at 1%
    cd4_signal_1pct = top_20['CD4-T-cells'] * 0.01
    other_conc = 0.99 / len(other_blood)
    bg_signal_1pct = sum(top_20[cell] * other_conc for cell in other_blood)
    fig.add_trace(
        go.Histogram(
            x=cd4_signal_1pct,
            name='CD4 Signal at 1%',
            opacity=0.7,
            nbinsx=20,
            marker_color='blue'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(
            x=bg_signal_1pct,
            name='Background at 1%',
            opacity=0.7,
            nbinsx=20,
            marker_color='red'
        ),
        row=1, col=2
    )
    # 3. Distribution of Signal Ratios
    ratios = []
    ratio_concs = []
    for conc in concentrations:
        cd4_signal = top_20['CD4-T-cells'] * conc
        other_conc = (1 - conc) / len(other_blood)
        bg_signal = sum(top_20[cell] * other_conc for cell in other_blood)
        ratio = cd4_signal / (bg_signal + 1e-10)
        ratios.extend(ratio)
        ratio_concs.extend([conc] * len(ratio))
    for conc in concentrations:
        mask = [c == conc for c in ratio_concs]
        fig.add_trace(
            go.Box(
                y=[r for r, m in zip(ratios, mask) if m],
                name=f'{conc*100}%',
                boxpoints='all'
            ),
            row=2, col=1
        )
    
    # 4. Signal vs Background Scatter at 1%
    fig.add_trace(
        go.Scatter(
            x=bg_signal_1pct,
            y=cd4_signal_1pct,
            mode='markers',
            name='Markers at 1% CD4',
            marker=dict(
                color=cd4_signal_1pct/(bg_signal_1pct + 1e-10),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Signal Ratio')
            )
        ),
        row=2, col=2
    )
    # Update layouts
    fig.update_xaxes(title_text='CD4 Concentration', type='log', row=1, col=1)
    fig.update_yaxes(title_text='Signal', row=1, col=1)
    fig.update_xaxes(title_text='Signal Value', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    fig.update_xaxes(title_text='Concentration', row=2, col=1)
    fig.update_yaxes(title_text='Signal Ratio (CD4/Background)', type='log', row=2, col=1)
    fig.update_xaxes(title_text='Background Signal', row=2, col=2)
    fig.update_yaxes(title_text='CD4 Signal', row=2, col=2)
    fig.update_layout(
        height=1000,
        width=1200,
        showlegend=True,
        title_text="CD4 Detection Limit Analysis",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.45,
            xanchor="center",
            x=0.5
        )
    )
    fig.write_html("CD4_detection_limits.html")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze methylation atlas data')
    parser.add_argument('input_file', help='Path to input atlas file')
    parser.add_argument('--output', default='atlas_analysis.html', help='Path to output HTML file')
    
    args = parser.parse_args()
    results = run_analysis(args.input_file, args.output)
    print("Analysis complete. Check the output HTML file for visualizations.")