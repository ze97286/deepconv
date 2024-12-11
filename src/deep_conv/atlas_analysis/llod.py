import pandas as pd
import numpy as np
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm  


import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

def theoretical_detection_limits(reference_profiles, base_coverage=30, n_jobs=-1):
    """
    Analyze theoretical detection limits without relying on training data
    """
    n_cell_types, n_markers = reference_profiles.shape
    concentrations = np.logspace(-4, -1, 50)  # 0.01% to 10%
    
    def process_cell_concentration(cell_type, conc):
        # Calculate theoretically perfect signal change
        target_profile = reference_profiles[cell_type]
        other_profiles = np.delete(reference_profiles, cell_type, axis=0)
        background = other_profiles.mean(axis=0)  # Mean profile of other cell types
        
        # Calculate delta and SNR
        mixture = (1 - conc) * background + conc * target_profile
        delta = np.abs(mixture - background)
        snr = delta * np.sqrt(base_coverage)  # Basic statistical SNR
        
        # Marker-level statistics
        threshold = 0.1  # Fixed threshold for distinguishing markers
        n_markers_above_threshold = np.sum(delta > threshold)
        max_delta = np.max(delta)
        mean_delta = np.mean(delta)
        max_snr = np.max(snr)
        mean_snr = np.mean(snr)
        
        return {
            'cell_type': cell_type,
            'concentration': conc,
            'max_delta': max_delta,
            'mean_delta': mean_delta,
            'n_markers_above_threshold': n_markers_above_threshold,
            'max_snr': max_snr,
            'mean_snr': mean_snr
        }
    
    # Parallelize computations across cell types and concentrations
    tasks = [(cell_type, conc) for cell_type in range(n_cell_types) for conc in concentrations]
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell_concentration)(cell_type, conc) for cell_type, conc in tqdm(tasks, desc="Processing Cell Types and Concentrations")
    )
    
    return pd.DataFrame(results)


def plot_detection_limits(results, cell_types):
    """
    Create an interactive Plotly visualization for theoretical detection limits.
    
    Args:
        results (pd.DataFrame): DataFrame containing detection limit analysis results.
        cell_types (list): List of cell type names.
        output_path (str): Path to save the interactive HTML plot.
    """
    fig = go.Figure()
    for cell_type in range(len(cell_types)):
        cell_data = results[results['cell_type'] == cell_type]
        # Plot number of distinguishable markers
        fig.add_trace(go.Scatter(
            x=cell_data['concentration'],
            y=cell_data['n_markers_above_threshold'],
            name=f"{cell_types[cell_type]} - Markers",
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            yaxis="y1"
        ))
        # Plot mean SNR on a secondary y-axis
        fig.add_trace(go.Scatter(
            x=cell_data['concentration'],
            y=cell_data['mean_snr'],
            name=f"{cell_types[cell_type]} - Mean SNR",
            mode='lines+markers',
            line=dict(dash='dot'),
            marker=dict(size=6),
            yaxis="y2"
        ))
    # Update layout for dual y-axes
    fig.update_layout(
        title="Theoretical Detection Limits: Number of Markers and Mean SNR by Concentration",
        xaxis=dict(title="Concentration", type="log"),
        yaxis=dict(title="Number of Distinguishable Markers", side="left"),
        yaxis2=dict(
            title="Mean SNR",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(title="Cell Types"),
        template="plotly_white",
        hovermode="closest"
    )
    # Save the plot
    fig.write_html("/users/zetzioni/sharedscratch/deepconv/data/images/theoretical_detection_limits.html")
    


def analyse_marker_distinguishability(reference_profiles, min_concentration=0.0001, max_concentration=0.2, n_samples=10000, n_jobs=-1):
    """
    Analyse marker distinguishability by comparing distributions of mixtures with and without target cell type
    """
    from scipy.stats import ttest_ind
    n_cell_types, n_markers = reference_profiles.shape
    concentrations = np.logspace(np.log10(min_concentration), np.log10(max_concentration), 50)
    results = []
    # Precompute all Dirichlet samples for background mixtures per cell type and concentration
    def process_cell_conc(cell_type, conc):
        # Background mixtures
        other_cell_types = np.delete(np.arange(n_cell_types), cell_type)
        other_profiles = reference_profiles[other_cell_types, :]  # Shape: (n_cell_types-1, n_markers)
        # Generate background proportions
        bg_props = np.random.dirichlet(alpha=np.ones(n_cell_types-1), size=n_samples)  # Shape: (n_samples, n_cell_types-1)
        background_mixtures = bg_props @ other_profiles  # Shape: (n_samples, n_markers)
        # Generate target proportions
        target_props = np.random.dirichlet(alpha=np.ones(n_cell_types-1), size=n_samples) * (1 - conc)  # Shape: (n_samples, n_cell_types-1)
        target_mixtures = conc * reference_profiles[cell_type] + target_props @ other_profiles  # Shape: (n_samples, n_markers)
        # Statistical testing per marker
        # Using t-test; switch to mannwhitneyu if non-parametric is desired
        t_stats, p_values = ttest_ind(target_mixtures, background_mixtures, axis=0, equal_var=False)
        # Calculate effect sizes (Cohen's d)
        bg_means = background_mixtures.mean(axis=0)
        bg_stds = background_mixtures.std(axis=0)
        target_means = target_mixtures.mean(axis=0)
        effect_sizes = np.abs(target_means - bg_means) / (bg_stds + 1e-10)
        # Calculate average absolute correlation for each marker with other markers in both backgrounds and targets
        # To save computation, compute only for each mixture type separately and average
        bg_corr_matrix = np.corrcoef(background_mixtures, rowvar=False)  # Shape: (n_markers, n_markers)
        target_corr_matrix = np.corrcoef(target_mixtures, rowvar=False)  # Shape: (n_markers, n_markers)
        # Avoid NaNs in correlation matrices by replacing them with zero (happens if a marker has zero variance)
        bg_corr_matrix = np.nan_to_num(bg_corr_matrix)
        target_corr_matrix = np.nan_to_num(target_corr_matrix)
        # Compute mean absolute correlation excluding self-correlation
        bg_mean_corr = (np.abs(bg_corr_matrix).sum(axis=1) - 1) / (n_markers - 1)  # Shape: (n_markers,)
        target_mean_corr = (np.abs(target_corr_matrix).sum(axis=1) - 1) / (n_markers - 1)  # Shape: (n_markers,)
        mean_correlation = (bg_mean_corr + target_mean_corr) / 2  # Shape: (n_markers,)
        
        normalized_correlation = (mean_correlation - mean_correlation.min()) / (mean_correlation.max() - mean_correlation.min() + 1e-10)
        independence_scores = 1 - normalized_correlation
        # independence_scores = 1 / (mean_correlation + 1e-10)  # Shape: (n_markers,)
        # independence_scores = 1 / (np.log1p(mean_correlation) + 1e-10)
        # Compute marker scores
        marker_scores = effect_sizes * independence_scores  # Shape: (n_markers,)
        # Determine distinguishable markers (e.g., p-value < 0.05 after correction and/or effect_size > threshold)
        # Here, using p-value < 0.05 and effect_size > 0.5 as an example
        significance_threshold = 0.05
        effect_size_threshold = 0.5
        distinguishable = (p_values < significance_threshold) & (effect_sizes > effect_size_threshold)
        n_distinguishable = np.sum(distinguishable)
        # Identify top 10 markers based on marker_scores
        top_markers = np.argsort(marker_scores)[-10:]
        return {
            'cell_type': cell_type,
            'concentration': conc,
            'n_distinguishable': n_distinguishable,
            'top_markers': top_markers.tolist()
        }
    tasks = []
    for cell_type in range(n_cell_types):
        for conc in concentrations:
            tasks.append((cell_type, conc))
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell_conc)(cell_type, conc) for cell_type, conc in tqdm(tasks, desc="Processing Cell Types and Concentrations")
    )
    results_df = pd.DataFrame(results)
    return results_df


def plot_analysis_results(df, cell_types, suffix):
    """Enhanced visualization including effect of correlation"""
    fig = go.Figure()
    for cell_type in range(len(cell_types)):
        cell_data = df[df['cell_type'] == cell_type]
        fig.add_trace(go.Scatter(
            x=cell_data['concentration'],
            y=cell_data['n_distinguishable'],
            name=cell_types[cell_type],
            mode='lines+markers'
        ))
    fig.update_layout(
        title='Number of Distinguishable Independent Markers by Concentration (with normalised correlation)',
        xaxis_title='Concentration',
        yaxis_title='Number of Distinguishable Independent Markers',
        xaxis_type='log',
        showlegend=True
    )
    fig.write_html(f"/users/zetzioni/sharedscratch/deepconv/data/images/Number_of_Distinguishable_Independent_Markers_by_Concentration_{suffix}.html")

