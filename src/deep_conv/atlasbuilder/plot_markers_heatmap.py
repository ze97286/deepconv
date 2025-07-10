import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def plot_methylation_heatmap(marker_df, coverage_df=None, figsize=(20, 12), 
                           cmap='RdBu_r', nan_color='gray',
                           cluster_samples=True, cluster_regions=False,
                           sample_labels=None, region_labels=None,
                           title='Methylation Marker Heatmap'):
    """
    Plot heatmap of methylation markers with coverage adjustment
    
    Parameters:
    -----------
    marker_df : pd.DataFrame
        DataFrame with regions as rows and samples as columns, containing marker values
    coverage_df : pd.DataFrame, optional
        DataFrame with same structure containing coverage values. If None, uses marker values only
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap for the heatmap
    nan_color : str
        Color to use for NaN values
    cluster_samples : bool
        Whether to cluster samples (columns)
    cluster_regions : bool
        Whether to cluster regions (rows)
    sample_labels : list, optional
        Custom labels for samples
    region_labels : list, optional
        Custom labels for regions
    title : str
        Plot title
    """
    
    # Create adjusted values matrix
    if coverage_df is not None:
        # Calculate min(1, marker_value * coverage)
        adjusted_values = np.minimum(1, marker_df * coverage_df)
    else:
        adjusted_values = marker_df.copy()
    
    # Create a mask for NaN values
    nan_mask = pd.isna(adjusted_values)
    
    # For clustering purposes, temporarily fill NaN with median of each row
    if cluster_samples or cluster_regions:
        adjusted_values_for_clustering = adjusted_values.copy()
        for idx in adjusted_values_for_clustering.index:
            row_median = adjusted_values_for_clustering.loc[idx].median()
            if pd.isna(row_median):  # If entire row is NaN
                adjusted_values_for_clustering.loc[idx] = 0
            else:
                adjusted_values_for_clustering.loc[idx].fillna(row_median, inplace=True)
    
    # Perform clustering if requested
    if cluster_samples and cluster_regions:
        # Cluster both
        g = sns.clustermap(adjusted_values_for_clustering, 
                          cmap=cmap, 
                          figsize=figsize,
                          cbar_pos=(0.02, 0.83, 0.03, 0.15),
                          dendrogram_ratio=(0.15, 0.15))
        
        # Reorder the original data and mask according to clustering
        adjusted_values = adjusted_values.iloc[g.dendrogram_row.reordered_ind, 
                                               g.dendrogram_col.reordered_ind]
        nan_mask = nan_mask.iloc[g.dendrogram_row.reordered_ind, 
                                g.dendrogram_col.reordered_ind]
        
        # Extract the heatmap axes
        ax = g.ax_heatmap
        fig = g.fig
        
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        if cluster_samples:
            # Cluster only columns
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist, squareform
            
            # Calculate linkage
            col_linkage = linkage(pdist(adjusted_values_for_clustering.T, metric='euclidean'), 
                                 method='average')
            col_dendrogram = dendrogram(col_linkage, no_plot=True)
            col_order = col_dendrogram['leaves']
            
            adjusted_values = adjusted_values.iloc[:, col_order]
            nan_mask = nan_mask.iloc[:, col_order]
        
        if cluster_regions:
            # Cluster only rows
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist, squareform
            
            # Calculate linkage
            row_linkage = linkage(pdist(adjusted_values_for_clustering, metric='euclidean'), 
                                 method='average')
            row_dendrogram = dendrogram(row_linkage, no_plot=True)
            row_order = row_dendrogram['leaves']
            
            adjusted_values = adjusted_values.iloc[row_order, :]
            nan_mask = nan_mask.iloc[row_order, :]
    
    # Plot the heatmap
    if not (cluster_samples and cluster_regions):  # If not using clustermap
        im = ax.imshow(adjusted_values, aspect='auto', cmap=cmap, 
                      interpolation='nearest', vmin=0, vmax=1)
        
        # Overlay NaN values
        for i in range(nan_mask.shape[0]):
            for j in range(nan_mask.shape[1]):
                if nan_mask.iloc[i, j]:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                             fill=True, color=nan_color, 
                                             linewidth=0))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Adjusted Marker Value\n(min(1, marker × coverage))', 
                      rotation=270, labelpad=20)
    
    # Set labels
    if sample_labels is None:
        sample_labels = adjusted_values.columns
    if region_labels is None:
        region_labels = [f'Region_{i}' for i in range(len(adjusted_values))]
    
    if not (cluster_samples and cluster_regions):
        ax.set_xticks(range(len(sample_labels)))
        ax.set_xticklabels(sample_labels, rotation=90, ha='center')
        ax.set_yticks(range(len(region_labels)))
        ax.set_yticklabels(region_labels, fontsize=8)
    
    # Add title
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Samples', fontsize=12)
    ax.set_ylabel('Regions', fontsize=12)
    
    # Add legend for NaN values
    nan_patch = mpatches.Patch(color=nan_color, label='Missing data')
    ax.legend(handles=[nan_patch], loc='upper left', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    return fig, ax, adjusted_values


def plot_sample_similarity(marker_df, coverage_df=None, method='correlation', 
                         figsize=(12, 10), cmap='coolwarm'):
    """
    Plot similarity matrix between samples based on their methylation patterns
    
    Parameters:
    -----------
    marker_df : pd.DataFrame
        DataFrame with regions as rows and samples as columns
    coverage_df : pd.DataFrame, optional
        DataFrame with coverage values
    method : str
        Similarity method: 'correlation' or 'euclidean'
    """
    
    # Create adjusted values
    if coverage_df is not None:
        adjusted_values = np.minimum(1, marker_df * coverage_df)
    else:
        adjusted_values = marker_df.copy()
    
    # Handle NaN values for similarity calculation
    adjusted_values_filled = adjusted_values.fillna(adjusted_values.median(axis=1))
    
    # Calculate similarity matrix
    if method == 'correlation':
        similarity_matrix = adjusted_values_filled.corr()
        title = 'Sample Correlation Matrix'
        vmin, vmax = -1, 1
    else:  # euclidean
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(adjusted_values_filled.T, metric='euclidean')
        similarity_matrix = pd.DataFrame(
            1 - squareform(distances) / distances.max(),
            index=adjusted_values.columns,
            columns=adjusted_values.columns
        )
        title = 'Sample Similarity Matrix (1 - normalized Euclidean distance)'
        vmin, vmax = 0, 1
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use hierarchical clustering to order samples
    g = sns.clustermap(similarity_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                      figsize=figsize, cbar_pos=(0.02, 0.83, 0.03, 0.15))
    
    g.ax_heatmap.set_title(title, fontsize=16, pad=20)
    
    return g


if __name__ == "__main__":
    marker_data = pd.read_parquet("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/first_line/AB/tissue/marker_values.parquet")
    coverage_data = pd.read_parquet("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/first_line/AB/tissue/coverage.parquet")

    baseline_samples = sorted([x for x in marker_data.columns if "Scr" in x and "tumour" in x])
    immonly_samples = sorted([x for x in marker_data.columns if "Imm" in x and "tumour" in x])

    # Plot heatmap
    fig, ax, adjusted_values = plot_methylation_heatmap(
        marker_data[baseline_samples], 
        coverage_data[baseline_samples],
        cluster_samples=True,
        cluster_regions=True,
        title='Methylation Markers - Adjusted by Coverage'
    )

    plt.savefig("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/first_line/AB/tissue/baseline_markers_heatmap_after_first_line_filtering.png")

    # Plot heatmap
    fig, ax, adjusted_values = plot_methylation_heatmap(
        marker_data[immonly_samples], 
        coverage_data[immonly_samples],
        cluster_samples=True,
        cluster_regions=True,
        title='Methylation Markers - Adjusted by Coverage'
    )

    plt.savefig("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/first_line/AB/tissue/immonly_markers_heatmap_after_first_line_filtering.png")


    # Plot sample similarity
    g = plot_sample_similarity(marker_data[baseline_samples], coverage_data[baseline_samples])
    plt.savefig("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/first_line/AB/tissue/baseline_markers_similarity_after_first_line_filtering.png")

    g = plot_sample_similarity(marker_data[immonly_samples], coverage_data[immonly_samples])
    plt.savefig("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/first_line/AB/tissue/immonly_markers_similarity_after_first_line_filtering.png")
