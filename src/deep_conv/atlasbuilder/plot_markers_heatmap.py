import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def plot_methylation_heatmap(marker_df, coverage_df=None, tf_df=None, figsize=(20, 12), 
                           cmap='RdBu_r', nan_color='gray',
                           cluster_samples=True, cluster_regions=False,
                           sample_labels=None, region_labels=None,
                           title='Methylation Marker Heatmap',
                           show_tf_values=False, show_ploidy_values=False,
                           discrete_annotations=False):
    """
    Plot heatmap of methylation markers with coverage adjustment and TF/ploidy annotations
    
    Parameters:
    -----------
    marker_df : pd.DataFrame
        DataFrame with regions as rows and samples as columns, containing marker values
    coverage_df : pd.DataFrame, optional
        DataFrame with same structure containing coverage values. If None, uses marker values only
    tf_df : pd.DataFrame, optional
        DataFrame with columns ['sample', 'tf', 'ploidy'] for sample annotations
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
    show_tf_values : bool, optional
        Whether to show TF values as text on the annotation track
    show_ploidy_values : bool, optional
        Whether to show ploidy values as text on the annotation track
    discrete_annotations : bool, optional
        Whether to use discrete categories for annotations instead of continuous gradient
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
    # Prepare TF/ploidy annotations if provided
    tf_annotation = None
    ploidy_annotation = None
    if tf_df is not None:
        # Create a mapping from sample name to tf and ploidy values
        tf_dict = dict(zip(tf_df['sample'], tf_df['tf']))
        ploidy_dict = dict(zip(tf_df['sample'], tf_df['ploidy']))
        # Map sample columns to tf and ploidy values
        tf_values = [tf_dict.get(col, np.nan) for col in adjusted_values.columns]
        ploidy_values = [ploidy_dict.get(col, np.nan) for col in adjusted_values.columns]
        tf_annotation = pd.DataFrame({'TF': tf_values}, index=adjusted_values.columns)
        ploidy_annotation = pd.DataFrame({'Ploidy': ploidy_values}, index=adjusted_values.columns)
        
        # Define discrete categories if requested
        if discrete_annotations:
            # TF categories: Low (0-0.2), Medium (0.2-0.5), High (>0.5)
            tf_categories = []
            tf_cat_colors = {'Low': '#440154', 'Medium': '#31688e', 'High': '#fde725', 'NA': 'lightgray'}
            for tf in tf_values:
                if pd.isna(tf):
                    tf_categories.append('NA')
                elif tf <= 0.2:
                    tf_categories.append('Low')
                elif tf <= 0.5:
                    tf_categories.append('Medium')
                else:
                    tf_categories.append('High')
            
            # Ploidy categories: Diploid (~2), Low Aneuploidy (2-3), High Aneuploidy (>3)
            ploidy_categories = []
            ploidy_cat_colors = {'Diploid': '#0d0887', 'Low Aneuploidy': '#cc4778', 'High Aneuploidy': '#f0f921', 'NA': 'lightgray'}
            for ploidy in ploidy_values:
                if pd.isna(ploidy):
                    ploidy_categories.append('NA')
                elif ploidy <= 2.2:
                    ploidy_categories.append('Diploid')
                elif ploidy <= 3.0:
                    ploidy_categories.append('Low Aneuploidy')
                else:
                    ploidy_categories.append('High Aneuploidy')
    # Perform clustering if requested
    if cluster_samples and cluster_regions:
        # Create row and column annotations for clustermap
        col_colors = None
        if tf_df is not None:
            if discrete_annotations:
                # Use discrete category colors
                tf_colors_list = [tf_cat_colors[cat] for cat in tf_categories]
                ploidy_colors_list = [ploidy_cat_colors[cat] for cat in ploidy_categories]
            else:
                # Create color mapping for TF values
                tf_norm = plt.Normalize(vmin=0, vmax=1)
                tf_cmap = plt.cm.viridis
                tf_colors_list = []
                for col in adjusted_values.columns:
                    tf_val = tf_annotation.loc[col, 'TF']
                    if pd.isna(tf_val):
                        tf_colors_list.append('lightgray')
                    else:
                        tf_colors_list.append(tf_cmap(tf_norm(tf_val)))
                # Create color mapping for ploidy values
                ploidy_norm = plt.Normalize(vmin=1.5, vmax=4.0)
                ploidy_cmap = plt.cm.plasma
                ploidy_colors_list = []
                for col in adjusted_values.columns:
                    ploidy_val = ploidy_annotation.loc[col, 'Ploidy']
                    if pd.isna(ploidy_val):
                        ploidy_colors_list.append('lightgray')
                    else:
                        ploidy_colors_list.append(ploidy_cmap(ploidy_norm(ploidy_val)))
            # Create DataFrame with colors for seaborn
            col_colors = pd.DataFrame({
                'TF': tf_colors_list,
                'Ploidy': ploidy_colors_list
            }, index=adjusted_values.columns)
        # Cluster both with annotations
        g = sns.clustermap(adjusted_values_for_clustering, 
                          cmap=cmap, 
                          figsize=figsize,
                          cbar_pos=(0.02, 0.83, 0.03, 0.15),
                          dendrogram_ratio=(0.15, 0.15),
                          col_colors=col_colors)
        # Reorder the original data and mask according to clustering
        adjusted_values = adjusted_values.iloc[g.dendrogram_row.reordered_ind, 
                                               g.dendrogram_col.reordered_ind]
        nan_mask = nan_mask.iloc[g.dendrogram_row.reordered_ind, 
                                g.dendrogram_col.reordered_ind]
        # Extract the heatmap axes
        ax = g.ax_heatmap
        fig = g.fig
        
        # Add legends for annotation colors when using clustermap
        if tf_df is not None:
            pos = ax.get_position()
            
            if discrete_annotations:
                # Create discrete legends
                # TF legend
                tf_legend_elements = [mpatches.Patch(color=color, label=label) 
                                     for label, color in tf_cat_colors.items() if label != 'NA']
                tf_legend = fig.add_subplot(111, frame_on=False)
                tf_legend.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
                tf_legend.legend(handles=tf_legend_elements, title='TF Categories',
                               bbox_to_anchor=(pos.x1 + 0.15, pos.y0 + pos.height * 0.85),
                               loc='upper left', frameon=False)
                
                # Ploidy legend
                ploidy_legend_elements = [mpatches.Patch(color=color, label=label) 
                                         for label, color in ploidy_cat_colors.items() if label != 'NA']
                ploidy_legend = fig.add_subplot(111, frame_on=False)
                ploidy_legend.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
                ploidy_legend.legend(handles=ploidy_legend_elements, title='Ploidy Categories',
                                   bbox_to_anchor=(pos.x1 + 0.15, pos.y0 + pos.height * 0.5),
                                   loc='upper left', frameon=False)
            else:
                # Create continuous colorbars
                # TF colorbar
                tf_norm = plt.Normalize(vmin=0, vmax=1)
                tf_sm = plt.cm.ScalarMappable(cmap='viridis', norm=tf_norm)
                tf_sm.set_array([])
                
                tf_cax = fig.add_axes([pos.x1 + 0.15, pos.y0 + pos.height * 0.7, 0.02, pos.height * 0.15])
                tf_cbar = fig.colorbar(tf_sm, cax=tf_cax)
                tf_cbar.set_label('TF', rotation=270, labelpad=15)
                
                # Ploidy colorbar
                ploidy_norm = plt.Normalize(vmin=1.5, vmax=4.0)
                ploidy_sm = plt.cm.ScalarMappable(cmap='plasma', norm=ploidy_norm)
                ploidy_sm.set_array([])
                
                ploidy_cax = fig.add_axes([pos.x1 + 0.15, pos.y0 + pos.height * 0.4, 0.02, pos.height * 0.15])
                ploidy_cbar = fig.colorbar(ploidy_sm, cax=ploidy_cax)
                ploidy_cbar.set_label('Ploidy', rotation=270, labelpad=15)
    else:
        # Create subplots with space for annotations
        if tf_df is not None:
            # Adjust figure layout to accommodate annotation tracks
            fig = plt.figure(figsize=figsize)
            # Create grid: 2 annotation rows + main heatmap
            gs = fig.add_gridspec(4, 1, height_ratios=[0.3, 0.3, 6, 0.1], hspace=0.02)
            # Annotation axes
            ax_tf = fig.add_subplot(gs[0])
            ax_ploidy = fig.add_subplot(gs[1])
            ax = fig.add_subplot(gs[2])  # Main heatmap
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
            # Reorder annotations too
            if tf_df is not None:
                tf_annotation = tf_annotation.iloc[col_order]
                ploidy_annotation = ploidy_annotation.iloc[col_order]
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
        # Plot annotation tracks if provided and not using clustermap
        if tf_df is not None and 'ax_tf' in locals():
            # Plot TF annotation
            tf_values_ordered = [tf_annotation.loc[col, 'TF'] for col in adjusted_values.columns]
            # Create TF colormap normalization
            tf_norm = plt.Normalize(vmin=0, vmax=1)
            tf_cmap = plt.cm.viridis
            for i, (sample, tf_val) in enumerate(zip(adjusted_values.columns, tf_values_ordered)):
                if pd.isna(tf_val):
                    color = 'lightgray'
                else:
                    color = tf_cmap(tf_norm(tf_val))
                ax_tf.add_patch(plt.Rectangle((i-0.5, -0.5), 1, 1, fill=True, color=color))
            ax_tf.set_xlim(-0.5, len(adjusted_values.columns)-0.5)
            ax_tf.set_ylim(-0.5, 0.5)
            ax_tf.set_ylabel('TF', rotation=0, ha='right', va='center')
            ax_tf.set_xticks([])
            ax_tf.set_yticks([])
            
            # Add text values if requested
            if show_tf_values:
                for i, tf_val in enumerate(tf_values_ordered):
                    if not pd.isna(tf_val):
                        ax_tf.text(i, 0, f'{tf_val:.2f}', ha='center', va='center', 
                                  fontsize=8, color='white' if tf_val > 0.5 else 'black')
            # Plot ploidy annotation
            ploidy_values_ordered = [ploidy_annotation.loc[col, 'Ploidy'] for col in adjusted_values.columns]
            # Create ploidy colormap normalization
            ploidy_norm = plt.Normalize(vmin=1.5, vmax=4.0)
            ploidy_cmap = plt.cm.plasma
            for i, (sample, ploidy_val) in enumerate(zip(adjusted_values.columns, ploidy_values_ordered)):
                if pd.isna(ploidy_val):
                    color = 'lightgray'
                else:
                    color = ploidy_cmap(ploidy_norm(ploidy_val))
                ax_ploidy.add_patch(plt.Rectangle((i-0.5, -0.5), 1, 1, fill=True, color=color))
            ax_ploidy.set_xlim(-0.5, len(adjusted_values.columns)-0.5)
            ax_ploidy.set_ylim(-0.5, 0.5)
            ax_ploidy.set_ylabel('Ploidy', rotation=0, ha='right', va='center')
            ax_ploidy.set_xticks([])
            ax_ploidy.set_yticks([])
            
            # Add text values if requested
            if show_ploidy_values:
                for i, ploidy_val in enumerate(ploidy_values_ordered):
                    if not pd.isna(ploidy_val):
                        # Determine text color based on ploidy value
                        text_color = 'white' if ploidy_val > 2.75 else 'black'
                        ax_ploidy.text(i, 0, f'{ploidy_val:.2f}', ha='center', va='center', 
                                      fontsize=8, color=text_color)
            # Add annotation colorbars
            # TF colorbar
            tf_sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
            tf_cbar = fig.colorbar(tf_sm, ax=ax_tf, fraction=0.02, pad=0.01, aspect=10)
            tf_cbar.set_label('TF', rotation=270, labelpad=10)
            # Ploidy colorbar  
            ploidy_sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=1.5, vmax=4.0))
            ploidy_cbar = fig.colorbar(ploidy_sm, ax=ax_ploidy, fraction=0.02, pad=0.01, aspect=10)
            ploidy_cbar.set_label('Ploidy', rotation=270, labelpad=10)
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
    marker_data = pd.read_parquet("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_10k_oac.l4/tissue/marker_values.parquet")
    coverage_data = pd.read_parquet("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_10k_oac.l4/tissue/coverage.parquet")

    baseline_samples = sorted([x for x in marker_data.columns if "Scr" in x and "tumour" in x])
    immonly_samples = sorted([x for x in marker_data.columns if "Imm" in x and "tumour" in x])

    tf = pd.read_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_10k_oac.l4/tissue/tissue_tf.csv", sep="\t")

    # Plot heatmap with discrete annotations for better interpretability
    fig, ax, adjusted_values = plot_methylation_heatmap(
        marker_data[baseline_samples],
        coverage_data[baseline_samples],
        tf_df=tf,
        cluster_samples=True,
        cluster_regions=True,
        title='Methylation Markers with TF/Ploidy',
        discrete_annotations=True,    # Clear categories
        show_tf_values=True,         # Show actual TF values
        show_ploidy_values=True      # Show actual ploidy values
    )

    plt.savefig("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_10k_oac.l4/tissue/baseline_markers_heatmap_10k_oac_markers.png")
    # continuous gradients with legends
    fig, ax, adjusted_values = plot_methylation_heatmap(
        marker_data[baseline_samples],
        coverage_data[baseline_samples],
        tf_df=tf,
        discrete_annotations=False   # Continuous gradients
    )

    plt.savefig("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_10k_oac.l4/tissue/continuous_baseline_markers_heatmap_10k_oac_markers.png")

    # Plot heatmap with discrete annotations for better interpretability
    fig, ax, adjusted_values = plot_methylation_heatmap(
        marker_data[immonly_samples], 
        coverage_data[immonly_samples],
        tf_df=tf,
        cluster_samples=True,
        cluster_regions=True,
        title='Methylation Markers - Adjusted by Coverage',
        discrete_annotations=True,
        show_tf_values=True,
        show_ploidy_values=True
    )

    plt.savefig("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/first_line/AB/tissue/immonly_markers_heatmap_after_first_line_filtering.png")


    # Plot sample similarity
    g = plot_sample_similarity(marker_data[baseline_samples], coverage_data[baseline_samples])
    plt.savefig("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/first_line/AB/tissue/baseline_markers_similarity_after_first_line_filtering.png")

    g = plot_sample_similarity(marker_data[immonly_samples], coverage_data[immonly_samples])
    plt.savefig("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/first_line/AB/tissue/immonly_markers_similarity_after_first_line_filtering.png")
