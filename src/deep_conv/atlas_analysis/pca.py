import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances


def decorrelate_reference_profiles(X, n_components=None):
    """
    Decorrelate cell type reference profiles using PCA.
    Parameters
    ----------
    X : np.ndarray
        Input matrix of shape (n_cell_types, n_markers) where rows are cell types 
        and columns are markers.
    n_components : int or None
        Number of principal components to keep. If None, keep all components.
    Returns
    -------
    X_pca : np.ndarray
        Decorrelated reference matrix of shape (n_cell_types, n_components).
    pca : PCA object
        Fitted PCA object that can be used for inverse_transform or to inspect components.
    kept_indices : np.ndarray
        Indices of columns (markers) that were kept after removing NaN columns.
    """
    # 1. Remove columns with NaNs
    # Identify columns without any NaN
    non_nan_mask = ~np.isnan(X).any(axis=0)
    X_clean = X[:, non_nan_mask]
    # If after removing NaN columns, no markers remain, raise an error
    if X_clean.shape[1] == 0:
        raise ValueError("All markers contain NaNs, no data left after filtering.")
    # 2. Center and scale data (optional, but recommended)
    # Standardizing ensures all markers contribute equally.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_clean)  # Shape: (n_cell_types, n_kept_markers)
    # 3. Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    return X_pca, pca, np.where(non_nan_mask)[0]

X = pd.read_csv("/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed",sep="\t")
cell_types = X.columns[8:]
X_original = X[X.columns[8:]].dropna().T.to_numpy()
X = X[X.columns[8:]].T.to_numpy()

X_pca, pca, kept_markers = decorrelate_reference_profiles(X)
plt.figure(figsize=(8,6))
for i, ct in enumerate(cell_types):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], label=ct)

plt.title("PCA Projection of Cell Types")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig("/mnt/lustre/users/zetzioni/deepconv/src/plots/atlas_pca.png")

dist_original = pairwise_distances(X_original, metric='euclidean')
dist_pca = pairwise_distances(X_pca, metric='euclidean')

fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=dist_original,
    x=cell_types,
    y=cell_types,
    colorscale='Viridis',
    showscale=True
))
fig.update_layout(title='Distance Matrix (Original)')
fig.write_html("/mnt/lustre/users/zetzioni/deepconv/src/plots/distance_matrix_original.html")

# Another figure for PCA distances
fig_pca = go.Figure()

fig_pca.add_trace(go.Heatmap(
    z=dist_pca,
    x=cell_types,
    y=cell_types,
    colorscale='Viridis',
    showscale=True
))
fig_pca.update_layout(title='Distance Matrix (PCA)')
fig_pca.write_html("/mnt/lustre/users/zetzioni/deepconv/src/plots/distance_matrix_pca.html")

corr_original = np.corrcoef(X_original)
corr_pca = np.corrcoef(X_pca)

fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=corr_original,
    x=cell_types,
    y=cell_types,
    colorscale='RdBu',
    zmin=-1, zmax=1
))
fig.update_layout(title='Correlation Matrix (Original)')
fig.write_html("/mnt/lustre/users/zetzioni/deepconv/src/plots/correlation_original.html")
fig.write_image("/mnt/lustre/users/zetzioni/deepconv/src/plots/correlation_original.png", scale=2)

fig_pca = go.Figure()
fig_pca.add_trace(go.Heatmap(
    z=corr_pca,
    x=cell_types,
    y=cell_types,
    colorscale='RdBu',
    zmin=-1, zmax=1
))
fig_pca.update_layout(title='Correlation Matrix (PCA)')
fig_pca.write_html("/mnt/lustre/users/zetzioni/deepconv/src/plots/correlation_pca.html")


mds_original = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
X_mds_original = mds_original.fit_transform(dist_original)

mds_pca = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
X_mds_pca = mds_pca.fit_transform(dist_pca)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X_mds_original[:, 0], 
    y=X_mds_original[:, 1],
    mode='markers+text',
    text=cell_types,
    textposition='bottom center'
))
fig.update_layout(title='MDS (Original)')
fig.write_html("/mnt/lustre/users/zetzioni/deepconv/src/plots/mds_original.html")

fig_pca = go.Figure()
fig_pca.add_trace(go.Scatter(
    x=X_mds_pca[:, 0], 
    y=X_mds_pca[:, 1],
    mode='markers+text',
    text=cell_types,
    textposition='bottom center'
))
fig_pca.update_layout(title='MDS (PCA)')
fig_pca.write_html("/mnt/lustre/users/zetzioni/deepconv/src/plots/mds_pca.html")