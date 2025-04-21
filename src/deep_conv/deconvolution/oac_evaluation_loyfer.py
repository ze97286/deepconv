from deep_conv.deconvolution.deepconv import *
from collections import defaultdict

from pathlib import Path
from deep_conv.benchmark.nnls import run_weighted_nnls
from deep_conv.benchmark.benchmark_utils import *
from scipy import stats

def plot_oac_analysis(df, name):
    # Create figure with custom specs for layout
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
        subplot_titles=('OAC Comparison by Sample', 
                       'ZOHAR_OAC vs ichorCNA', 'BEN_OAC vs ichorCNA'),
        specs=[[{"colspan": 2}, None],
               [{}, {}]]
    )
    
    metrics = ['OAC_nnls', 'OAC_deepconv', 'tf']
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']
    
    # Add bar plots spanning full width
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(name=metric, x=df['sample'], y=df[metric], marker_color=colors[i]),
            row=1, col=1
        )
    
    # Add scatter plots in bottom row
    fig.add_trace(
        go.Scatter(x=df['tf'], y=df['OAC_deepconv'], mode='markers',
                  hovertext=df['sample'], hoverinfo='text+x+y',
                  name='ZOHAR_OAC vs ichorCNA'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['tf'], y=df['OAC_nnls'], mode='markers',
                  hovertext=df['sample'], hoverinfo='text+x+y',
                  name='BEN_OAC vs ichorCNA'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        showlegend=True,
        title_text="OAC Analysis Dashboard",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Sample", row=1, col=1, tickangle=-45)
    fig.update_xaxes(title_text="ichorCNA", row=2, col=1)
    fig.update_xaxes(title_text="ichorCNA", row=2, col=2)
    
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="ZOHAR_OAC", row=2, col=1)
    fig.update_yaxes(title_text="BEN_OAC", row=2, col=2)
    
    # Write to file
    fig.write_html(f"{name}.html")


def create_correlation_plot(df, cell_types, tf_column, output_dir):
    """
    Create a subplot matrix showing correlations between multiple columns and a target column
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    cell_types (list): List of column names to correlate with tf_column
    tf_column (str): Name of the target column for correlation
    
    Returns:
    plotly.graph_objects.Figure: The generated figure
    """
    n = len(cell_types)
    n_rows = (n + 2) // 3  
    n_cols = min(n, 3)     
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=cell_types,
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
        row_heights=[1] * n_rows
    )

    # Add traces for each cell type
    for idx, col in enumerate(cell_types):
        row = idx // 3 + 1
        col_num = idx % 3 + 1
        
        # Get data
        x = df[tf_column]
        y = df[col]
        
        # Calculate correlation coefficient and p-value
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 1:  # Need at least 2 points for correlation
            r, p = stats.pearsonr(x[mask], y[mask])
            r2 = r**2
        else:
            r = r2 = p = np.nan
            
        # Calculate the global range for both x and y
        max_val = max(max(x), max(y))
        min_val = min(min(x), min(y))
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name=col,
                marker=dict(
                    size=6,
                    opacity=0.6
                ),
                showlegend=False
            ),
            row=row,
            col=col_num
        )
        
        # Add y=x line using the global range
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=row,
            col=col_num
        )
        
        # Add statistical annotations
        fig.add_annotation(
            text=f'R² = {r2:.3f}<br>r = {r:.3f}<br>p = {p:.2e}',
            xref=f'x{idx+1}',
            yref=f'y{idx+1}',
            x=min_val + (max_val - min_val) * 0.05,  # 5% from left edge
            y=max_val - (max_val - min_val) * 0.05,  # 5% from top edge
            showarrow=False,
            font=dict(size=10),
            align='left',
            bgcolor='rgba(255, 255, 255, 0.8)'  # Semi-transparent white background
        )
        
        # Update axes to use the same range
        fig.update_xaxes(
            range=[min_val, max_val],
            row=row,
            col=col_num,
            title=tf_column if row == n_rows else None,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgray'
        )
        
        fig.update_yaxes(
            range=[min_val, max_val],
            row=row,
            col=col_num,
            title=col if col_num == 1 else None,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgray'
        )
    
    fig.update_layout(
        height=400 * n_rows,
        width=1200,
        title=f'Correlation with {tf_column}',
        showlegend=False,
        template='plotly_white',
        margin=dict(t=50, r=50, b=50, l=50)
    )
    
    # Force square aspect ratio for all subplots
    for i in range(n * n_cols):
        fig.update_xaxes(scaleanchor=f"y{i+1}", scaleratio=1, row=(i//n_cols)+1, col=(i%n_cols)+1)
    
    fig.write_html(output_dir)


def plot_analysis(df, cell_types, title, clinical_benefit_col=None, cancer_type_col=None):
    """
    Create a combined figure with stacked bar plot, optional annotations, and heatmap
    Args:
        df: pandas DataFrame with 'sample' column and cell types
        cell_types: list of cell type columns
        title: str, title for stacked bar plot
        clinical_benefit_col: str or None, name of column containing clinical benefit data ('Y'/'N')
        cancer_type_col: str or None, name of column containing cancer type data
    Returns:
        plotly Figure object
    """
    # Determine number of rows and heights based on which annotations are present
    n_annotation_rows = sum(x is not None for x in [clinical_benefit_col, cancer_type_col])
    
    if n_annotation_rows == 0:
        # Original two-row layout with large gap
        n_rows = 2
        row_heights = [0.35, 0.65]
        vertical_spacing = 0.15
    else:
        # Put bar plot separate from heatmap section
        n_rows = n_annotation_rows + 2
        row_heights = [0.35] + [0.65/n_rows] * (n_rows-1)
        vertical_spacing = None  # Will set custom spacing
    
    # Create figure with subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=1,
        subplot_titles=(title, *[""] * (n_rows - 1)),
        row_heights=row_heights,
        vertical_spacing=vertical_spacing
    )

    # Get colors for stacked bars
    colors = px.colors.qualitative.Set3[:len(cell_types)]
    if len(cell_types) > 12:
        colors.extend(px.colors.qualitative.Plotly[:(len(cell_types)-12)])

    # Add stacked bar traces
    for cell_type, color in zip(cell_types, colors):
        fig.add_trace(
            go.Bar(
                name=cell_type,
                x=df['sample'],
                y=df[cell_type],
                marker_color=color,
                hovertemplate=f"{cell_type}: %{{y}}<extra></extra>"
            ),
            row=1, col=1
        )

    samples = df['sample'].unique()
    current_row = 2  # Track the current row for adding plots

    # Add clinical benefit annotation if column provided
    if clinical_benefit_col is not None:
        benefit_values = [df[df['sample'] == sample][clinical_benefit_col].iloc[0] for sample in samples]
        benefit_numeric = [1 if x == 'Y' else 0 for x in benefit_values]
        
        benefit_colorscale = [[0, 'rgb(200, 200, 255)'],  # Light blue for 'N'
                             [1, 'rgb(255, 150, 255)']]   # Pink for 'Y'
        
        fig.add_trace(
            go.Heatmap(
                z=[benefit_numeric],
                x=samples,
                y=[''],
                colorscale=benefit_colorscale,
                showscale=False,
                hoverongaps=False,
                hovertemplate='Clinical Benefit: %{text}<extra></extra>',
                text=[[v for v in benefit_values]]
            ),
            row=current_row, col=1
        )
        
        # Add clinical benefit legend
        for benefit, color in zip(['N', 'Y'], ['rgb(200, 200, 255)', 'rgb(255, 150, 255)']):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'Benefit: {benefit}',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        current_row += 1

    # Add cancer type annotation if column provided
    if cancer_type_col is not None:
        cancer_types = [df[df['sample'] == sample][cancer_type_col].iloc[0] for sample in samples]
        unique_types = sorted(list(set(cancer_types)))
        type_to_num = {t: i for i, t in enumerate(unique_types)}
        cancer_numeric = [type_to_num[t] for t in cancer_types]
        
        n_types = len(unique_types)
        cancer_colors = px.colors.qualitative.Set2[:n_types] if n_types <= 8 else px.colors.qualitative.Alphabet[:n_types]
        cancer_colorscale = [[0, cancer_colors[0]], [1, cancer_colors[0]]] if n_types == 1 else \
                           [[i/(n_types-1), color] for i, color in enumerate(cancer_colors)]
        
        fig.add_trace(
            go.Heatmap(
                z=[cancer_numeric],
                x=samples,
                y=[''],
                colorscale=cancer_colorscale,
                showscale=False,
                hoverongaps=False,
                hovertemplate='Cancer Type: %{text}<extra></extra>',
                text=[[t for t in cancer_types]]
            ),
            row=current_row, col=1
        )

        # Add cancer type legend
        for cancer_type, color in zip(unique_types, cancer_colors):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'Cancer: {cancer_type}',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        current_row += 1

    # Create matrix for main heatmap
    matrix = []
    for cell_type in cell_types:
        row = []
        for sample in samples:
            value = df[df['sample'] == sample][cell_type].iloc[0]
            row.append(value)
        matrix.append(row)

    # Custom colorscale for main heatmap
    colors_heatmap = [
        [0, 'rgb(0, 0, 255)'],
        [0.2, 'rgb(115, 155, 255)'],
        [0.4, 'rgb(230, 230, 255)'],
        [0.5, 'rgb(255, 255, 255)'],
        [0.6, 'rgb(255, 230, 230)'],
        [0.8, 'rgb(255, 155, 115)'],
        [1, 'rgb(255, 0, 0)']
    ]

    # Add main heatmap
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=samples,
            y=cell_types,
            colorscale=colors_heatmap,
            zmin=0,
            zmax=1,
            hoverongaps=False,
            hovertemplate='Sample: %{x}<br>Cell Type: %{y}<br>Value: %{z:.3f}<extra></extra>',
            colorbar=dict(
                title='Value',
                x=1.02,
                y=0.25,
                len=0.5
            )
        ),
        row=current_row, col=1
    )

    # Update layout
    fig.update_layout(
        height=1300,
        width=1400,
        showlegend=True,
        barmode='stack',
        plot_bgcolor='white',
        legend=dict(
            x=1.02,
            y=0.9,
            xanchor='left',
            yanchor='top'
        ),
        margin=dict(r=150, t=100, b=50)
    )

    # Set custom spacing between subplots if we have annotations
    if n_annotation_rows > 0:
        # Calculate positions with gap between bar plot and annotations
        # Increased gap to accommodate x-axis labels
        bar_plot_bottom = 0.75  # Bottom of bar plot section (increased to make room for labels)
        heatmap_top = 0.6      # Top of heatmap section (lowered to create more space)
        annotation_height = 0.02  # Height for each annotation bar
        
        # Position bar plot at top with room for labels
        fig.update_yaxes(row=1, col=1, domain=[bar_plot_bottom, 1.0])
        
        # Position heatmap and annotation layers with no gaps between them
        if n_annotation_rows == 2:
            fig.update_yaxes(row=2, col=1, domain=[heatmap_top + annotation_height, heatmap_top + 2*annotation_height])
            fig.update_yaxes(row=3, col=1, domain=[heatmap_top, heatmap_top + annotation_height])
            fig.update_yaxes(row=4, col=1, domain=[0, heatmap_top])
        else:  # 1 annotation
            fig.update_yaxes(row=2, col=1, domain=[heatmap_top, heatmap_top + annotation_height])
            fig.update_yaxes(row=3, col=1, domain=[0, heatmap_top])

    # Update xaxis for all plots
    for row in range(1, n_rows + 1):
        show_labels = (row == 1) or (row == n_rows)  # Only show labels for bar plot and main heatmap
        fig.update_xaxes(
            tickangle=90,
            title='Sample' if row == n_rows else None,
            row=row,
            tickmode='array',
            ticktext=samples if show_labels else [],
            tickvals=list(range(len(samples))),
            dtick=1,
            showticklabels=show_labels
        )

    # Update yaxis
    fig.update_yaxes(title='Proportion', row=1)
    # Remove all traces of axes for annotation bars
    if clinical_benefit_col is not None:
        fig.update_yaxes(showticklabels=False, showline=False, zeroline=False, row=2)
    if cancer_type_col is not None:
        fig.update_yaxes(showticklabels=False, showline=False, zeroline=False, 
                        row=2 + (clinical_benefit_col is not None))
    fig.update_yaxes(title='Cell Type', row=n_rows, autorange='reversed')

    return fig


oac_dilutions = [0.4,0.3,0.25,0.2,0.15,0.10,0.05,0.01,0.005,0.001,0.0001,0.00001]
tcell_dilutions = [0.10,0.05,0.01,0.005,0.001,0.0001,0.00001]

def sample_to_dilution(sample):
    return int(sample.split("_")[1][3:])-1


def prepare_deconv_input(atlas_path, eval_pat_dir, dilutions):
    atlas = pd.read_csv(atlas_path, sep="\t").dropna()
    names = set(atlas.name.unique())
    # load marker values, coverage, and ground truth
    X_val = pd.read_parquet(Path(eval_pat_dir)/"marker_values.parquet")
    coverage_val = pd.read_parquet(Path(eval_pat_dir)/"coverage.parquet")
    y_val = pd.read_parquet(Path(eval_pat_dir)/"ground_truth_y.parquet") 
    y_val['sample'] = list(X_val.columns[2:])
    y_val['dilution'] = y_val['sample'].apply(sample_to_dilution).apply(lambda x: dilutions[x])
    y_dilutions = pd.DataFrame(y_val['dilution'].values, columns=['dilution'])
    y_val = y_val.drop(columns=['dilution','sample']).to_numpy()                   
    X_val = X_val[X_val.name.isin(names)].drop(columns=["name", "direction"]).T.to_numpy()
    coverage_val = coverage_val[coverage_val.name.isin(names)].drop(columns=["name", "direction"]).T.to_numpy()    
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_val = y_val / y_val.sum(dim=1, keepdim=True)
    y_val_df = pd.DataFrame(y_val, columns=list(atlas.columns[8:]))
    return X_val, coverage_val, y_val_df, y_dilutions


def debug_model_predictions(model, X_val, coverage_val, y_true_df, threshold=0.005):
    """
    Debug function to analyze presence predictions and model outputs
    
    Args:
        model: The cell type deconvolution model
        X_val: Input methylation values
        coverage_val: Coverage values
        y_true_df: DataFrame with ground truth
        threshold: Presence threshold for analysis
    """
    # Convert to tensors if needed
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.float32)
    if not isinstance(coverage_val, torch.Tensor):
        coverage_val = torch.tensor(coverage_val, dtype=torch.float32)
    
    y_true = torch.tensor(y_true_df.values, dtype=torch.float32)
    
    device = next(model.parameters()).device
    X_val = X_val.to(device)
    coverage_val = coverage_val.to(device)
    y_true = y_true.to(device)
    
    model.eval()
    with torch.no_grad():
        # Forward pass through model to get all outputs
        props, reconstructed, valid_mask, presence_probs, presence_logits = model(X_val, coverage_val)
        
        # Check true presence (using same threshold as loss_fn)
        true_presence = (y_true > threshold).float()
        pred_presence = (presence_probs > 0.5).float()
        
        print("\n===== MODEL OUTPUT ANALYSIS =====")
        print(f"Using presence threshold: {threshold}")
        
        # Aggregated stats
        print("\n----- OVERALL STATISTICS -----")
        print(f"Proportions: min={props.min().item():.6f}, max={props.max().item():.6f}, mean={props.mean().item():.6f}")
        print(f"Presence probs: min={presence_probs.min().item():.6f}, max={presence_probs.max().item():.6f}, mean={presence_probs.mean().item():.6f}")
        
        # Calculate metrics per cell type
        print("\n----- CELL TYPE PRESENCE DETECTION -----")
        cell_types = y_true_df.columns
        f1_scores = []
        
        for i, cell_type in enumerate(cell_types):
            tp = torch.sum((pred_presence[:, i] == 1) & (true_presence[:, i] == 1)).item()
            fp = torch.sum((pred_presence[:, i] == 1) & (true_presence[:, i] == 0)).item()
            tn = torch.sum((pred_presence[:, i] == 0) & (true_presence[:, i] == 0)).item()
            fn = torch.sum((pred_presence[:, i] == 0) & (true_presence[:, i] == 1)).item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(f1)
            
            true_mean = y_true[:, i].mean().item()
            pred_mean = props[:, i].mean().item()
            
            print(f"{cell_type}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, TP={tp}, FP={fp}, FN={fn}")
            print(f"  Mean: True={true_mean:.6f}, Pred={pred_mean:.6f}, Ratio={pred_mean/max(true_mean, 1e-8):.2f}")
        
        print(f"\nAverage F1 Score: {sum(f1_scores)/len(f1_scores):.4f}")
        
        return props, presence_probs


def deepconv_estimate(atlas_path, eval_pat_dir, dilutions, presence_model_name, model_name, batch_size=256, device=torch.device('cpu')):
    """
    Consistent evaluation function that matches training behavior.
    
    Args:
        atlas_path: Path to the atlas file
        eval_pat_dir: Directory with evaluation data
        model: The model to evaluate
        dilutions: Dilution information
        min_cpgs: Minimum CpGs required
        threads: Number of threads to use
        
    Returns:
        y_true_df, predictions_df, y_dilutions: DataFrames with results
    """
    deepconv_atlas = pd.read_csv(atlas_path, sep="\t")
    cell_types = list(deepconv_atlas.columns[8:])

    target_ids = deepconv_atlas["target"].map(lambda x: cell_types.index(x)).to_numpy()
    # Create model
    model = CellTypeDeconvolutionModel(
        num_markers=len(deepconv_atlas),
        num_cell_types=len(cell_types),
        target_ids=target_ids,
        presence_models_dir=f"/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/{presence_model_name}",
        feature_dim=64,
    )

    atlas_np = deepconv_atlas[deepconv_atlas.columns[8:]].T.to_numpy()
    X_val, coverage_val, y_true_df, y_dilutions = prepare_deconv_input(atlas_path, eval_pat_dir, dilutions)

    num_cell_types = len(cell_types)
    num_samples = len(X_val)
    presence_probs = torch.zeros(num_samples, num_cell_types)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    coverage_val_tensor = torch.tensor(coverage_val, dtype=torch.float32)
    temp_dataset = TensorDataset(X_val_tensor, coverage_val_tensor)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_idx, (batch_fraction, batch_coverage) in enumerate(temp_loader):
            batch_fraction = batch_fraction.to(device)
            batch_coverage = batch_coverage.to(device)
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_presence_probs, _ = model.predict_presence_with_separate_models(batch_fraction, batch_coverage)
            presence_probs[start_idx:end_idx] = batch_presence_probs.cpu()

    presence_mean_before = presence_probs.mean().item()
    presence_std_before = presence_probs.std().item()
    print(f"Presence probs before loading checkpoint: mean={presence_mean_before:.6f}, std={presence_std_before:.6f}")


    # Load checkpoint
    best_model = "best_model.pt"
    checkpoint = torch.load(f"/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/{model_name}/{best_model}")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    commit_hash = checkpoint["commit_hash"] if "commit_hash" in checkpoint else "unknown"
    epoch = checkpoint["epoch"]
    # Print the 'best_threshold' if it exists in the checkpoint
    if 'best_threshold' in checkpoint:
        print(f"Model's best threshold from training: {checkpoint['best_threshold']}")

    presence_probs_post = torch.zeros(num_samples, num_cell_types)
    temp_dataset = TensorDataset(X_val_tensor, coverage_val_tensor)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_idx, (batch_fraction, batch_coverage) in enumerate(temp_loader):
            batch_fraction = batch_fraction.to(device)
            batch_coverage = batch_coverage.to(device)
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_presence_probs, _ = model.predict_presence_with_separate_models(batch_fraction, batch_coverage)
            presence_probs_post[start_idx:end_idx] = batch_presence_probs.cpu()

    presence_mean_after = presence_probs_post.mean().item()
    presence_std_after = presence_probs_post.std().item()
    print(f"Presence probs before loading checkpoint: mean={presence_mean_after:.6f}, std={presence_std_after:.6f}")

    # Prepare data
    predictions = model.predict(X_val, coverage_val, atlas=atlas_np, presence_probs=presence_probs)
    predictions_df = pd.DataFrame(predictions, columns=list(y_true_df.columns))

    # Log summary statistics
    print("\n===== PREDICTION SUMMARY =====")
    print(f"Predictions mean: {predictions.mean():.6f}")
    print(f"Max prediction: {predictions.max():.6f}")
    print(f"Min prediction: {predictions.min():.6f}")

    return y_true_df, predictions_df, y_dilutions, commit_hash, epoch 


def eval_OAC(atlas_path, pat_dir, title, prefix, atlas_name, batch, model, type, out_dir,cd_tissue_mapping, model_name=None,ichorCNA=None, clinical_benefit=None, cancer_type=None, presence_model_name=None, tcell_col="T-cells"):
    pat_dir = Path(pat_dir)
    atlas = pd.read_csv(atlas_path,sep="\t")
    X_val = pd.read_parquet(Path(pat_dir)/"marker_values.parquet")
    coverage_val = pd.read_parquet(Path(pat_dir)/"coverage.parquet")
    samples = list(X_val.columns[2:])
    atlas = atlas.dropna()
    names = set(atlas.name.unique()) 
    X_val = X_val[X_val.name.isin(names)]
    coverage_val = coverage_val[coverage_val.name.isin(names)]
    X_val = X_val.drop(columns=["name", "direction"]).T.to_numpy()
    coverage_val = coverage_val.drop(columns=["name", "direction"]).T.to_numpy()

    if model_name is not None:
        cell_types = list(atlas.columns[8:])
        target_ids = atlas["target"].map(lambda x: cell_types.index(x)).to_numpy()
        model = CellTypeDeconvolutionModel(
            num_markers=len(atlas),
            num_cell_types=len(cell_types), 
            target_ids=target_ids, 
            presence_models_dir=f"/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/{presence_model_name}",
            feature_dim=64,
        )
        checkpoint = torch.load(f"/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/{model_name}/best_model.pt")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        atlas_np = atlas[atlas.columns[8:]].T.to_numpy()
        estimation = model.predict(X_val,coverage_val, atlas=atlas_np)
    else:
        estimation = run_weighted_nnls(X_val, coverage_val, atlas[atlas.columns[8:]].T.values)
    df = pd.DataFrame(estimation, columns=list(atlas.columns[8:]))
    cols = sorted(list(df.columns))
    df['sample'] = samples    
    def extract_sample(sample):
        sample=sample.split("_plasma")[0]
        sample=sample.split("_md")[0]
        return sample
    def map_sample_index_to_name(sample):
        index = sample.split("-")[-1]
        name=cd_tissue_mapping[index]
        return name

    df['sample'] = df['sample'].apply(extract_sample)
    if cd_tissue_mapping is not None:
        df['sample'] = df['sample'].apply(map_sample_index_to_name)
    df['atlas_name'] = atlas_name
    df['batch'] = batch
    df['model'] = model 
    df['type'] = type
    out_dir = Path(out_dir)
    df.to_csv(out_dir/f"{prefix}_deconvolution.csv", sep="\t", index=False)

    clinical_benefit_col = None
    if clinical_benefit is not None:
        df['clinical_benefit'] = df['sample'].apply(lambda x: clinical_benefit[x.split('_')[0]])
        clinical_benefit_col = 'clinical_benefit'

    cancer_type_col = None
    if cancer_type is not None:
        df['cancer_type'] = df['sample'].apply(lambda x: cancer_type[x.split('_')[0]])
        cancer_type_col = 'cancer_type'

    def remove_clinical_benefit_cancer_type_cols(filtered_cols):
        if not 'clinical_benefit' in df.columns:
            filtered_cols.remove("clinical_benefit")
        if not 'cancer_type' in df.columns: 
            filtered_cols.remove("cancer_type")
        return filtered_cols

    filtered_cols_all = remove_clinical_benefit_cancer_type_cols(cols+['sample','clinical_benefit','cancer_type'])
    fig = plot_analysis(df[filtered_cols_all], cols, title, clinical_benefit_col, cancer_type_col)
    fig.write_html(out_dir/f"{prefix}_deconvolution.html")

    if ichorCNA is not None:
        merged_df_with_tf = df.merge(ichorCNA,on="sample", how="outer").dropna() 
        if len(merged_df_with_tf)>0:
            create_correlation_plot(merged_df_with_tf, cols, "tf", out_dir/f"{prefix}_cell_type_vs_oac_correlation.html")

    # plot T-cells concentrations
    filtered_cols_tcells = remove_clinical_benefit_cancer_type_cols(['sample',tcell_col,'clinical_benefit','cancer_type'])
    fig = plot_analysis(df[filtered_cols_tcells], [tcell_col], title, clinical_benefit_col, cancer_type_col)
    fig.write_html(out_dir/f"{prefix}_tcells_deconvolution.html")

    # plot baseline concentrations
    baseline = df[df['sample'].str.contains("ScrBsl")]
    fig = plot_analysis(baseline[filtered_cols_all], cols, "Baseline: "+title, clinical_benefit_col, cancer_type_col)
    fig.write_html(out_dir/f"{prefix}_ScrBsl_deconvolution.html")

    # plot baseline T-cells concentrations
    fig = plot_analysis(baseline[filtered_cols_tcells], [tcell_col], "Baseline T-Cells: "+title, clinical_benefit_col, cancer_type_col)
    fig.write_html(out_dir/f"{prefix}_ScrBsl_tcells_deconvolution.html")

    if ichorCNA is not None:
        merged_df_with_tf = baseline.merge(ichorCNA,on="sample", how="outer").dropna() 
        if len(merged_df_with_tf)>0:
            create_correlation_plot(merged_df_with_tf, cols, "tf", out_dir/f"{prefix}_baseline_cell_type_vs_oac_correlation.html")

    # plot immonly concentrations
    immonly = df[df['sample'].str.contains("Immonly")]
    fig = plot_analysis(immonly[filtered_cols_all], cols, "ImmOnly: "+title, clinical_benefit_col, cancer_type_col)
    fig.write_html(out_dir/f"{prefix}_immonly_deconvolution.html")

    # plot immonly T-cells concentrations
    fig = plot_analysis(immonly[filtered_cols_tcells], [tcell_col], "Immonly T-Cells: "+title, clinical_benefit_col, cancer_type_col)
    fig.write_html(out_dir/f"{prefix}_immonly_tcells_deconvolution.html")

    if ichorCNA is not None:
        merged_df_with_tf = immonly.merge(ichorCNA,on="sample", how="outer").dropna() 
        if len(merged_df_with_tf)>0:
            create_correlation_plot(merged_df_with_tf, cols, "tf", out_dir/f"{prefix}_immonly_cell_type_vs_oac_correlation.html")

    # plot surgery concentrations
    surg = df[df['sample'].str.contains("Surg")]
    if len(surg)>0:
        fig = plot_analysis(surg[filtered_cols_all], cols, "Surgery: "+title, clinical_benefit_col, cancer_type_col)
        fig.write_html(out_dir/f"{prefix}_surg_deconvolution.html")
        # plot C1 T-cells concentrations
        fig = plot_analysis(surg[filtered_cols_tcells], [tcell_col], "Surgery T-Cells: "+title, clinical_benefit_col, cancer_type_col)
        fig.write_html(out_dir/f"{prefix}_surg_tcells_deconvolution.html")
        if ichorCNA is not None:
            merged_df_with_tf = surg.merge(ichorCNA,on="sample", how="outer").dropna() 
            if len(merged_df_with_tf)>0:
                create_correlation_plot(merged_df_with_tf, cols, "tf", out_dir/f"{prefix}_surg_cell_type_vs_oac_correlation.html")

    # plot C1 concentrations
    c1 = df[df['sample'].str.contains("C1")]
    if len(c1)>0:
        fig = plot_analysis(c1[filtered_cols_all], cols, "C1: "+title, clinical_benefit_col, cancer_type_col)
        fig.write_html(out_dir/f"{prefix}_c1_deconvolution.html")
        # plot C1 T-cells concentrations
        fig = plot_analysis(c1[filtered_cols_tcells], [tcell_col], "C1 T-Cells: "+title, clinical_benefit_col, cancer_type_col)
        fig.write_html(out_dir/f"{prefix}_c1_tcells_deconvolution.html")
        if ichorCNA is not None:
            merged_df_with_tf = c1.merge(ichorCNA,on="sample", how="outer").dropna() 
            if len(merged_df_with_tf)>0:
                create_correlation_plot(merged_df_with_tf, cols, "tf", out_dir/f"{prefix}_c1_cell_type_vs_oac_correlation.html")

    # plot C6 concentrations
    c6 = df[df['sample'].str.contains("C6")]
    if len(c6)>0:
        fig = plot_analysis(c6[filtered_cols_all], cols, "C6: "+title, clinical_benefit_col, cancer_type_col)
        fig.write_html(out_dir/f"{prefix}_c6_deconvolution.html")

        # plot C6 T-cells concentrations
        fig = plot_analysis(c6[filtered_cols_tcells], [tcell_col], "C6 T-Cells: "+title, clinical_benefit_col, cancer_type_col)
        fig.write_html(out_dir/f"{prefix}_c6_tcells_deconvolution.html")

        if ichorCNA is not None:
            merged_df_with_tf = c6.merge(ichorCNA,on="sample", how="outer").dropna() 
            if len(merged_df_with_tf)>0:
                create_correlation_plot(merged_df_with_tf, cols, "tf", out_dir/f"{prefix}_c6_cell_type_vs_oac_correlation.html")

    # plot post treatment concentrations
    pt = df[df['sample'].str.contains("PT")]
    if len(pt)>0:
        fig = plot_analysis(pt[filtered_cols_all], cols, "PT: "+title, clinical_benefit_col, cancer_type_col)
        fig.write_html(out_dir/f"{prefix}_pt_deconvolution.html")

        # plot pt T-cells concentrations
        fig = plot_analysis(pt[filtered_cols_tcells], [tcell_col], "PT T-Cells: "+title, clinical_benefit_col, cancer_type_col)
        fig.write_html(out_dir/f"{prefix}_pt_tcells_deconvolution.html")

        if ichorCNA is not None:
            merged_df_with_tf = pt.merge(ichorCNA,on="sample", how="outer").dropna() 
            if len(merged_df_with_tf)>0:
                create_correlation_plot(merged_df_with_tf, cols, "tf", out_dir/f"{prefix}_pt_cell_type_vs_oac_correlation.html")

    # plot control concentrations
    controls = df[(df['sample'].str.contains("X")) | df['sample'].str.contains("TP") | df['sample'].str.contains("GI") | df['sample'].str.contains("SCAN")]
    if len(controls)>0:
        fig = plot_analysis(controls[cols+['sample']], cols, "Controls: "+title)
        fig.write_html(out_dir/f"{prefix}_control_deconvolution.html")

        # plot controls T-cells concentrations
        fig = plot_analysis(controls[['sample',tcell_col]], [tcell_col], "Control T-Cells: "+title)
        fig.write_html(out_dir/f"{prefix}_control_tcells_deconvolution.html")


# 1
def train_and_evaluate(model_name, presence_model_name):
    atlas_path = "/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed"
    train_pat_dir = "/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train"
    eval_pat_dir = "/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval"

    threads = 32
    output_path = Path("/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/"+model_name+"/")
    presence_path = "/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/"+presence_model_name+"/"
    train_and_eval(atlas_path=atlas_path, 
                           train_pat_dir=train_pat_dir, 
                           eval_pat_dir=eval_pat_dir, 
                           threads=threads,
                           output_path=output_path, 
                           presence_models_dir=presence_path)


def nnls_estimate(atlas_path, eval_pat_dir, dilutions):
    atlas = pd.read_csv(atlas_path,sep="\t").dropna()
    marker_read_proportions, counts, y_true_df, y_dilutions = prepare_deconv_input(atlas_path, eval_pat_dir, dilutions)
    print("median coverage", np.median(counts, axis=1), np.median(np.median(counts, axis=1)), np.median(counts, axis=1).mean())
    marker_read_proportions, counts = marker_read_proportions, counts
    predictions = run_weighted_nnls(marker_read_proportions, counts, atlas[atlas.columns[8:]].T.values)
    predictions_df = pd.DataFrame(predictions, columns=list(y_true_df.columns))
    return y_true_df, predictions_df, y_dilutions


def get_git_commit() -> Dict[str, str]:
    """Get git repository information"""
    import subprocess

    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).strip().decode('utf-8')
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).strip().decode('utf-8')
        
        status = subprocess.check_output(
            ['git', 'status', '--porcelain']
        ).strip().decode('utf-8')
        
       
        return commit_hash
    except subprocess.CalledProcessError:
        return 'unknown'


def eval_admixtures_nnls(atlas_path, size="low"):
    suffix = f"_{size}/"

    commit_hash = get_git_commit()
    
    pat_dir_tcells = f"/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval{suffix}T-cells/"
    pat_dir_tcells_with_heart = f"/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval{suffix}heart/"
    pat_dir_oac = f"/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval{suffix}OAC/"

    y_true_df, predictions_df, y_dilutions= nnls_estimate(atlas_path, pat_dir_tcells, tcell_dilutions)
    plot_deconvolution_evaluation(y_true_df, predictions_df, y_dilutions['dilution'], pat_dir_tcells+"nnls/", commit_hash)
    y_true_df, predictions_df, y_dilutions = nnls_estimate(atlas_path, pat_dir_oac, oac_dilutions)
    plot_deconvolution_evaluation(y_true_df, predictions_df, y_dilutions['dilution'], pat_dir_oac+"nnls/", commit_hash)
    y_true_df, predictions_df, y_dilutions = nnls_estimate(atlas_path, pat_dir_tcells_with_heart, tcell_dilutions)
    plot_deconvolution_evaluation(y_true_df, predictions_df, y_dilutions['dilution'], pat_dir_tcells_with_heart+"nnls/", commit_hash)


def eval_admixtures(model_name,presence_model_name, size="low"):
    # evaluate Zohar's atlas with deepconv
    eval_admixtures_deepconv(model_name, presence_model_name, size)  
    # evaluate Zohar's atlas (primary markers only) with nnls
    eval_admixtures_nnls("/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed", size)


# 2
def eval_admixtures_deepconv(model_name, presence_model_name, size="low"):
    """
    Evaluate deepconv model with consistent parameters.
    
    Args:
        model_name: Name of the model directory
        presence_model_name: Name of the presence model directory
        use_low_depth: Whether to use low depth data
    """
    suffix = f"_{size}/"

    deepconv_atlas_path = "/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed"
    

    # Evaluation paths
    deepconv_eval_pat_dir_tcells = f"/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval{suffix}T-cells/"
    deepconv_eval_pat_dir_oac = f"/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval{suffix}OAC/"
    deepconv_eval_pat_dir_tcells_with_heart = f"/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval{suffix}heart/"

    # Run evaluations
    print("\n===== EVALUATING T-CELLS =====")
    y_true_df, predictions_df, y_dilutions, commit_hash, epoch = deepconv_estimate(
        deepconv_atlas_path, deepconv_eval_pat_dir_tcells, tcell_dilutions, presence_model_name, model_name
    )
    plot_deconvolution_evaluation(
        y_true_df, predictions_df, y_dilutions['dilution'], 
        deepconv_eval_pat_dir_tcells+f"{model_name}/",
        commit_hash,
        epoch=epoch,
    )

    print("\n===== EVALUATING OAC =====")
    y_true_df, predictions_df, y_dilutions, commit_hash, epoch = deepconv_estimate(
        deepconv_atlas_path, deepconv_eval_pat_dir_oac, oac_dilutions, presence_model_name, model_name
    )
    plot_deconvolution_evaluation(
        y_true_df, predictions_df, y_dilutions['dilution'], 
        deepconv_eval_pat_dir_oac+f"{model_name}/",
        commit_hash,
        epoch=epoch,
    )

    print("\n===== EVALUATING heart =====")
    y_true_df, predictions_df, y_dilutions, commit_hash, epoch = deepconv_estimate(
        deepconv_atlas_path, deepconv_eval_pat_dir_tcells_with_heart, tcell_dilutions, presence_model_name, model_name
    )
    plot_deconvolution_evaluation(
        y_true_df, predictions_df, y_dilutions['dilution'], 
        deepconv_eval_pat_dir_tcells_with_heart+f"{model_name}/",
        commit_hash,
        epoch=epoch,
    )

# 3
def run_oac_analysis(model_name, presence_model_name):
    out_base_dir = "/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis"
    ichorcna_cf_ab = pd.read_csv(out_base_dir+"/AB/cfDNA/ab_ichorcna_cfdna.csv", sep="\t")
    ichorcna_cf_ab.columns=['sample', 'tf','ploidy']
    ichorcna_cf_cd = pd.read_csv(out_base_dir+"/CD/cfDNA/cd_ichorcna_cfdna.csv", sep="\t")
    ichorcna_cf_cd.columns=['sample', 'tf','ploidy']
  
    ab_metadata = pd.read_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/AB_patient_summary_HannahFuchs2023.csv")
    def subject_to_sample(subject):
        split = subject.split("-")
        return split[0]+"-"+split[1]
    ab_metadata['sample']=ab_metadata['subject'].apply(subject_to_sample)
    ab_metadata['cancer_type']=ab_metadata['subject_recode'].map(lambda x:x.split('-')[0])
    ab_sample_to_cb =  defaultdict(
        lambda: 'NA', 
        zip(ab_metadata['sample'], ab_metadata['Clinical_Benefit'])
    )
    
    ab_sample_to_ct =  defaultdict(
        lambda: 'NA', 
        zip(ab_metadata['sample'], ab_metadata['cancer_type'])
    )
    
    cd_metadata = pd.read_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/CD/cfDNA/CD_patient_summary_HannahFuchs2023.csv")    
    cd_metadata['sample']=cd_metadata['subject'].apply(subject_to_sample)
    cd_metadata['cancer_type']=cd_metadata['subject_recode'].map(lambda x:x.split('-')[0])
    cd_sample_to_ct =  defaultdict(
        lambda: 'NA', 
        zip(cd_metadata['sample'], cd_metadata['cancer_type'])
    )

    # cd_tissue_mapping_table = pd.read_csv(out_dir+"/cd_tissue_metadata.tsv",sep="\t")
    # cd_tissue_mapping = dict(zip(cd_tissue_mapping_table['sample_index'].values, cd_tissue_mapping_table['sample'].values))
    cd_tissue_mapping = None
    none_tissue_mapping = None

    zohar_model = model_name
    zohar_model_name = model_name
    zohar_atlas_path = "/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed"

    # tissue deep conv AB
    zohar_atlas_name = "atlas_oac.blood+gi+tum.l4"
    zohar_batch="AB"
    zohar_type = "tissue"
    zohar_prefix_ab_tissue = "deep_conv_ab_tissue"
    zohar_pat_dir_ab_tissue = "/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/AB/tissue"
    zohar_title_ab_tissue=f"DeepConv deconvolution using atlas {zohar_atlas_path} on AB tissue"
    out_dir = str(Path(out_base_dir)/"AB"/"tissue"/model_name)
    # eval_OAC(zohar_atlas_path, zohar_pat_dir_ab_tissue, zohar_title_ab_tissue, zohar_prefix_ab_tissue, zohar_atlas_name, zohar_batch, zohar_model, zohar_type, out_dir, none_tissue_mapping, zohar_model_name)

    zohar_type = "cfDNA"
    zohar_prefix_ab_cf = "deep_conv_ab_cfDNA"
    zohar_pat_dir_ab_cf = "/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/AB/cfDNA"
    zohar_title_ab_cf=f"DeepConv deconvolution using atlas {zohar_atlas_path} on AB cfDNA"
    out_dir = str(Path(out_base_dir)/"AB"/"cfDNA"/model_name)
    os.makedirs(out_dir, exist_ok=True)
    eval_OAC(zohar_atlas_path, zohar_pat_dir_ab_cf, zohar_title_ab_cf, zohar_prefix_ab_cf, zohar_atlas_name, zohar_batch, zohar_model, zohar_type, out_dir, none_tissue_mapping, zohar_model_name, ichorcna_cf_ab, ab_sample_to_cb, ab_sample_to_ct, presence_model_name=presence_model_name)

    zohar_batch="CD"
    zohar_prefix_cd_tissue = "deep_conv_cd_tissue"
    zohar_pat_dir_cd_tissue = "/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/CD/tissue"
    zohar_title_cd_tissue=f"DeepConv deconvolution using atlas {zohar_atlas_path} on CD tissue"
    out_dir = str(Path(out_base_dir)/"CD"/"tissue"/model_name)
    # eval_OAC(zohar_atlas_path, zohar_pat_dir_cd_tissue, zohar_title_cd_tissue, zohar_prefix_cd_tissue, zohar_atlas_name, zohar_batch, zohar_model, zohar_type, out_dir, cd_tissue_mapping, zohar_model_name, ichorcna_cf_cd)

    zohar_type = "cfDNA"
    zohar_prefix_cd_cf = "deep_conv_cd_cfDNA"
    zohar_pat_dir_cd_cf = "/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/CD/cfDNA"
    zohar_title_cd_cf=f"DeepConv deconvolution using atlas {zohar_atlas_path} on CD cfDNA"
    out_dir = str(Path(out_base_dir)/"CD"/"cfDNA"/model_name)
    os.makedirs(out_dir, exist_ok=True)
    eval_OAC(zohar_atlas_path, zohar_pat_dir_cd_cf, zohar_title_cd_cf, zohar_prefix_cd_cf, zohar_atlas_name, zohar_batch, zohar_model, zohar_type, out_dir, none_tissue_mapping, zohar_model_name, presence_model_name=presence_model_name)

    ben_model_name = None
    ben_atlas_name = "atlas_dmr_by_read.blood+gi+tum.U100.l4"
    ben_atlas_path = "/users/zetzioni/sharedscratch/atlas/atlas/atlas_dmr_by_read.blood+gi+tum.U100.l4.bed"

    ben_model = "nnls"
    ben_batch="AB"
    ben_type = "tissue"
    ben_prefix_ab_tissue = "nnls_ab_tissue"
    ben_pat_dir_ab_tissue = "/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_fixed_dmr_by_read.blood+gi+tum.U100.l4/AB/tissue"                            
    ben_title_ab_tissue = f"NNLS deconvolution using atlas {ben_atlas_path} on AB tissue"
    out_dir = str(Path(out_base_dir)/"AB"/"tissue"/"nnls")

    # eval_OAC(ben_atlas_path, ben_pat_dir_ab_tissue, ben_title_ab_tissue, ben_prefix_ab_tissue, ben_atlas_name, ben_batch, ben_model, ben_type, out_dir, none_tissue_mapping, ben_model_name)

    ben_type = "cfDNA"
    ben_prefix_ab_cf = "nnls_ab_cfDNA"
    ben_pat_dir_ab_cf = "/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_fixed_dmr_by_read.blood+gi+tum.U100.l4/AB/cfDNA/"
    ben_title_ab_cf = f"NNLS deconvolution using atlas {ben_atlas_path} on AB cfDNA"
    out_dir = str(Path(out_base_dir)/"AB"/"cfDNA"/"nnls")
    eval_OAC(ben_atlas_path, ben_pat_dir_ab_cf, ben_title_ab_cf, ben_prefix_ab_cf, ben_atlas_name, ben_batch, ben_model, ben_type, out_dir, none_tissue_mapping, ben_model_name, ichorcna_cf_ab, ab_sample_to_cb, ab_sample_to_ct, tcell_col="CD4-T-cells")

    ben_batch="CD"
    ben_type = "tissue"
    ben_prefix_cd_tissue = "nnls_cd_tissue"
    ben_pat_dir_cd_tissue = "/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_fixed_dmr_by_read.blood+gi+tum.U100.l4/CD/tissue/"
    ben_title_cd_tissue = f"NNLS deconvolution using atlas {ben_atlas_path} on CD tissue"
    out_dir = str(Path(out_base_dir)/"CD"/"tissue"/"nnls")

    # eval_OAC(ben_atlas_path, ben_pat_dir_cd_tissue, ben_title_cd_tissue, ben_prefix_cd_tissue, ben_atlas_name, ben_batch, ben_model, ben_type, out_dir, cd_tissue_mapping, ben_model_name)

    ben_type = "cfDNA"
    ben_prefix_cd_cf= "nnls_cd_cfDNA"
    ben_pat_dir_cd_cf = "/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_fixed_dmr_by_read.blood+gi+tum.U100.l4/CD/cfDNA"
    ben_title_cd_cf = f"NNLS deconvolution using atlas {ben_atlas_path} on CD cfDNA"
    out_dir = str(Path(out_base_dir)/"CD"/"cfDNA"/"nnls")

    eval_OAC(ben_atlas_path, ben_pat_dir_cd_cf, ben_title_cd_cf, ben_prefix_cd_cf, ben_atlas_name, ben_batch, ben_model, ben_type, out_dir, none_tissue_mapping, ben_model_name, ichorcna_cf_cd, None, cd_sample_to_ct, tcell_col="CD4-T-cells")

    # plot AB cohort cfDNA Ben's Atlas with NNLS vs Deepcon with Zohar's atlas OAC concentration vs ichorCNA
    ben_cf_ab = pd.read_csv(out_base_dir+"/AB/cfDNA/nnls/nnls_ab_cfDNA_deconvolution.csv",sep="\t")
    zohar_cf_ab = pd.read_csv(out_base_dir+"/AB/cfDNA/deepconv/deep_conv_ab_cfDNA_deconvolution.csv",sep="\t")
    merged_cf_ab_oac = zohar_cf_ab.merge(ben_cf_ab, suffixes=('_deepconv','_nnls'), on='sample')[['sample','OAC_deepconv', 'OAC_nnls']]
    merged_cf_ab_oac = merged_cf_ab_oac.merge(ichorcna_cf_ab,on="sample", how="outer").dropna()
    merged_cf_ab_oac.to_csv(out_base_dir+"/AB/cfDNA/ab_cf_vs_ichorcna.csv",sep="\t",index=False)
    name = out_base_dir+"/AB/cfDNA/ab_cf_vs_ichorcna"
    plot_oac_analysis(merged_cf_ab_oac, name)

    # plot CD cohort cfDNA Ben's Atlas with NNLS vs Deepcon with Zohar's atlas OAC concentration vs ichorCNA
    ben_cf_cd = pd.read_csv(out_base_dir+"/CD/cfDNA//nnls/nnls_cd_cfDNA_deconvolution.csv",sep="\t")
    zohar_cf_cd = pd.read_csv(out_base_dir+"/CD/cfDNA//deepconv/deep_conv_cd_cfDNA_deconvolution.csv",sep="\t")
    merged_cf_cd_oac = zohar_cf_cd.merge(ben_cf_cd, suffixes=('_deepconv','_nnls'), on='sample')[['sample','OAC_deepconv', 'OAC_nnls']]
    merged_cf_cd_oac = merged_cf_cd_oac.merge(ichorcna_cf_cd,on="sample", how="outer").dropna()
    merged_cf_cd_oac.to_csv(out_base_dir+"/CD/cfDNA/cd_cf_vs_ichorcna.csv",sep="\t",index=False)
    name = out_base_dir+"/CD/cfDNA/cd_cf_vs_ichorcna"
    plot_oac_analysis(merged_cf_cd_oac, name)
