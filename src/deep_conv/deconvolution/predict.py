import torch
import pandas as pd
from deep_conv.deconvolution.deconv import *
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def calculate_dilution_metrics(results_df):
    metrics = []
    overall_r2 = r2_score(
        results_df['dilution'],
        results_df['contribution'] / results_df['contribution'].max() * results_df['dilution'].max()
    )
    for dilution in sorted(results_df['dilution'].unique()):
        dilution_results = results_df[results_df['dilution'] == dilution]
        y_pred = dilution_results['contribution'].values
        mse = mean_squared_error(
            dilution * np.ones(len(y_pred)), 
            y_pred
        )
        rmse = np.sqrt(mse)
        rel_error = np.mean(np.abs(y_pred - dilution) / dilution)
        expected_ratio = dilution / results_df['dilution'].max()
        actual_ratio = np.mean(y_pred) / np.mean(results_df['contribution'].max())
        ratio_error = abs(expected_ratio - actual_ratio) / expected_ratio
        metrics.append({
            'dilution': dilution,
            'mse': mse,
            'rmse': rmse,
            'ratio_error': ratio_error,
            'relative_error': rel_error,
            'mean_pred': np.mean(y_pred),
            'std_pred': np.std(y_pred),
            'n_samples': len(dilution_results)
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df['overall_r2'] = overall_r2
    return metrics_df


def predict_cell_type_contribution(model_path, marker_read_proportions, marker_read_coverage, cell_type, atlas, device=None, debug=True):
   """
   Predict cell type contributions for all samples.
   Returns both specific cell type results and full predictions matrix.

   Args:
       model_path: Path to saved model
       marker_read_proportions: DataFrame with read proportions values
       marker_read_coverage: DataFrame with coverage values 
       cell_type: Cell type to extract specific results for
       atlas: Atlas DataFrame
       device: Compute device
       debug: Print debug info

   Returns:
       results_df: DataFrame with dilution and contribution for specified cell type
       all_predictions_df: DataFrame with predictions for all cell types (samples x cell types)
   """
   if device is None:
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   if debug:
       print(f"Using device: {device}")
       print(f"Input shapes - marker_read_proportions: {marker_read_proportions.shape}, marker_read_coverage: {marker_read_coverage.shape}, atlas: {atlas.shape}")

   # Ensure marker_read_proportions and marker_read_coverage are properly formatted
   methylation_cols = [col for col in marker_read_proportions.columns if col not in ['name', 'direction']]
   marker_read_proportions_data = marker_read_proportions[methylation_cols]
   marker_read_coverage_data = marker_read_coverage[methylation_cols]

   # Handle NaN values by filling with 0
   marker_read_proportions_data = marker_read_proportions_data.fillna(0)

   n_regions = len(marker_read_proportions)
   n_cell_types = len(atlas.columns[8:])

   if debug:
       print(f"Model parameters - n_regions: {n_regions}, n_cell_types: {n_cell_types}")

   # Initialize model
   model = DeconvolutionModel(n_regions=n_regions, n_cell_types=n_cell_types)
   model.load_state_dict(torch.load(model_path, map_location=device))
   model = model.to(device)
   model.eval()

   # Get cell types and validate requested cell type
   cell_types = atlas.columns[8:].tolist()
   if cell_type not in cell_types:
       raise ValueError(f"Cell type {cell_type} not found in atlas. Available types: {cell_types}")
   cell_type_idx = cell_types.index(cell_type)

   if debug:
       print(f"Cell type {cell_type} found at index {cell_type_idx}")
       print(f"Number of samples to process: {len(methylation_cols)}")
       print(f"First few samples: {methylation_cols[:5]}")

   results = []
   all_predictions = []

   # Process each sample
   for sample in methylation_cols:
       methylation_values = marker_read_proportions_data[sample].values
       coverage_values = marker_read_coverage_data[sample].values

       if debug and len(results) == 0:
           print(f"\nFirst sample stats:")
           print(f"Methylation values range: [{methylation_values.min()}, {methylation_values.max()}]")
           print(f"Coverage values range: [{coverage_values.min()}, {coverage_values.max()}]")

       methylation_input = torch.FloatTensor(methylation_values).unsqueeze(0).to(device)
       coverage_input = torch.FloatTensor(coverage_values).unsqueeze(0).to(device)

       with torch.no_grad():
           proportions = model(methylation_input, coverage_input)
           if debug and len(results) == 0:
               print(f"Model output shape: {proportions.shape}")
               print(f"Model output range: [{proportions.min().item()}, {proportions.max().item()}]")
               print(f"All proportions: {proportions.cpu().numpy()[0]}")

           all_props = proportions.cpu().numpy()[0]
           contribution = all_props[cell_type_idx]

           results.append({
               'sample': sample,
               'dilution': extract_dilution(sample),
               'contribution': contribution
           })
           all_predictions.append(all_props)

   # Create results DataFrame for specified cell type
   results_df = pd.DataFrame(results)
   if debug:
       print("\nFinal results stats:")
       print(results_df.describe())

   results_df['sample_idx'] = results_df['sample'].str.split('_').str[1].astype(int)
   results_df = results_df.sort_values(['dilution', 'sample_idx'])
   del results_df['sample_idx']

   # Create DataFrame with all predictions
   all_predictions_df = pd.DataFrame(
       all_predictions,
       columns=cell_types,
       index=methylation_cols
   )

   return results_df, all_predictions_df


def extract_dilution(sample_name):
    try:
        dilution_str = sample_name.split('_')[0][1:]
        if dilution_str == '1e-4':
            return 0.0001
        return float(dilution_str)
    except:
        return None
    

def plot_dilution_results(results_df, cell_type, all_predictions_df, output_path, show_metrics=True):
    # Calculate metrics
    metrics_df = calculate_dilution_metrics(results_df)
    # Calculate mean and std per dilution
    summary = results_df.groupby('dilution').agg({'contribution': ['mean', 'std']}).reset_index()
    summary.columns = ['dilution', 'mean', 'std']
    # Ensure all cell types are included
    all_cell_types = all_predictions_df.columns.tolist()
    # Calculate the mean proportion for each cell type
    cell_type_means = all_predictions_df.mean(axis=0)
    # Sort the cell types by descending mean proportion (for red at top, blue at bottom)
    sorted_cell_types = cell_type_means.sort_values(ascending=False).index.tolist()  # Corrected to descending
    # Reorder the data accordingly
    sorted_heatmap_data = all_predictions_df[sorted_cell_types]
    # Create figure with subplots
    violin_titles = ['CD34-erythroblasts', 'CD34-megakaryocytes', 'Monocytes', 'Neutrophils']
    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Mean Contribution vs Dilution',    # 1
            'Performance Metrics',              # 2
            'Cell Type Proportions Heatmap',    # 3
            'CD34-erythroblasts',              # 7
            'CD34-megakaryocytes',             # 8
            'Monocytes',                       # 9
            'Neutrophils'                      # 10
        ),
        specs=[
            [{"type": "scatter"}, {"type": "table"}],        # row 1
            [{"type": "heatmap", "colspan": 2}, None],       # row 2
            [{"type": "violin"}, {"type": "violin"}],        # row 3
            [{"type": "violin"}, {"type": "violin"}]         # row 4
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        row_heights=[0.25, 0.35, 0.2, 0.2]
    )
    # Plot 1: Error bar plot
    fig.add_trace(
        go.Scatter(
            x=summary['dilution'],
            y=summary['mean'],
            error_y=dict(type='data', array=summary['std'], visible=True),
            mode='markers+lines',
            name=cell_type
        ),
        row=1, col=1
    )
    fig.update_xaxes(type="log", title="Dilution", row=1, col=1)
    fig.update_yaxes(type="log", title="Mean Fraction", row=1, col=1)
    # Plot 2: Metrics table
    if show_metrics:
        metrics_display = metrics_df.round(4)
        fig.add_trace(
            go.Table(
                header=dict(values=list(metrics_display.columns), align='left'),
                cells=dict(values=[metrics_display[col] for col in metrics_display.columns], align='left')
            ),
            row=1, col=2
        )
    # Plot 3: Heatmap with sorted cell types
    fig.add_trace(
        go.Heatmap(
            z=sorted_heatmap_data.T.values,  # Transposed sorted data
            x=sorted_heatmap_data.index,     # Sample indices
            y=sorted_cell_types,             # Sorted cell types (high to low)
            colorscale='RdBu_r',             # Red for high, blue for low
            colorbar=dict(title='Proportion'),
            zmin=0,
            zmax=0.4
        ),
        row=2, col=1
    )
    fig.update_xaxes(showticklabels=False, title="Samples", row=2, col=1)
    fig.update_yaxes(title="Cell Types", row=2, col=1)
    # Plot 4: Violin plots
    violin_data = all_predictions_df.copy()
    violin_data['dilution'] = results_df['dilution'].values  # Add dilution column
    violin_data = violin_data[violin_titles + ['dilution']]
    # Explicitly map violin titles to specific subplot locations
    violin_titles_mapping = {
        (3, 1): 'CD34-erythroblasts',
        (3, 2): 'CD34-megakaryocytes',
        (4, 1): 'Monocytes',
        (4, 2): 'Neutrophils'
    }
    # Ensure data matches titles
    print("Violin Data Columns:", violin_data.columns.tolist())
    print("Violin Titles Expected:", list(violin_titles_mapping.values()))
    violin_positions = [(3,1), (3,2), (4,1), (4,2)]
    cell_types_config = {
        'CD34-erythroblasts': {
            'color': 'rgba(147, 112, 219, 0.3)',
            'line_color': 'rgb(147, 112, 219)'
        },
        'CD34-megakaryocytes': {
            'color': 'rgba(255, 165, 0, 0.3)',
            'line_color': 'rgb(255, 165, 0)'
        },
        'Monocytes': {
            'color': 'rgba(0, 191, 255, 0.3)',
            'line_color': 'rgb(0, 191, 255)'
        },
        'Neutrophils': {
            'color': 'rgba(255, 192, 203, 0.3)',
            'line_color': 'rgb(255, 192, 203)'
        }
    }
    violin_cell_types = ['CD34-erythroblasts', 'CD34-megakaryocytes', 'Monocytes', 'Neutrophils']
    for (row, col), cell_type_name in zip(violin_positions, violin_cell_types):
        fig.add_trace(
            go.Violin(
                x=violin_data['dilution'].astype(str),  # Convert to string to make categorical
                y=violin_data[cell_type_name],
                name=cell_type_name,
                fillcolor=cell_types_config[cell_type_name]['color'],
                line_color=cell_types_config[cell_type_name]['line_color'],
                box=dict(
                    visible=True,
                    fillcolor='white',
                    line=dict(color='black', width=1)
                ),
                meanline=dict(visible=False),
                points=False,
                side='both',
                hoveron='violins',
                scalegroup='all',  # Keep consistent scaling across all plots
                scalemode='width'  # Scale by width rather than count
            ),
            row=row, col=col
        )
    fig.update_yaxes(range=[0, None])
    # Automatically set subplot titles
    fig.update_layout(
        title=dict(
            text=f"Violin Plot Subplots",
            x=0.5,  # Center the overall title
            xanchor="center"
        )
    )
    # Update x-axes for violin plots
    fig.update_xaxes(title="Dilution", row=3, col=1)
    fig.update_xaxes(title="Dilution", row=3, col=2)
    fig.update_xaxes(title="Dilution", row=4, col=1)
    fig.update_xaxes(title="Dilution", row=4, col=2)
    # Update y-axes for violin plots to avoid clipping
    for row in [3, 4]:
        for col in [1, 2]:
            fig.update_yaxes(title="Proportion", range=[0, 0.5], row=row, col=col)  # Adjusted to 0-0.5 to avoid clipping
    # Update overall layout
    fig.update_layout(
        height=1400,
        width=1600,
        showlegend=False,
        title=dict(
            text=f"Cell Type Analysis: {cell_type}",
            x=0.5,
            xanchor='center'
        )
    )
    # Save outputs
    metrics_df.to_csv(f"{output_path}{cell_type}_model_performance_metrics.csv", index=False)
    fig.write_html(f"{output_path}{cell_type}.html")
    fig.write_image(f"{output_path}{cell_type}.png", scale=2)
    return fig, metrics_df


def predict(model_path, cell_type, marker_read_proportions, marker_read_coverage, atlas_path, output_path):
    print("marker_read_proportions", marker_read_proportions.head())
    print("marker_read_coverage", marker_read_coverage.head())
    results, all_cell_types_df = predict_cell_type_contribution(
        model_path=model_path,
        marker_read_proportions=marker_read_proportions,
        marker_read_coverage=marker_read_coverage,
        cell_type=cell_type,
        atlas=pd.read_csv(atlas_path, sep='\t'),
    )
    plot_dilution_results(
        results_df=results,
        cell_type=cell_type, 
        all_predictions_df=all_cell_types_df, 
        output_path=output_path)



def plot_nnls_results_for_benchmark(nnls_out_path, cell_type, output_path):
    df = pd.read_csv(nnls_out_path)
    cell_type_df = get_contribution_dataframe(df, cell_type)
    all_predictions_df = df.set_index('CellType').transpose()
    all_predictions_df['sample_idx'] = all_predictions_df.index.str.split('_').str[1].astype(int)
    all_predictions_df['dilution'] = all_predictions_df.index.map(lambda x: x.split('_')[0][1:]) 
    all_predictions_df = all_predictions_df.sort_values(['dilution', 'sample_idx'])
    del all_predictions_df['sample_idx']
    del all_predictions_df['dilution']
    plot_dilution_results(
        results_df=cell_type_df,
        cell_type=cell_type, 
        all_predictions_df=all_predictions_df, 
        output_path=output_path)


def get_contribution_dataframe(df, cell_type):
    if cell_type not in df['CellType'].values:
        raise ValueError(f"CellType '{cell_type}' not found in the dataframe.")
    
    # Extract the row corresponding to the given CellType
    row = df[df['CellType'] == cell_type]
    
    # Extract the columns that are sample-related (excluding 'CellType')
    samples = row.columns.drop('CellType')
    contributions = row[samples].values.flatten()
    
    # Create the resulting dataframe
    result_df = pd.DataFrame({
        'sample': samples,
        'dilution': [extract_dilution(sample) for sample in samples],
        'contribution': contributions
    })
    results_df['sample_idx'] = results_df['sample'].str.split('_').str[1].astype(int)
    results_df = results_df.sort_values(['dilution', 'sample_idx'])
    del results_df['sample_idx']
    return result_df