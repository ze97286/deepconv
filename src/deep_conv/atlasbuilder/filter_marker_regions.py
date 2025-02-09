import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List
import logging
from scipy import stats
import concurrent.futur
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from concurrent.futures import ProcessPoolExecutor
from functools import partial

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

CELL_TYPES = [
    "B-cells",
    "CD34-erythroblasts",
    "CD34-megakaryocytes",
    "CD4-T-cells",
    "CD8-T-cells",
    "Colon",
    "Duodenum",
    "Eosinophils",
    "Esophagus",
    "Monocytes",
    "Neutrophils",    
    "NK-cells",
    "Pancreas",
    "Stomach",
    "OAC",
]


def select_markers_for_cell_type(df: pd.DataFrame, min_markers: int = 100):
    """Select markers for a cell type ensuring minimum non-overlapping markers plus redundancy"""
    # First, find non-overlapping markers
    # Calculate quality score with capped SNRs on log scale
    cell_type_markers = df.copy()
    # Vectorized quality score calculation
    max_snr_capped = np.minimum(cell_type_markers['snr'], 100)
    median_snr_capped = np.minimum(cell_type_markers['snr_vs_median'], 100)
    cell_type_markers['quality_score'] = (
        cell_type_markers['target_value'] * 
        np.log1p(max_snr_capped) * 
        np.log1p(median_snr_capped) * 
        (1 - cell_type_markers['background_std'])
    )
    print("calculated cell_type_markers")
    # Select non-overlapping markers first
    selected = []
    sorted_markers = cell_type_markers.sort_values('quality_score', ascending=False)
    print("sorted cell_type_markers")
    for _, marker in sorted_markers.iterrows():
        # Check if overlaps with any selected marker
        overlaps = False
        for selected_marker in selected:
            if (marker['chr'] == selected_marker['chr'] and
                marker['start'] <= selected_marker['end'] and
                marker['end'] >= selected_marker['start']):
                overlaps = True
                break
        if not overlaps:
            selected.append(marker.to_dict())
            if len(selected) >= min_markers:
                break
    # Now add redundant markers for each selected marker
    redundant_markers = []
    for marker in selected:
        # Find overlapping markers with good scores
        overlapping = cell_type_markers[
            (cell_type_markers['chr'] == marker['chr']) &
            (cell_type_markers['start'] <= marker['end']) &
            (cell_type_markers['end'] >= marker['start']) &
            (cell_type_markers['quality_score'] > marker['quality_score'] * 0.8)
        ]
        # Take top 2 redundant markers
        redundant_markers.extend(overlapping.nlargest(2, 'quality_score').to_dict('records'))
    return pd.DataFrame(selected + redundant_markers)


def evaluate_low_signal_performance(df):
    # Look at ratio of target to background when target is in bottom quartile
    low_signal = df[df['target_value'] <= df['target_value'].quantile(0.25)]
    return {
        'min_separation': (low_signal['target_value'] / low_signal['max_background']).min(),
        'median_separation': (low_signal['target_value'] / low_signal['max_background']).median()
    }


def evaluate_marker_separation(df):
    # For each marker
    separability_metrics = {
        'target_bg_gap': df['target_value'] - df['max_background'],
        'target_bg_ratio': df['target_value'] / df['max_background'],
        'background_spread': df['max_background'] - df['median_background'],
        'signal_noise_ratio': (df['target_value'] - df['mean_background']) / df['background_std']
    }
    return pd.DataFrame(separability_metrics)


def plot_marker_comparison(new_markers_df: pd.DataFrame, old_markers_df: pd.DataFrame, cell_type: str):
    # Calculate metrics for both sets
    new_metrics = evaluate_marker_separation(new_markers_df)
    old_metrics = evaluate_marker_separation(old_markers_df)
    
    # Calculate low signal performance
    new_low_signal = evaluate_low_signal_performance(new_markers_df)
    old_low_signal = evaluate_low_signal_performance(old_markers_df)
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Target/Background Ratio Distribution',
            'Signal/Noise Ratio Distribution',
            'Target-Background Gap Distribution',
            'Background Spread Distribution',
            'Low Signal Performance Comparison',
            'Target Value Distribution in Bottom Quartile'
        )
    )
    
    # Add histograms for each metric
    fig.add_trace(
        go.Histogram(x=new_metrics['target_bg_ratio'], name='New Markers', opacity=0.75),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=old_metrics['target_bg_ratio'], name='Old Markers', opacity=0.75),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=new_metrics['signal_noise_ratio'], name='New Markers', opacity=0.75),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=old_metrics['signal_noise_ratio'], name='Old Markers', opacity=0.75),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Histogram(x=new_metrics['target_bg_gap'], name='New Markers', opacity=0.75),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=old_metrics['target_bg_gap'], name='Old Markers', opacity=0.75),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=new_metrics['background_spread'], name='New Markers', opacity=0.75),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=old_metrics['background_spread'], name='Old Markers', opacity=0.75),
        row=2, col=2
    )
    
    # Add bar chart for low signal performance comparison
    fig.add_trace(
        go.Bar(
            x=['Min Separation', 'Median Separation'],
            y=[new_low_signal['min_separation'], new_low_signal['median_separation']],
            name='New Markers'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(
            x=['Min Separation', 'Median Separation'],
            y=[old_low_signal['min_separation'], old_low_signal['median_separation']],
            name='Old Markers'
        ),
        row=3, col=1
    )
    
    # Add distribution of target values in bottom quartile
    new_low_quartile = new_markers_df[new_markers_df['target_value'] <= new_markers_df['target_value'].quantile(0.25)]
    old_low_quartile = old_markers_df[old_markers_df['target_value'] <= old_markers_df['target_value'].quantile(0.25)]
    
    fig.add_trace(
        go.Histogram(
            x=new_low_quartile['target_value'],
            name='New Markers',
            opacity=0.75
        ),
        row=3, col=2
    )
    fig.add_trace(
        go.Histogram(
            x=old_low_quartile['target_value'],
            name='Old Markers',
            opacity=0.75
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'Marker Comparison for {cell_type}',
        height=1200,  # Increased height for additional plots
        width=1000,
        showlegend=True,
        barmode='group'  # For the bar charts
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Separation Metric', row=3, col=1)
    fig.update_xaxes(title_text='Target Value', row=3, col=2)
    fig.update_yaxes(title_text='Count', row=3, col=2)
    fig.update_yaxes(title_text='Separation Ratio', row=3, col=1)
    
    fig.write_html(f"/users/zetzioni/sharedscratch/atlas/comapre_markers/{cell_type}_old_vs_new_atlas.html")


def evaluate_marker_independence(df):
    # Calculate correlations between marker values across cell types
    cell_type_cols = [col for col in df.columns if col in CELL_TYPES]
    correlations = df[cell_type_cols].corr()
    return correlations


def process_cell_type(input_dir: Path, 
                     output_dir: Path,
                     cell_type: str,
                     old_atlas: pd.DataFrame):
    """Process markers for a single cell type with statistics"""
    logging.info(f"\nProcessing {cell_type}")
    # Find all marker files for this cell type
    marker_files = list(input_dir.glob(f"*_{cell_type}_markers_*.parquet"))
    if not marker_files:
        logging.warning(f"No marker files found for {cell_type}")
        return
    combined_df = pd.read_parquet(marker_files)
    logging.info(f"Loaded {len(combined_df)} total markers for {cell_type}")
    filtered_df = select_markers_for_cell_type(combined_df)
    print(f"filtering {cell_type} => {len(filtered_df)}, nonoverlapping: {len(filtered_df.groupby('startCpG').count())}")
    old_atlas = old_atlas.copy()
    old_atlas = old_atlas[old_atlas.target==cell_type]
    old_atlas['target_value']=old_atlas[cell_type]
    old_atlas['max_background']=old_atlas[old_atlas.columns[8:]].drop(columns=[cell_type]).max(axis=1)
    old_atlas['median_background']=old_atlas[old_atlas.columns[8:]].drop(columns=[cell_type]).median(axis=1)
    old_atlas['mean_background']=old_atlas[old_atlas.columns[8:]].drop(columns=[cell_type]).mean(axis=1)
    old_atlas['background_std']=old_atlas[old_atlas.columns[8:]].drop(columns=[cell_type]).std(axis=1)
    plot_marker_comparison(filtered_df, old_atlas, cell_type)
    print("saving cell type markers with",len(filtered_df),"markers")
    # Save filtered markers
    output_file = output_dir / f"{cell_type}_filtered_markers.parquet"
    filtered_df.to_parquet(output_file)
    logging.info(f"Saved filtered markers to {output_file}")


def process_cell_type_wrapper(cell_type, input_dir, output_dir, old_atlas):
    try:
        process_cell_type(
            input_dir=input_dir,
            output_dir=output_dir,
            cell_type=cell_type,
            old_atlas=old_atlas,
        )
    except Exception as e:
        logging.error(f"Error processing {cell_type}: {e}")


def run_in_parallel(num_threads, cell_types, input_dir, output_dir):
    old_atlas = pd.read_csv("/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed", sep="\t").dropna()
    old_atlas.rename(columns={'duodenum':'Duodenum'}, inplace=True)
    old_atlas['target']=old_atlas['target'].map(lambda x: x if x != 'duodenum' else 'Duodenum')
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_cell_type_wrapper, cell_type, input_dir, output_dir, old_atlas)
            for cell_type in cell_types
        ]
        
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise any exceptions that occurred during execution
            except Exception as e:
                logging.error(f"Error in future: {e}")



def main():
    parser = argparse.ArgumentParser(description='Filter methylation markers for each cell type')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing marker files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save filtered markers')
    parser.add_argument('--output_markers', type=str, required=True, help='markers output file name')
    parser.add_argument('--output_atlas', type=str, required=True, help='atlas output file name')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting marker filtering")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    threads = len(CELL_TYPES)
    run_in_parallel(threads, CELL_TYPES, input_dir, output_dir)
    markers = pd.read_parquet(list(output_dir.glob("*.parquet")))
    markers = markers.dropna()
    
    # markers[['chr','start','end','startCpG','endCpG','target','name','direction','B-cells','CD34-erythroblasts','CD34-megakaryocytes','CD4-T-cells','CD8-T-cells','Colon','Duodenum','Eosinophils','Esophagus','Monocytes','NK-cells','Neutrophils','OAC','Pancreas','Stomach','B-cells_coverage','CD34-erythroblasts_coverage','CD34-megakaryocytes_coverage','CD4-T-cells_coverage','CD8-T-cells_coverage','Colon_coverage','Duodenum_coverage','Eosinophils_coverage','Esophagus_coverage','Monocytes_coverage','NK-cells_coverage','Neutrophils_coverage','OAC_coverage','Pancreas_coverage','Stomach_coverage','snr', 'snr_vs_median', 'snr_vs_mean', 'target_value','max_background', 'median_background', 'mean_background','background_std', 'background_range','background_quartile_ratio', 'signal_to_noise_area','relative_signal_strength','quality_score']].to_csv("/users/zetzioni/sharedscratch/atlas/markers/zohar.blood+gi+tum.l4.bed", sep="\t", index=False)
    # markers[['chr','start','end','startCpG','endCpG','target','name','direction','B-cells','CD34-erythroblasts','CD34-megakaryocytes','CD4-T-cells','CD8-T-cells','Colon','Duodenum','Eosinophils','Esophagus','Monocytes','NK-cells','Neutrophils','OAC','Pancreas','Stomach']].to_csv("/users/zetzioni/sharedscratch/atlas/atlas/atlas_zohar.blood+gi+tum.l4.bed", sep="\t", index=False)
    markers[['chr','start','end','startCpG','endCpG','target','name','direction','B-cells','CD34-erythroblasts','CD34-megakaryocytes','CD4-T-cells','CD8-T-cells','Colon','Duodenum','Eosinophils','Esophagus','Monocytes','NK-cells','Neutrophils','OAC','Pancreas','Stomach','B-cells_coverage','CD34-erythroblasts_coverage','CD34-megakaryocytes_coverage','CD4-T-cells_coverage','CD8-T-cells_coverage','Colon_coverage','Duodenum_coverage','Eosinophils_coverage','Esophagus_coverage','Monocytes_coverage','NK-cells_coverage','Neutrophils_coverage','OAC_coverage','Pancreas_coverage','Stomach_coverage','snr', 'snr_vs_median', 'snr_vs_mean', 'target_value','max_background', 'median_background', 'mean_background','background_std', 'background_range','background_quartile_ratio', 'signal_to_noise_area','relative_signal_strength','quality_score']].to_csv(args.output_markers, sep="\t", index=False)
    markers[['chr','start','end','startCpG','endCpG','target','name','direction','B-cells','CD34-erythroblasts','CD34-megakaryocytes','CD4-T-cells','CD8-T-cells','Colon','Duodenum','Eosinophils','Esophagus','Monocytes','NK-cells','Neutrophils','OAC','Pancreas','Stomach']].to_csv(args.output_atlas, sep="\t", index=False)

    logging.info(f"\nCompleted marker filtering")
    
if __name__ == "__main__":
    main()