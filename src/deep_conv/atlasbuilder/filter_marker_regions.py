import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import concurrent.futures


logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

CELL_TYPES = [
    'B-cells',
    'CD34-erythroblasts',
    'CD34-megakaryocytes',
    'Colon',
    'Esophagus',
    'Gastric',
    'Granulocytes',
    'Monocytes',
    'NK-cells',
    'OAC',
    'Small-intestine',
    'T-cells'
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


def process_cell_type(input_dir: Path, 
                     output_dir: Path,
                     cell_type: str):
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
    output_file = output_dir / f"{cell_type}_filtered_markers.parquet"
    filtered_df.to_parquet(output_file)
    logging.info(f"Saved filtered markers to {output_file}")


def process_cell_type_wrapper(cell_type, input_dir, output_dir):
    try:
        process_cell_type(
            input_dir=input_dir,
            output_dir=output_dir,
            cell_type=cell_type,
        )
    except Exception as e:
        logging.error(f"Error processing {cell_type}: {e}")


def run_in_parallel(num_threads, cell_types, input_dir, output_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_cell_type_wrapper, cell_type, input_dir, output_dir)
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
    min_cpgs = args.min_cpgs
    
    logging.info(f"Starting marker filtering")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    threads = len(CELL_TYPES)
    run_in_parallel(threads, CELL_TYPES, input_dir, output_dir, min_cpgs)
    markers = pd.read_parquet(list(output_dir.glob("*.parquet")))
    markers = markers.dropna()
    
    markers[['chr','start','end','startCpG','endCpG','target','name','direction','B-cells', 'CD34-erythroblasts', 'CD34-megakaryocytes', 'Colon', 'Esophagus', 'Gastric', 'Granulocytes', 'Monocytes', 'NK-cells', 'OAC', 'Small-intestine','T-cells', 'B-cells_coverage', 'CD34-erythroblasts_coverage', 'CD34-megakaryocytes_coverage', 'Colon_coverage', 'Esophagus_coverage', 'Gastric_coverage', 'Granulocytes_coverage',   'Monocytes_coverage', 'NK-cells_coverage','OAC_coverage', 'Small-intestine_coverage','T-cells_coverage', 'snr', 'snr_vs_median', 'snr_vs_mean', 'target_value','max_background', 'median_background', 'mean_background','background_std', 'background_range','background_quartile_ratio', 'signal_to_noise_area','relative_signal_strength','quality_score']].to_csv(args.output_markers, sep="\t", index=False)
    markers[['chr','start','end','startCpG','endCpG','target','name','direction','B-cells', 'CD34-erythroblasts', 'CD34-megakaryocytes', 'Colon', 'Esophagus', 'Gastric', 'Granulocytes', 'Monocytes', 'NK-cells', 'OAC', 'Small-intestine','T-cells']].to_csv(args.output_atlas, sep="\t", index=False)

    logging.info(f"\nCompleted marker filtering")
    
if __name__ == "__main__":
    main()