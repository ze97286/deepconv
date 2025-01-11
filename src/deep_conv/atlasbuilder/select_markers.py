import pandas as pd
import glob

def select_markers_per_cell_type(df_dict, params_dict):
    """
    Select markers for each cell type with specific parameters
    
    Args:
        df_dict: Dictionary of {cell_type: DataFrame}
        params_dict: Dictionary of {cell_type: parameter_dict}
    """
    results = {}
    for cell_type, df in df_dict.items():
        params = params_dict.get(cell_type, {})
        results[cell_type] = select_non_overlapping_markers(
            df,
            min_snr=params.get('min_snr', 2.0),
            min_signal=params.get('min_signal', 0.4),
            min_markers=params.get('min_markers', 100),
            max_markers=params.get('max_markers', 1000)
        )
    return results

def select_non_overlapping_markers(df, min_snr=2.0, min_signal=0.4, min_markers=100, max_markers=1000):
    """
    Select non-overlapping markers optimizing for:
    - Higher SNR
    - Higher target value (signal)
    - Lower background noise
    - Reasonable length
    
    Args:
        df: DataFrame with columns: name, target_value, snr, background_std, length
        min_snr: Minimum SNR threshold
        min_signal: Minimum target value threshold
        min_markers: Minimum number of markers to select
        max_markers: Maximum number of markers to select
    """
    # First filter by basic thresholds
    candidates = df[
        (df['snr'] >= min_snr) & 
        (df['target_value'] >= min_signal)
    ].copy()
    # Extract coordinates for overlap checking
    candidates[['chr', 'start', 'end']] = candidates['name'].str.extract(r'(chr\w+):(\d+)-(\d+)')
    candidates[['start', 'end']] = candidates[['start', 'end']].astype(int)
    # Sort by multiple criteria (can adjust weights)
    candidates['score'] = (
        candidates['snr'] * 2 +              # Higher weight on SNR
        candidates['target_value'] -         # Higher signal is better
        candidates['background_std'] * 0.5   # Lower noise is better
    )
    candidates = candidates.sort_values('score', ascending=False)
    # Select non-overlapping regions
    selected = []
    used_regions = set()  # Track regions we've used
    def overlaps_with_used(row):
        """Check if region overlaps with any previously selected region"""
        for chrom, start, end in used_regions:
            if (row['chr'] == chrom and 
                row['start'] <= end and 
                row['end'] >= start):
                return True
        return False
    for _, row in candidates.iterrows():
        if len(selected) >= max_markers:
            break
            
        if not overlaps_with_used(row):
            selected.append(row)
            used_regions.add((row['chr'], row['start'], row['end']))
            
        if len(selected) >= min_markers:
            # If we have enough markers, can do additional filtering
            if row['snr'] < min_snr * 1.2:  # Higher SNR threshold
                break
    return pd.DataFrame(selected)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process pat files for UXM analysis')
    parser.add_argument('--input_dir', required=True, help='Path to parts directory')
    parser.add_argument('--marker_output_name', required=True, help='Path to output marker directory')
    parser.add_argument('--atlas_output_name', required=True, help='Path to output marker directory')
    args = parser.parse_args()

    config = {
        # cell_type: [min_score, max_markers]
        "B-cells":[10, 100],
        "CD34-erythroblasts":[10, 100],
        "CD34-megakaryocytes":[10, 100],
        "CD4-T-cells": [6,250],
        "CD8-T-cells": [6,200],
        "Colon":[10, 100],
        "Eosinophils":[10, 100],
        "Esophagus":[10, 100],
        "Monocytes":[10, 100],
        "Neutrophils":[10,100],
        "NK-cells":[10,100],
        "OAC":[10,100],
        "Pancreas":[10, 100],
        "Stomach":[8, 100],
        "Duodenum":[10, 100],
    }

    markers_dfs = []
    input_dir = args.input_dir
    marker_output_name = args.marker_output_name
    atlas_output_name = args.atlas_output_name

    for ct, ct_config in config.items():
        df = pd.read_parquet(glob.glob(f"{input_dir}/*{ct}*"))
        score, max_markers = ct_config
        print("preparing target", ct)
        scores = select_non_overlapping_markers(df)
        scores['target'] = ct
        markers_dfs.append(scores[scores.score>=score].iloc[:max_markers])
        
    markers = pd.concat(markers_dfs, axis=0, ignore_index=True)
    markers.to_csv(marker_output_name, sep="\t")
    atlas = markers[['chr','start','end','startCpG','endCpG','target','name','direction','B-cells','CD34-erythroblasts','CD34-megakaryocytes','CD4-T-cells','CD8-T-cells','Colon','Duodenum','Eosinophils','Esophagus','Monocytes','NK-cells','Neutrophils','OAC','Pancreas','Stomach']]
    atlas.to_csv(atlas_output_name, sep="\t")



if __name__ == "__main__":    
    main()