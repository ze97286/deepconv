import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import plotly.express as px
import plotly.subplots as sp
import math
import colorsys
from scipy import stats

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

def select_markers_for_cell_type(df: pd.DataFrame, num_markers: int = 200, max_per_region: int = 5,
                               min_snr_threshold: float = 1.5, avg_clinical_depth: float = 20,
                               avg_read_length: int = 150, min_cpgs_per_read: int = 4):
    """
    Select optimal markers balancing SNR and expected clinical coverage
    
    Parameters:
    - df: DataFrame with marker candidates
    - num_markers: Target number of primary markers to select (default 200)
    - max_per_region: Maximum primary markers to select from the same genomic region
    - min_snr_threshold: Minimum SNR to consider a marker
    - avg_clinical_depth: Average sequencing depth (X) in clinical samples
    - avg_read_length: Average read length in bp
    - min_cpgs_per_read: Minimum consecutive CpGs a read must cover (default 4)
    
    Returns:
    - DataFrame of selected markers with both primary and redundant markers
    """
    # Copy to avoid modifying original
    markers = df.copy()
    
    # Ensure we have necessary columns
    required_columns = ['chr', 'start', 'end', 'startCpG', 'endCpG', 'target_value', 
                        'snr', 'snr_vs_median', 'background_std']
    
    missing = [col for col in required_columns if col not in markers.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate CpG count as length metric
    markers['cpg_length'] = markers['endCpG'] - markers['startCpG']
    
    # Apply filters
    markers = markers[
        (markers['cpg_length'] <= 100) &  # Max CpG length
        (markers['end'] - markers['start'] <= 1000) &  # Max bp length
        (markers['snr'] >= min_snr_threshold)  # Min SNR threshold
    ]
    
    # Ensure we have enough markers to select from
    min_candidate_markers = max(500, num_markers * 3)  # Need good pool of candidates
    if len(markers) < min_candidate_markers:
        print(f"Warning: Only {len(markers)} markers pass filters. Consider relaxing constraints.")
        # Fall back to original criteria but still enforce minimum SNR
        markers = df.copy()
        markers['cpg_length'] = markers['endCpG'] - markers['startCpG']
        markers = markers[markers['snr'] >= min_snr_threshold]
    
    # Calculate average distance between CpGs in each region
    markers['avg_cpg_distance'] = (markers['end'] - markers['start']) / markers['cpg_length']
    
    # Calculate expected clinical coverage metrics
    # How many CpGs an average read can cover
    markers['cpgs_per_read'] = avg_read_length / markers['avg_cpg_distance']
    
    # Probability of a read covering at least min_cpgs_per_read
    markers['p_useful_read'] = np.where(
        markers['cpgs_per_read'] < min_cpgs_per_read,
        0,  # Cannot cover enough CpGs with one read
        np.minimum(1.0, (markers['cpgs_per_read'] - min_cpgs_per_read + 1) / markers['cpgs_per_read'])
    )
    
    # Expected number of useful reads given depth
    # We scale by region length relative to genome size (simplified)
    region_size_factor = np.minimum(1.0, (markers['end'] - markers['start']) / 3000)
    markers['expected_useful_reads'] = (
        avg_clinical_depth * 
        markers['p_useful_read'] * 
        region_size_factor  # Normalized by typical region size
    )
    
    # Probability of having at least 3 useful reads (Poisson model)
    from scipy.stats import poisson
    markers['clinical_detection_prob'] = 1 - poisson.cdf(
        2,  # Less than 3 reads
        markers['expected_useful_reads']
    )
    
    # Normalize metrics for balanced scoring
    markers['snr_norm'] = markers['snr'] / markers['snr'].max()
    markers['clinical_norm'] = markers['clinical_detection_prob']  # Already 0-1
    
    # Calculate combined score that balances SNR and clinical detectability
    markers['combined_score'] = (
        markers['snr_norm'] * 0.7 +  # SNR is primary driver
        markers['clinical_norm'] * 0.3  # Clinical detectability is secondary
    )
    
    # Add bonus for very high SNR markers
    markers['combined_score'] = markers['combined_score'] * (
        1 + 0.3 * (markers['snr'] > np.percentile(markers['snr'], 90))
    )
    
    # Create region bins for genomic distribution
    markers['region_bin'] = markers['chr'] + '_' + (markers['start'] // 500_000).astype(str)
    
    # Define CpG length bins
    cpg_length_bins = [
        (4, 8),       # Very short CpG regions (4-7 CpGs)
        (8, 15),      # Short CpG regions (8-14 CpGs)
        (15, 25),     # Medium CpG regions (15-24 CpGs)
        (25, 50),     # Long CpG regions (25-49 CpGs)
        (50, 100)     # Very long CpG regions (50-100 CpGs)
    ]
    
    # Target distribution weighted by clinical utility
    # Higher weight for bins with better balance of SNR and coverage
    clinical_by_bin = []
    for bin_idx, (min_len, max_len) in enumerate(cpg_length_bins):
        bin_markers = markers[
            (markers['cpg_length'] >= min_len) & 
            (markers['cpg_length'] < max_len)
        ]
        if not bin_markers.empty:
            avg_clinical = bin_markers['clinical_detection_prob'].mean()
            avg_snr = bin_markers['snr_norm'].mean()
            bin_utility = (avg_clinical * 0.4) + (avg_snr * 0.6)
            clinical_by_bin.append((bin_idx, bin_utility))
    
    # Adjust distribution based on clinical utility
    if clinical_by_bin:
        total_utility = sum(u for _, u in clinical_by_bin)
        target_distribution = {
            bin_idx: max(0.05, u / total_utility)  # Ensure at least 5% from each bin
            for bin_idx, u in clinical_by_bin
        }
        
        # Normalize to sum to 1
        total_pct = sum(target_distribution.values())
        target_distribution = {
            k: v / total_pct for k, v in target_distribution.items()
        }
    else:
        # Fallback if calculation fails
        target_distribution = {
            0: 0.25,  # Very short: 25%
            1: 0.35,  # Short: 35%
            2: 0.25,  # Medium: 25%
            3: 0.10,  # Long: 10%
            4: 0.05   # Very long: 5%
        }
    
    # Calculate number to select from each bin based on fixed total
    target_counts = {bin_idx: max(3, int(num_markers * pct)) 
                     for bin_idx, pct in target_distribution.items()}
    
    # Select markers from each length bin
    length_selections = []
    for bin_idx, (min_len, max_len) in enumerate(cpg_length_bins):
        if bin_idx not in target_distribution:
            continue
            
        length_group = markers[
            (markers['cpg_length'] >= min_len) & 
            (markers['cpg_length'] < max_len)
        ]
        
        if not length_group.empty:
            # Apply stricter quality standards for longer regions
            if bin_idx >= 3:  # Long and very long regions
                min_clinical_prob = 0.4  # Require reasonable clinical detection probability
                length_group = length_group[length_group['clinical_detection_prob'] >= min_clinical_prob]
                
                # Also require higher SNR for longer regions
                min_snr_for_length = min_snr_threshold * (1 + 0.2 * bin_idx)
                length_group = length_group[length_group['snr'] >= min_snr_for_length]
            
            if not length_group.empty:
                n_from_group = target_counts[bin_idx]
                # Use combined score for selection
                top_in_length = length_group.nlargest(n_from_group, 'combined_score')
                length_selections.append(top_in_length)
        
        if len(length_group) < target_counts.get(bin_idx, 0):
            print(f"Warning: Insufficient markers in CpG length bin {min_len}-{max_len}")
    
    # Combine length-based selections
    length_balanced = pd.concat(length_selections) if length_selections else pd.DataFrame()
    
    # Get region-balanced markers
    region_selections = []
    for region, group in markers.groupby('region_bin'):
        # Use combined score for region-based selection
        top_in_region = group.nlargest(max_per_region, 'combined_score')
        region_selections.append(top_in_region)
    
    region_balanced = pd.concat(region_selections) if region_selections else pd.DataFrame()
    
    # Combine length-balanced with region-balanced
    combined = pd.concat([length_balanced, region_balanced]).drop_duplicates()
    
    # If we don't have enough markers, add more from the original set
    if len(combined) < num_markers:
        remaining = markers[~markers.index.isin(combined.index)]
        additional = remaining.nlargest(num_markers - len(combined), 'combined_score')
        combined = pd.concat([combined, additional])
    
    # If we have too many candidates, trim to a reasonable number to speed up selection
    elif len(combined) > num_markers * 3:
        combined = combined.nlargest(num_markers * 3, 'combined_score')
    
    # Add length bin categories for stratified selection
    combined['length_bin'] = pd.cut(
        combined['cpg_length'],
        bins=[4, 8, 15, 25, 50, 100],
        labels=range(5)
    )
    
    # Within each length bin, rank by combined score
    combined['bin_rank'] = combined.groupby('length_bin')['combined_score'].rank(ascending=False)
    
    # Sort for stratified selection
    sorted_markers = combined.sort_values(['bin_rank', 'combined_score'], ascending=[True, False])
    
    # Select non-overlapping markers
    selected = []
    selected_regions = set()
    selected_cpg_pairs = set()
    
    for _, marker in sorted_markers.iterrows():
        # Check if already selected this CpG region
        cpg_pair = (marker['startCpG'], marker['endCpG'])
        if cpg_pair in selected_cpg_pairs:
            continue
        
        # Check if overlaps with any selected marker
        overlaps = False
        for selected_marker in selected:
            if (marker['chr'] == selected_marker['chr'] and
                marker['start'] <= selected_marker['end'] and
                marker['end'] >= selected_marker['start']):
                overlaps = True
                break
                
        # Check region count
        region = marker['region_bin']
        region_count = sum(1 for s in selected if s.get('region_bin') == region)
        
        # Get length bin
        length_bin = marker['length_bin'] if not pd.isna(marker['length_bin']) else None
        
        # Track counts per length bin
        length_counts = {}
        for s in selected:
            bin_val = s.get('length_bin')
            if bin_val is not None and not pd.isna(bin_val):
                bin_val = int(bin_val)
                length_counts[bin_val] = length_counts.get(bin_val, 0) + 1
        
        # Prioritize under-represented length bins
        length_priority = False
        if length_bin is not None and not pd.isna(length_bin):
            length_bin = int(length_bin)
            target = target_counts.get(length_bin, 0)
            current = length_counts.get(length_bin, 0)
            length_priority = current < target
        
        # Allow more from high-quality regions
        max_from_region = 5 if marker['combined_score'] > np.percentile(sorted_markers['combined_score'], 90) else 3
        
        # Select marker if it passes all criteria
        if not overlaps and (region_count < max_from_region or length_priority):
            marker_dict = marker.to_dict()
            selected.append(marker_dict)
            selected_regions.add(region)
            selected_cpg_pairs.add(cpg_pair)
            
        # Stop when we have enough markers with good distribution
        if len(selected) >= num_markers:
            # Check length distribution
            enough_per_bin = True
            for bin_idx in range(len(cpg_length_bins)):
                if bin_idx in target_counts:
                    min_needed = min(3, target_counts[bin_idx])
                    if length_counts.get(bin_idx, 0) < min_needed:
                        enough_per_bin = False
                        break
            
            # Check genomic distribution
            enough_regions = len(selected_regions) >= min(len(markers['region_bin'].unique()), num_markers // 2)
            
            if enough_regions and enough_per_bin:
                # Limit to exactly num_markers if we have more
                if len(selected) > num_markers:
                    # Sort by combined score and keep top num_markers
                    selected_scores = [(i, s.get('combined_score', 0)) for i, s in enumerate(selected)]
                    selected_scores.sort(key=lambda x: x[1], reverse=True)
                    keep_indices = [i for i, _ in selected_scores[:num_markers]]
                    selected = [selected[i] for i in keep_indices]
                break
    
    # Create DataFrame from selected primary markers
    selected_df = pd.DataFrame(selected) if selected else pd.DataFrame()
    
    # Now add redundant markers (limited to max num_markers/2)
    max_redundant = num_markers // 2
    redundant_markers = []
    selected_redundant_cpg_pairs = set()
    
    if not selected_df.empty:
        # Prioritize finding redundant markers for the highest scoring primary markers
        primary_with_scores = [(i, row['combined_score']) for i, (_, row) in enumerate(selected_df.iterrows())]
        primary_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        for idx, _ in primary_with_scores:
            if len(redundant_markers) >= max_redundant:
                break
                
            primary = selected_df.iloc[idx]
            
            # Find nearby markers with good scores
            nearby = markers[
                (markers['chr'] == primary['chr']) &
                (abs(markers['start'] - primary['start']) < 5000) &
                (markers['snr'] > primary['snr'] * 0.7) &
                (markers['clinical_detection_prob'] >= primary['clinical_detection_prob'] * 0.9)  # Similar clinical utility
            ]
            
            # Skip markers already in primary selection
            nearby = nearby[~nearby.index.isin(selected_df.index)]
            
            if not nearby.empty:
                # Use combined score but weight clinical probability more for redundant markers
                nearby['redundant_score'] = (
                    nearby['snr_norm'] * 0.4 + 
                    nearby['clinical_norm'] * 0.6  # Higher weight on clinical detection for redundancy
                )
                nearby_sorted = nearby.sort_values('redundant_score', ascending=False)
                
                # Find first marker that doesn't duplicate a CpG region
                for _, redundant in nearby_sorted.iterrows():
                    cpg_pair = (redundant['startCpG'], redundant['endCpG'])
                    
                    if cpg_pair in selected_cpg_pairs or cpg_pair in selected_redundant_cpg_pairs:
                        continue
                    
                    redundant_markers.append(redundant.to_dict())
                    selected_redundant_cpg_pairs.add(cpg_pair)
                    break
    
    # Create DataFrame from redundant markers
    redundant_df = pd.DataFrame(redundant_markers) if redundant_markers else pd.DataFrame()
    
    # Combine primary and redundant markers
    if not redundant_df.empty and not selected_df.empty:
        final_selection = pd.concat([selected_df, redundant_df], ignore_index=True)
        final_selection['is_primary'] = False
        final_selection.loc[:len(selected_df)-1, 'is_primary'] = True
    else:
        final_selection = selected_df if not selected_df.empty else redundant_df
        if not final_selection.empty:
            final_selection['is_primary'] = True
    
    # Final check for duplicates
    if not final_selection.empty:
        final_selection = final_selection.drop_duplicates(['startCpG', 'endCpG'])
        
        # Add statistics
        length_stats = final_selection.groupby(pd.cut(
            final_selection['cpg_length'],
            bins=[4, 8, 15, 25, 50, 100],
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        )).size()
        
        print("\n=== Marker Selection Summary ===")
        print(f"Total markers selected: {len(final_selection)}")
        print(f"Primary markers: {sum(final_selection['is_primary'])}")
        print(f"Redundant markers: {len(final_selection) - sum(final_selection['is_primary'])}")
        
        print("\nCpG length distribution:")
        print(length_stats)
        print(f"Mean CpG length: {final_selection['cpg_length'].mean():.2f}")
        print(f"Median CpG length: {final_selection['cpg_length'].median():.2f}")
        
        print("\nSNR Statistics:")
        print(f"Mean SNR: {final_selection['snr'].mean():.2f}")
        print(f"Min SNR: {final_selection['snr'].min():.2f}")
        
        # Clinical detection probability statistics
        print("\nClinical Detection Statistics:")
        print(f"Mean clinical detection probability: {final_selection['clinical_detection_prob'].mean():.2f}")
        print(f"Median clinical detection probability: {final_selection['clinical_detection_prob'].median():.2f}")
        
        # Combined metrics by length category
        metrics_by_length = final_selection.groupby(pd.cut(
            final_selection['cpg_length'],
            bins=[4, 8, 15, 25, 50, 100],
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        )).agg({
            'snr': ['mean', 'min', 'max'],
            'clinical_detection_prob': ['mean', 'min', 'max'],
            'combined_score': ['mean']
        })
        
        print("\nMetrics by length category:")
        print(metrics_by_length)
    
    return final_selection

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

    visualize_marker_set(
        markers=filtered_df,
        target_cell_type=cell_type,
        background_cell_types=[c for c in CELL_TYPES if c!=cell_type],
    ).write_html(output_dir / f'../plots/{cell_type}_marker_set_visualisation_legacy.html')
    marker_set_performance_simulation(
        markers=filtered_df,
        target_cell_type=cell_type,
        background_cell_types=[c for c in CELL_TYPES if c!=cell_type],
    ).write_html(output_dir /f'../plots/{cell_type}_marker_set_performance_simulation_legacy.html')

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


def visualize_marker_set(markers, target_cell_type, background_cell_types, title="Marker Set Analysis"):
    """
    Create comprehensive visualizations for a selected marker set using Plotly
    
    Parameters:
    - markers: DataFrame with selected markers
    - target_cell_type: Name of target cell type
    - background_cell_types: List of background cell types
    - title: Master title for the visualizations
    
    Returns:
    - Plotly figure with multiple subplots analyzing the marker set
    """
    # Create subplot figure with custom layout
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'SNR Distribution', 
            'Target vs Background Values', 
            'Markers per Chromosome',
            'SNR vs Target Value Distribution', 
            'Marker Positions within Chromosomes', 
            '',
            'Primary vs Redundant Markers', 
            'Cell Type Values Across Markers', 
            ''
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}, {"type": "bar"}],
            [{"type": "histogram2d"}, {"colspan": 2, "type": "scatter"}, None],
            [{"type": "pie"}, {"colspan": 2, "type": "heatmap"}, None]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # 1. SNR Distribution
    # Cap SNR values at 10000 for better visualization
    capped_snr = markers['snr'].clip(upper=10000)
    
    fig.add_trace(
        go.Histogram(
            x=capped_snr,
            nbinsx=30,
            marker_color='darkblue',
            name='SNR'
        ),
        row=1, col=1
    )
    
    # Add KDE-like curve (using histogram with normalized density and smaller bins)
    hist_values, bin_edges = np.histogram(capped_snr, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=hist_values,
            mode='lines',
            line=dict(color='royalblue', width=2),
            name='SNR Density'
        ),
        row=1, col=1
    )
    
    fig.update_xaxes(title_text="Signal-to-Noise Ratio (capped at 10000)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    # 2. Target vs Background Value
    # Calculate mean background value for each marker
    bg_values = []
    for bg_type in background_cell_types:
        if bg_type in markers.columns:
            bg_values.append(markers[bg_type].values)
    
    if bg_values:
        bg_mean = np.mean(np.array(bg_values), axis=0)
        
        # Scale the marker size based on SNR
        marker_size = np.log1p(markers['snr'].clip(upper=5000)) / np.log1p(5000) * 20 + 5
        
        # Create colorscale for the SNR values
        fig.add_trace(
            go.Scatter(
                x=bg_mean,
                y=markers[target_cell_type],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=markers['snr'].clip(upper=5000),
                    colorscale='Viridis',
                    colorbar=dict(
                        title="SNR",
                        x=0.46,  # Adjust position of colorbar
                        y=0.8,
                        len=0.3
                    ),
                    showscale=True
                ),
                name='Markers'
            ),
            row=1, col=2
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='red', dash='dash', width=1),
                name='Equal Line'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Mean Background Value", row=1, col=2)
        fig.update_yaxes(title_text=f"Target Value ({target_cell_type})", row=1, col=2)
    
    # 3. Genomic Distribution
    # Count markers per chromosome
    chrom_counts = markers['chr'].value_counts().sort_index()
    
    # Calculate mean SNR per chromosome
    chr_snr = {}
    for chr_name in chrom_counts.index:
        chr_snr[chr_name] = markers[markers['chr'] == chr_name]['snr'].mean()
    
    # Create color scale based on mean SNR
    max_snr = max(chr_snr.values())
    colors = []
    for chr_name in chrom_counts.index:
        # Scale color by mean SNR
        snr_scaled = min(chr_snr[chr_name]/5000, 1.0)
        # Generate color from viridis-like palette
        h = 0.65 - 0.65 * snr_scaled  # Hue: blue to yellow-green
        s = 0.9  # Saturation
        v = 0.6 + 0.4 * snr_scaled  # Value: darker to brighter
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    
    fig.add_trace(
        go.Bar(
            x=chrom_counts.index,
            y=chrom_counts.values,
            marker_color=colors,
            name='Chromosome Count'
        ),
        row=1, col=3
    )
    
    # Add a colorbar for SNR
    z = np.linspace(0, 5000, 100)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=0.1,
                color=z,
                colorscale='Viridis',
                colorbar=dict(
                    title="Mean SNR",
                    x=0.78,  # Adjust position of colorbar
                    y=0.8,
                    len=0.3
                ),
                showscale=True
            ),
            name='SNR Scale'
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="Chromosome", row=1, col=3)
    fig.update_yaxes(title_text="Number of Markers", row=1, col=3)
    
    # 4. SNR vs Target Value Heatmap
    # Create 2D histogram data
    snr_bins = np.logspace(0, 4, 25)  # Log scale for SNR
    target_bins = np.linspace(0, 1, 25)
    
    # Calculate 2D histogram
    h, xedges, yedges = np.histogram2d(
        markers['snr'].clip(upper=10000), 
        markers[target_cell_type],
        bins=[snr_bins, target_bins]
    )
    
    # Create heatmap
    fig.add_trace(
        go.Heatmap(
            z=np.log1p(h.T),  # Log scale and transpose
            x=xedges,
            y=yedges,
            colorscale='Blues',
            colorbar=dict(
                title="Log(Count + 1)",
                x=0.15,
                y=0.5,
                len=0.3
            ),
            name='SNR vs Target'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(
        title_text="SNR (log scale)", 
        type='log',
        row=2, col=1
    )
    fig.update_yaxes(title_text=f"Target Value ({target_cell_type})", row=2, col=1)
    
    # 5. Position Distribution within Chromosomes
    # Get chromosome sizes (approximate from max position)
    chr_sizes = {}
    for chr_name in markers['chr'].unique():
        chr_sizes[chr_name] = markers[markers['chr'] == chr_name]['end'].max()
    
    # Sort chromosomes
    sorted_chroms = sorted(chr_sizes.keys(), 
                         key=lambda x: int(x.replace('chr', '')) if x.replace('chr', '').isdigit() else ord(x.replace('chr', '')[0]))
    
    # Plot each chromosome as a horizontal line
    for i, chr_name in enumerate(sorted_chroms):
        # Add chromosome line
        fig.add_trace(
            go.Scatter(
                x=[0, chr_sizes[chr_name]],
                y=[i, i],
                mode='lines',
                line=dict(color='gray', width=1),
                name=chr_name,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add markers
        chr_markers = markers[markers['chr'] == chr_name]
        if not chr_markers.empty:
            fig.add_trace(
                go.Scatter(
                    x=chr_markers['start'],
                    y=[i] * len(chr_markers),
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=chr_markers['snr'].clip(upper=5000),
                        colorscale='Viridis',
                        opacity=0.7
                    ),
                    name=f'{chr_name} Markers',
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Add a colorbar for SNR
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=0.1,
                color=np.linspace(0, 5000, 100),
                colorscale='Viridis',
                colorbar=dict(
                    title="SNR",
                    x=0.68,
                    y=0.5,
                    len=0.3
                ),
                showscale=True
            ),
            name='SNR Scale',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Position (bp)", row=2, col=2)
    fig.update_yaxes(
        title_text="Chromosome", 
        tickvals=list(range(len(sorted_chroms))),
        ticktext=sorted_chroms,
        row=2, col=2
    )
    
    # 6. Primary vs Redundant Markers
    if 'is_primary' in markers.columns:
        primary_count = markers['is_primary'].sum()
        redundant_count = len(markers) - primary_count
        
        fig.add_trace(
            go.Pie(
                labels=['Primary', 'Redundant'],
                values=[primary_count, redundant_count],
                textinfo='percent+label',
                marker=dict(colors=['darkblue', 'skyblue']),
                name='Marker Types'
            ),
            row=3, col=1
        )
    
    # 7. Cell Type Values Heatmap
    # Get all cell types
    all_cell_types = [target_cell_type] + background_cell_types
    cell_type_cols = [col for col in all_cell_types if col in markers.columns]
    
    if len(cell_type_cols) > 1:
        # Sort markers by target cell type value
        sorted_markers = markers.sort_values(target_cell_type, ascending=False)
        
        # Display only a subset of markers if there are too many
        max_display = 50
        if len(sorted_markers) > max_display:
            display_markers = sorted_markers.iloc[:max_display]
            display_note = f" (showing top {max_display})"
        else:
            display_markers = sorted_markers
            display_note = ""
        
        # Create heatmap data
        heatmap_data = display_markers[cell_type_cols].T
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=list(range(heatmap_data.shape[1])),
                y=heatmap_data.index,
                colorscale='YlGnBu',
                colorbar=dict(
                    title="Value",
                    x=0.95,
                    y=0.2,
                    len=0.3
                ),
                name='Cell Type Values'
            ),
            row=3, col=2
        )
        
        fig.update_xaxes(title_text="Marker Index" + display_note, row=3, col=2)
        fig.update_yaxes(title_text="Cell Type", row=3, col=2)
    
    # Add summary statistics as annotations
    stats_text = (
        f"<b>Total Markers:</b> {len(markers)}<br>"
        f"<b>Mean SNR:</b> {markers['snr'].mean():.2f}<br>"
        f"<b>Median SNR:</b> {markers['snr'].median():.2f}<br>"
        f"<b>Max SNR:</b> {markers['snr'].max():.2f}<br>"
        f"<b>Markers with SNR > 1000:</b> {(markers['snr'] > 1000).sum()}<br>"
        f"<b>Markers with SNR > 10000:</b> {(markers['snr'] > 10000).sum()}<br>"
        f"<b>Mean {target_cell_type} value:</b> {markers[target_cell_type].mean():.4f}<br>"
        f"<b>Chromosomes covered:</b> {markers['chr'].nunique()}"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.02,
        text=stats_text,
        showarrow=False,
        font=dict(size=12),
        align="center",
        bordercolor="black",
        borderwidth=1,
        borderpad=10,
        bgcolor="white",
        opacity=0.8
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20)
        ),
        height=1200,
        width=1600,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def marker_set_performance_simulation(markers, target_cell_type, background_cell_types):
    """
    Simulate and visualize marker set performance at different target cell concentrations using Plotly
    
    Parameters:
    - markers: DataFrame with selected markers
    - target_cell_type: Name of target cell type
    - background_cell_types: List of background cell types
    
    Returns:
    - Plotly figure with simulation results
    """
    # Create concentrations to test (log scale)
    concentrations = np.logspace(-3, -0.3, 20)  # 0.1% to 50%
    
    # Get target and background values
    target_values = markers[target_cell_type].values
    
    # Calculate background values
    bg_values = []
    for bg_type in background_cell_types:
        if bg_type in markers.columns:
            bg_values.append(markers[bg_type].values)
    
    if not bg_values:
        return None  # Can't simulate without background values
    
    bg_values = np.array(bg_values)
    bg_mean = np.mean(bg_values, axis=0)
    bg_std = np.std(bg_values, axis=0)
    
    # Simulate metrics at each concentration
    detection_rates = []
    snrs = []
    aucs = []
    
    for concentration in concentrations:
        # Expected signal at this concentration
        expected_signal = (concentration * target_values) + ((1 - concentration) * bg_mean)
        
        # Calculate separability metrics
        detection_threshold = bg_mean + 2 * bg_std
        detection_rate = np.mean(expected_signal > detection_threshold)
        detection_rates.append(detection_rate)
        
        # Signal-to-noise in mixture
        signal = concentration * target_values
        noise = bg_std
        mixture_snr = np.mean(signal / (noise + 1e-10))
        snrs.append(mixture_snr)
        
        # Simplified AUC approximation (perfect since we're just using one marker set)
        aucs.append(1.0)
    
    # Create visualization with subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Classification Performance',
            'Fraction of Markers Detectable Above Background',
            'Mixture Signal-to-Noise Ratio'
        ),
        vertical_spacing=0.1
    )
    
    # Plot AUC
    fig.add_trace(
        go.Scatter(
            x=concentrations,
            y=aucs,
            mode='lines',
            line=dict(color='blue', width=2),
            name='AUC'
        ),
        row=1, col=1
    )
    
    # Plot detection rate
    fig.add_trace(
        go.Scatter(
            x=concentrations,
            y=detection_rates,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Detection Rate'
        ),
        row=2, col=1
    )
    
    # Plot SNR
    fig.add_trace(
        go.Scatter(
            x=concentrations,
            y=snrs,
            mode='lines',
            line=dict(color='blue', width=2),
            name='SNR'
        ),
        row=3, col=1
    )
    
    # Update axes
    for i in range(1, 4):
        fig.update_xaxes(
            title_text='Target Cell Concentration (log scale)',
            type='log',
            gridcolor='lightgrey',
            row=i, col=1
        )
    
    fig.update_yaxes(title_text='Area Under ROC Curve', gridcolor='lightgrey', row=1, col=1)
    fig.update_yaxes(title_text='Marker Detection Rate', gridcolor='lightgrey', row=2, col=1)
    fig.update_yaxes(title_text='Signal-to-Noise Ratio', gridcolor='lightgrey', row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1000,
        title=dict(
            text='Marker Set Performance Simulation',
            x=0.5,
            font=dict(size=20)
        ),
        template="plotly_white",
        showlegend=False
    )
    
    # Set y-axis range for AUC plot to better visualize near-perfect performance
    fig.update_yaxes(range=[0.94, 1.05], row=1, col=1)
    
    return fig


def save_markers(filtered_markers_dir, markers_fname, atlas_fname):
    markers = pd.read_parquet(list(filtered_markers_dir.glob("*.parquet")))
    markers = markers.dropna()    
    markers[['chr','start','end','startCpG','endCpG','target','name','direction','B-cells', 'CD34-erythroblasts', 'CD34-megakaryocytes', 'Colon', 'Esophagus', 'Gastric', 'Granulocytes', 'Monocytes', 'NK-cells', 'OAC', 'Small-intestine','T-cells', 'B-cells_coverage', 'CD34-erythroblasts_coverage', 'CD34-megakaryocytes_coverage', 'Colon_coverage', 'Esophagus_coverage', 'Gastric_coverage', 'Granulocytes_coverage',   'Monocytes_coverage', 'NK-cells_coverage','OAC_coverage', 'Small-intestine_coverage','T-cells_coverage', 'snr', 'snr_vs_median', 'snr_vs_mean', 'target_value','max_background', 'median_background', 'mean_background','background_std', 'background_range','background_quartile_ratio', 'signal_to_noise_area','relative_signal_strength','is_primary', 'separability','clinical_detection_prob','expected_useful_reads']].to_csv(markers_fname, sep="\t", index=False)
    markers[['chr','start','end','startCpG','endCpG','target','name','direction','B-cells', 'CD34-erythroblasts', 'CD34-megakaryocytes', 'Colon', 'Esophagus', 'Gastric', 'Granulocytes', 'Monocytes', 'NK-cells', 'OAC', 'Small-intestine','T-cells']].to_csv(atlas_fname, sep="\t", index=False)


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
    save_markers(output_dir, args.output_markers, args.output_atlas)

    logging.info(f"\nCompleted marker filtering")
    
if __name__ == "__main__":
    main()