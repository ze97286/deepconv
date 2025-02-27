import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd

def analyze_marker_overlaps(df):
    """
    Analyze overlapping markers in the atlas.
    
    Args:
        df: DataFrame with columns ['target', 'startCpG', 'endCpG']
    
    Returns:
        Dictionary with overlap statistics per target
    """
    results = {}
    
    # For each target, analyze its markers
    for target in df['target'].unique():
        target_markers = df[df['target'] == target].copy()
        
        # Sort by start position for easier overlap detection
        target_markers = target_markers.sort_values('startCpG')
        
        overlaps = []
        n_markers = len(target_markers)
        
        # Compare each marker with subsequent markers
        for i in range(n_markers):
            current = target_markers.iloc[i]
            
            # Look at all subsequent markers for overlaps
            for j in range(i + 1, n_markers):
                next_marker = target_markers.iloc[j]
                
                # If next marker starts after current ends, no more overlaps possible
                if next_marker['startCpG'] > current['endCpG']:
                    break
                
                # Calculate overlap
                overlap_start = max(current['startCpG'], next_marker['startCpG'])
                overlap_end = min(current['endCpG'], next_marker['endCpG'])
                overlap_length = overlap_end - overlap_start
                
                if overlap_length > 0:
                    overlaps.append({
                        'marker1_start': current['startCpG'],
                        'marker1_end': current['endCpG'],
                        'marker2_start': next_marker['startCpG'],
                        'marker2_end': next_marker['endCpG'],
                        'overlap_length': overlap_length,
                        'marker1_length': current['endCpG'] - current['startCpG'],
                        'marker2_length': next_marker['endCpG'] - next_marker['startCpG']
                    })
        
        # Calculate statistics
        if overlaps:
            overlap_df = pd.DataFrame(overlaps)
            
            results[target] = {
                'total_markers': n_markers,
                'markers_with_overlap': len(set(overlap_df['marker1_start'].tolist() + 
                                              overlap_df['marker2_start'].tolist())),
                'num_overlaps': len(overlaps),
                'mean_overlap_length': overlap_df['overlap_length'].mean(),
                'max_overlap_length': overlap_df['overlap_length'].max(),
                'mean_overlap_ratio': (overlap_df['overlap_length'] / 
                                     overlap_df[['marker1_length', 'marker2_length']].min(axis=1)).mean(),
                'overlapping_pairs': overlaps
            }
        else:
            results[target] = {
                'total_markers': n_markers,
                'markers_with_overlap': 0,
                'num_overlaps': 0,
                'mean_overlap_length': 0,
                'max_overlap_length': 0,
                'mean_overlap_ratio': 0,
                'overlapping_pairs': []
            }
    
    # Calculate overall statistics
    total_markers = sum(r['total_markers'] for r in results.values())
    total_overlapping = sum(r['markers_with_overlap'] for r in results.values())
    
    results['overall'] = {
        'total_markers': total_markers,
        'total_overlapping_markers': total_overlapping,
        'percent_overlapping': (total_overlapping / total_markers) * 100 if total_markers > 0 else 0
    }
    
    return results

def print_overlap_summary(results):
    """
    Print a readable summary of the overlap analysis.
    """
    print("\nOverall Statistics:")
    print(f"Total markers: {results['overall']['total_markers']}")
    print(f"Total markers with overlaps: {results['overall']['total_overlapping_markers']}")
    print(f"Percent markers overlapping: {results['overall']['percent_overlapping']:.2f}%")
    
    print("\nPer-target Statistics:")
    for target, stats in results.items():
        if target != 'overall':
            print(f"\n{target}:")
            print(f"  Total markers: {stats['total_markers']}")
            if stats['num_overlaps'] > 0:
                print(f"  Markers with overlaps: {stats['markers_with_overlap']}")
                print(f"  Number of overlapping pairs: {stats['num_overlaps']}")
                print(f"  Mean overlap length: {stats['mean_overlap_length']:.2f}")
                print(f"  Max overlap length: {stats['max_overlap_length']}")
                print(f"  Mean overlap ratio: {stats['mean_overlap_ratio']:.2f}")
            else:
                print("  No overlapping markers")




def visualise_overlap_stats(results):
    # Convert results to DataFrame for easier plotting
    stats = []
    for target, data in results.items():
        if target != 'overall':
            stats.append({
                'target': target,
                'total_markers': data['total_markers'],
                'overlapping_markers': data['markers_with_overlap'],
                'num_overlaps': data['num_overlaps'],
                'mean_overlap_length': data['mean_overlap_length'],
                'max_overlap_length': data['max_overlap_length'],
                'mean_overlap_ratio': data['mean_overlap_ratio']
            })
    
    df = pd.DataFrame(stats)
    
    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Marker Counts per Cell Type',
            'Overlap Lengths',
            'Number of Overlapping Pairs',
            'Mean Overlap Ratio'
        )
    )
    
    # 1. Marker counts
    fig.add_trace(
        go.Bar(
            name='Total Markers',
            x=df['target'],
            y=df['total_markers'],
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name='Overlapping Markers',
            x=df['target'],
            y=df['overlapping_markers'],
            marker_color='orange'
        ),
        row=1, col=1
    )
    
    # 2. Overlap lengths
    fig.add_trace(
        go.Bar(
            name='Mean Overlap Length',
            x=df['target'],
            y=df['mean_overlap_length'],
            marker_color='lightgreen'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            name='Max Overlap Length',
            x=df['target'],
            y=df['max_overlap_length'],
            marker_color='darkgreen'
        ),
        row=1, col=2
    )
    
    # 3. Number of overlapping pairs
    fig.add_trace(
        go.Bar(
            name='Overlapping Pairs',
            x=df['target'],
            y=df['num_overlaps'],
            marker_color='purple'
        ),
        row=2, col=1
    )
    
    # 4. Mean overlap ratio
    fig.add_trace(
        go.Bar(
            name='Mean Overlap Ratio',
            x=df['target'],
            y=df['mean_overlap_ratio'],
            marker_color='red'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Marker Overlap Analysis by Cell Type",
        barmode='group'
    )
    
    # Update x-axis for all subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(tickangle=45, row=i, col=j)
    
    # Add a text annotation with overall statistics
    overall_stats = (
        f"Overall Statistics:\n"
        f"Total Markers: {results['overall']['total_markers']}\n"
        f"Total Overlapping Markers: {results['overall']['total_overlapping_markers']}\n"
        f"Percent Overlapping: {results['overall']['percent_overlapping']:.1f}%"
    )
    
    fig.add_annotation(
        text=overall_stats,
        xref="paper", yref="paper",
        x=0.5, y=1.1,
        showarrow=False,
        font=dict(size=12),
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        opacity=0.8
    )
    
    return fig


def visualise_overlap_distribution(results):
    # Collect all overlap lengths
    all_overlaps = []
    for target, data in results.items():
        if target != 'overall':
            for pair in data['overlapping_pairs']:
                all_overlaps.append({
                    'target': target,
                    'overlap_length': pair['overlap_length'],
                    'ratio': pair['overlap_length'] / min(pair['marker1_length'], pair['marker2_length'])
                })
    
    df = pd.DataFrame(all_overlaps)
    
    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Distribution of Overlap Lengths',
            'Distribution of Overlap Ratios'
        )
    )
    
    # Histogram of overlap lengths
    fig.add_trace(
        go.Histogram(
            x=df['overlap_length'],
            name='Overlap Length',
            nbinsx=30,
            marker_color='blue'
        ),
        row=1, col=1
    )
    
    # Histogram of overlap ratios
    fig.add_trace(
        go.Histogram(
            x=df['ratio'],
            name='Overlap Ratio',
            nbinsx=30,
            marker_color='red'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="Distribution of Marker Overlaps"
    )
    
    return fig


def analyse_atlas(atlas_path, out_path):
    atlas = pd.read_csv(atlas_path, sep="\t")
    results = analyze_marker_overlaps(atlas)
    fig1 = visualise_overlap_stats(results)
    fig1.write_html(out_path+"/overlap_stats.html")
    fig1.write_image(out_path+"/overlap_stats.png")
    fig2 = visualise_overlap_distribution(results)
    fig2.write_html(out_path+"/overlap_dist.html")
    fig2.write_image(out_path+"/overlap_dist.png")

