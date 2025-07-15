#!/usr/bin/env python3
"""
Select the best K markers from the atlas based on atlas characteristics only
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys
import os

def load_atlas(atlas_path):
    """Load atlas file and return DataFrame"""
    if atlas_path.endswith('.bed'):
        # BED format: chr, start, end, name, score, strand, ...
        atlas = pd.read_csv(atlas_path, sep='\t', header=None)
        # Assume standard BED format columns
        atlas.columns = ['chr', 'start', 'end', 'name', 'score', 'strand'] + [f'col_{i}' for i in range(6, len(atlas.columns))]
    else:
        # CSV/TSV format
        atlas = pd.read_csv(atlas_path, sep='\t')
    
    return atlas

def compute_atlas_based_scores(atlas, target_cell_type):
    """
    Compute marker quality scores based purely on atlas characteristics
    """
    # Filter to target cell type markers
    target_atlas = atlas[atlas['target'] == target_cell_type].copy() if 'target' in atlas.columns else atlas.copy()
    
    n_markers = len(target_atlas)
    scores = np.zeros(n_markers)
    
    print(f"Computing atlas-based scores for {n_markers} {target_cell_type} markers...")
    
    # Find atlas columns that likely contain signal information
    signal_cols = []
    for col in target_atlas.columns:
        if target_cell_type.lower() in col.lower():
            signal_cols.append(col)
    
    if not signal_cols:
        print("Warning: No target-specific signal columns found in atlas")
        # Use all numeric columns as potential signal
        signal_cols = target_atlas.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Using signal columns: {signal_cols}")
    
    for idx, (_, row) in enumerate(target_atlas.iterrows()):
        score = 0.0
        
        # 1. Signal strength (mean signal across conditions)
        if signal_cols:
            signal_values = [row[col] for col in signal_cols if pd.notna(row[col])]
            if signal_values:
                signal_strength = np.mean(signal_values)
                score += 0.4 * min(signal_strength, 1.0)  # Cap at 1.0
        
        # 2. Background signal (if available)
        background_cols = [col for col in target_atlas.columns if 'background' in col.lower() or 'control' in col.lower()]
        if background_cols:
            background_values = [row[col] for col in background_cols if pd.notna(row[col])]
            if background_values:
                background_signal = np.mean(background_values)
                # Lower background is better
                score += 0.2 * max(0, 1.0 - background_signal)
        
        # 3. Region length (shorter regions might be more specific)
        if 'start' in row and 'end' in row:
            region_length = row['end'] - row['start']
            # Normalize by typical CpG region length (assume 1kb is optimal)
            length_score = max(0, 1.0 - abs(region_length - 1000) / 1000)
            score += 0.1 * length_score
        
        # 4. Chromosome distribution bonus (spread across chromosomes is good)
        # This will be computed globally after all markers are scored
        
        # 5. Coverage-related score (if available in atlas)
        coverage_cols = [col for col in target_atlas.columns if 'coverage' in col.lower() or 'depth' in col.lower()]
        if coverage_cols:
            coverage_values = [row[col] for col in coverage_cols if pd.notna(row[col])]
            if coverage_values:
                coverage_score = min(np.mean(coverage_values) / 10.0, 1.0)  # Normalize
                score += 0.1 * coverage_score
        
        # 6. Variability/SNR (if multiple conditions available)
        if len(signal_cols) > 1:
            signal_values = [row[col] for col in signal_cols if pd.notna(row[col])]
            if len(signal_values) > 1:
                signal_var = np.var(signal_values)
                signal_mean = np.mean(signal_values)
                if signal_mean > 0:
                    cv = signal_var / signal_mean  # Coefficient of variation
                    score += 0.2 * min(cv, 1.0)
        
        scores[idx] = score
    
    # 4. Chromosome distribution bonus
    if 'chr' in target_atlas.columns:
        chr_counts = target_atlas['chr'].value_counts()
        for idx, (_, row) in enumerate(target_atlas.iterrows()):
            chr_name = row['chr']
            # Bonus for chromosomes with fewer markers (better distribution)
            chr_bonus = 0.1 * max(0, 1.0 - chr_counts[chr_name] / len(target_atlas))
            scores[idx] += chr_bonus
    
    return scores, target_atlas

def select_best_markers_from_atlas(atlas_path, target_cell_type, top_k=1000):
    """
    Select the top K markers based purely on atlas characteristics
    """
    print(f"Loading atlas from {atlas_path}")
    atlas = load_atlas(atlas_path)
    
    print(f"Atlas shape: {atlas.shape}")
    print(f"Atlas columns: {list(atlas.columns)}")
    
    # Compute scores
    scores, target_atlas = compute_atlas_based_scores(atlas, target_cell_type)
    
    # Get top K markers
    top_indices = np.argsort(scores)[-top_k:][::-1]  # Descending order
    top_scores = scores[top_indices]
    
    # Select top markers from atlas
    selected_atlas = target_atlas.iloc[top_indices].copy()
    selected_atlas['selection_score'] = top_scores
    
    print(f"\nSelected {len(selected_atlas)} markers:")
    print(f"Score range: {top_scores.min():.4f} - {top_scores.max():.4f}")
    print(f"Mean score: {top_scores.mean():.4f}")
    
    # Print top 10 for inspection
    print(f"\nTop 10 markers:")
    for i in range(min(10, len(selected_atlas))):
        row = selected_atlas.iloc[i]
        score = top_scores[i]
        marker_name = row.get('name', f'marker_{i}')
        print(f"  {marker_name}: score={score:.4f}")
    
    return selected_atlas

def main():
    parser = argparse.ArgumentParser(description='Select best markers from atlas')
    parser.add_argument('--atlas_path', type=str, required=True, 
                       help='Path to atlas file')
    parser.add_argument('--target_cell_type', type=str, required=True, 
                       help='Target cell type')
    parser.add_argument('--top_k', type=int, default=1000, 
                       help='Number of top markers to select')
    parser.add_argument('--output', type=str, default='best_markers_atlas.bed',
                       help='Output filtered atlas file')
    
    args = parser.parse_args()
    
    # Select best markers from atlas
    selected_atlas = select_best_markers_from_atlas(
        args.atlas_path, args.target_cell_type, args.top_k
    )
    
    # Save filtered atlas
    if args.output.endswith('.bed'):
        # Save as BED format
        selected_atlas.to_csv(args.output, sep='\t', header=False, index=False)
    else:
        # Save as TSV format
        selected_atlas.to_csv(args.output, sep='\t', index=False)
    
    print(f"\nSaved filtered atlas with {len(selected_atlas)} markers to {args.output}")
    
    # Also save a summary
    summary_file = args.output.replace('.bed', '_summary.txt').replace('.tsv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Atlas marker selection summary\n")
        f.write(f"Original atlas: {args.atlas_path}\n")
        f.write(f"Target cell type: {args.target_cell_type}\n")
        f.write(f"Selected markers: {len(selected_atlas)}\n")
        f.write(f"Score range: {selected_atlas['selection_score'].min():.4f} - {selected_atlas['selection_score'].max():.4f}\n")
        f.write(f"Mean score: {selected_atlas['selection_score'].mean():.4f}\n")
        
        # Chromosome distribution
        if 'chr' in selected_atlas.columns:
            chr_counts = selected_atlas['chr'].value_counts()
            f.write(f"\nChromosome distribution:\n")
            for chr_name, count in chr_counts.head(10).items():
                f.write(f"  {chr_name}: {count} markers\n")
    
    print(f"Saved summary to {summary_file}")

if __name__ == "__main__":
    main()