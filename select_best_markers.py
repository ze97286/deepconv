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
    # Always try to read with header first
    atlas = pd.read_csv(atlas_path, sep='\t')
    
    # Convert numeric columns to proper types
    numeric_cols = ['start', 'end', 'startCpG', 'endCpG', 
                    'B-cells', 'CD34-erythroblasts', 'CD34-megakaryocytes', 
                    'Colon', 'Esophagus', 'Gastric', 'Granulocytes', 
                    'Monocytes', 'NK-cells', 'OAC', 'Small-intestine', 'T-cells']
    
    for col in numeric_cols:
        if col in atlas.columns:
            atlas[col] = pd.to_numeric(atlas[col], errors='coerce')
    
    return atlas

def compute_atlas_based_scores(atlas, target_cell_type):
    """
    Compute marker quality scores based purely on atlas characteristics
    """
    # Filter to target cell type markers
    target_atlas = atlas[atlas['target'] == target_cell_type].copy()
    
    n_markers = len(target_atlas)
    scores = np.zeros(n_markers)
    
    print(f"Computing atlas-based scores for {n_markers} {target_cell_type} markers...")
    
    # Get cell type columns
    cell_type_cols = ['B-cells', 'CD34-erythroblasts', 'CD34-megakaryocytes', 
                      'Colon', 'Esophagus', 'Gastric', 'Granulocytes', 
                      'Monocytes', 'NK-cells', 'OAC', 'Small-intestine', 'T-cells']
    
    # Filter to existing columns
    cell_type_cols = [col for col in cell_type_cols if col in target_atlas.columns]
    other_cell_types = [col for col in cell_type_cols if col != target_cell_type]
    
    print(f"Target column: {target_cell_type}")
    print(f"Other cell types: {other_cell_types}")
    
    for idx, (_, row) in enumerate(target_atlas.iterrows()):
        score = 0.0
        
        # 1. Target cell type signal strength
        if target_cell_type in row and pd.notna(row[target_cell_type]):
            target_signal = row[target_cell_type]
            score += 0.3 * min(abs(target_signal), 1.0)
        
        # 2. Specificity (high in target, low in others)
        if other_cell_types and target_cell_type in row:
            target_val = row[target_cell_type] if pd.notna(row[target_cell_type]) else 0
            other_vals = [row[col] for col in other_cell_types if col in row and pd.notna(row[col])]
            if other_vals:
                mean_other = np.mean(other_vals)
                # Higher difference = more specific
                specificity = target_val - mean_other
                score += 0.3 * max(0, min(specificity, 1.0))
        
        # 3. Region length (shorter regions might be more specific)
        if pd.notna(row['start']) and pd.notna(row['end']):
            region_length = row['end'] - row['start']
            # Normalize by typical CpG region length (assume 500bp is optimal)
            length_score = max(0, 1.0 - abs(region_length - 500) / 500)
            score += 0.2 * length_score
        
        # 4. Number of CpGs (if available)
        if 'startCpG' in row and 'endCpG' in row:
            if pd.notna(row['startCpG']) and pd.notna(row['endCpG']):
                num_cpgs = row['endCpG'] - row['startCpG']
                # Normalize (assume 10 CpGs is good)
                cpg_score = min(num_cpgs / 10.0, 1.0)
                score += 0.1 * cpg_score
        
        scores[idx] = score
    
    # 5. Chromosome distribution bonus
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