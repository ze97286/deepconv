import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple
import re

def apply_control_coverage_filter(signal_df: pd.DataFrame, 
                                coverage_df: pd.DataFrame,
                                xtp_min_coverage: int = 10,
                                gi_min_coverage: int = 4,
                                coverage_quorum: float = 0.8) -> pd.DataFrame:
    """
    Apply coverage filtering to control samples with different thresholds.
    X###/TP### controls need higher coverage than GI controls.
    """
    control_cols = [col for col in signal_df.columns if col.startswith('Control_')]
    
    if not control_cols:
        print("No control columns found!")
        return signal_df
    
    # Classify control samples
    xtp_controls = [col for col in control_cols if 
                   re.match(r'Control_X\d+', col) or re.match(r'Control_TP\d+', col)]
    gi_controls = [col for col in control_cols if 'GI' in col]
    other_controls = [col for col in control_cols if col not in xtp_controls and col not in gi_controls]
    
    print(f"\nControl sample classification:")
    print(f"  X###/TP### controls: {len(xtp_controls)}")
    if xtp_controls:
        print(f"    Samples: {', '.join(sorted(xtp_controls)[:5])}{' ...' if len(xtp_controls) > 5 else ''}")
    print(f"  GI controls: {len(gi_controls)}")
    if gi_controls:
        print(f"    Samples: {', '.join(sorted(gi_controls)[:5])}{' ...' if len(gi_controls) > 5 else ''}")
    print(f"  Other controls: {len(other_controls)}")
    if other_controls:
        print(f"    Samples: {', '.join(sorted(other_controls)[:5])}{' ...' if len(other_controls) > 5 else ''}")
    
    # Apply coverage filtering
    coverage_mask = pd.Series(True, index=signal_df.index)
    
    # X###/TP### controls - require quorum to have sufficient coverage
    if xtp_controls:
        xtp_coverage_ok = (coverage_df[xtp_controls] >= xtp_min_coverage).sum(axis=1)
        xtp_mask = xtp_coverage_ok >= (len(xtp_controls) * coverage_quorum)
        print(f"\n  X###/TP### coverage ≥{xtp_min_coverage} (≥{coverage_quorum:.0%} of samples):")
        print(f"    Passing: {xtp_mask.sum()}/{len(xtp_mask)} ({100*xtp_mask.sum()/len(xtp_mask):.1f}%)")
        coverage_mask &= xtp_mask
    
    # GI controls - require quorum to have sufficient coverage
    if gi_controls:
        gi_coverage_ok = (coverage_df[gi_controls] >= gi_min_coverage).sum(axis=1)
        gi_mask = gi_coverage_ok >= (len(gi_controls) * coverage_quorum)
        print(f"  GI coverage ≥{gi_min_coverage} (≥{coverage_quorum:.0%} of samples):")
        print(f"    Passing: {gi_mask.sum()}/{len(gi_mask)} ({100*gi_mask.sum()/len(gi_mask):.1f}%)")
        coverage_mask &= gi_mask
    
    # Other controls - use GI threshold
    if other_controls:
        other_coverage_ok = (coverage_df[other_controls] >= gi_min_coverage).sum(axis=1)
        other_mask = other_coverage_ok >= (len(other_controls) * coverage_quorum)
        print(f"  Other controls coverage ≥{gi_min_coverage} (≥{coverage_quorum:.0%} of samples):")
        print(f"    Passing: {other_mask.sum()}/{len(other_mask)} ({100*other_mask.sum()/len(other_mask):.1f}%)")
        coverage_mask &= other_mask
    
    print(f"\n  Combined coverage filter: {coverage_mask.sum()}/{len(coverage_mask)} ({100*coverage_mask.sum()/len(coverage_mask):.1f}%) regions pass")
    
    return signal_df[coverage_mask].copy(), coverage_df[coverage_mask].copy()

def apply_control_signal_filter(signal_df: pd.DataFrame,
                              xtp_max_signal: float = 0.001,
                              gi_max_signal: float = None,
                              other_max_signal: float = None) -> pd.DataFrame:
    """
    Apply signal filtering to control samples.
    Primary focus on X###/TP### controls being < threshold.
    """
    control_cols = [col for col in signal_df.columns if col.startswith('Control_')]
    
    if not control_cols:
        print("No control columns found!")
        return signal_df
    
    # Classify control samples
    xtp_controls = [col for col in control_cols if 
                   re.match(r'Control_X\d+', col) or re.match(r'Control_TP\d+', col)]
    gi_controls = [col for col in control_cols if 'GI' in col]
    other_controls = [col for col in control_cols if col not in xtp_controls and col not in gi_controls]
    
    # Calculate max signal for each control type
    signal_mask = pd.Series(True, index=signal_df.index)
    
    if xtp_controls:
        xtp_max = signal_df[xtp_controls].max(axis=1)
        xtp_median = signal_df[xtp_controls].median(axis=1)
        
        print(f"\nX###/TP### control signal distribution:")
        print(f"  Max signal - min: {xtp_max.min():.6f}, max: {xtp_max.max():.6f}")
        print(f"  Max signal - percentiles: 25th={xtp_max.quantile(0.25):.6f}, 50th={xtp_max.quantile(0.5):.6f}, 75th={xtp_max.quantile(0.75):.6f}")
        print(f"  Median signal - min: {xtp_median.min():.6f}, max: {xtp_median.max():.6f}")
        print(f"  Median signal - percentiles: 25th={xtp_median.quantile(0.25):.6f}, 50th={xtp_median.quantile(0.5):.6f}, 75th={xtp_median.quantile(0.75):.6f}")
        
        # Apply filter
        xtp_pass = xtp_max <= xtp_max_signal
        print(f"\n  Regions with all X###/TP### < {xtp_max_signal}: {xtp_pass.sum()}/{len(xtp_pass)} ({100*xtp_pass.sum()/len(xtp_pass):.1f}%)")
        signal_mask &= xtp_pass
        
        # Show what we'd get with different thresholds
        print(f"\n  Alternative thresholds for X###/TP### (max signal):")
        for thresh in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            passing = (xtp_max <= thresh).sum()
            print(f"    <= {thresh}: {passing} regions ({100*passing/len(xtp_max):.1f}%)")
    
    if gi_controls and gi_max_signal is not None:
        gi_max = signal_df[gi_controls].max(axis=1)
        gi_pass = gi_max <= gi_max_signal
        print(f"\n  GI control signal filter:")
        print(f"    Max signal <= {gi_max_signal}: {gi_pass.sum()}/{len(gi_pass)} ({100*gi_pass.sum()/len(gi_pass):.1f}%)")
        signal_mask &= gi_pass
    
    if other_controls and other_max_signal is not None:
        other_max = signal_df[other_controls].max(axis=1)
        other_pass = other_max <= other_max_signal
        print(f"\n  Other control signal filter:")
        print(f"    Max signal <= {other_max_signal}: {other_pass.sum()}/{len(other_pass)} ({100*other_pass.sum()/len(other_pass):.1f}%)")
        signal_mask &= other_pass
    
    return signal_df[signal_mask].copy()

def analyze_region_lengths(atlas_df: pd.DataFrame):
    """Analyze the length distribution of regions."""
    atlas_df['length'] = atlas_df['end'] - atlas_df['start']
    
    print("\nRegion length analysis:")
    print(f"  Min: {atlas_df['length'].min()} bp")
    print(f"  25th percentile: {atlas_df['length'].quantile(0.25):.0f} bp")
    print(f"  Median: {atlas_df['length'].median():.0f} bp")
    print(f"  75th percentile: {atlas_df['length'].quantile(0.75):.0f} bp") 
    print(f"  Max: {atlas_df['length'].max()} bp")
    print(f"  Mean: {atlas_df['length'].mean():.1f} bp")
    
    # Length distribution
    print("\nLength distribution:")
    bins = [0, 100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, float('inf')]
    labels = ['<100', '100-200', '200-300', '300-400', '400-500', '500-1k', '1k-2k', '2k-5k', '5k-10k', '>10k']
    atlas_df['length_bin'] = pd.cut(atlas_df['length'], bins=bins, labels=labels, right=False)
    length_dist = atlas_df['length_bin'].value_counts()
    for bin_label in labels:
        count = length_dist.get(bin_label, 0)
        print(f"  {bin_label} bp: {count} regions ({100*count/len(atlas_df):.1f}%)")
    
    # Drop temporary columns
    atlas_df.drop(['length', 'length_bin'], axis=1, inplace=True)

def sort_atlas_by_position(atlas_df: pd.DataFrame) -> pd.DataFrame:
    """Sort atlas by chromosome (numeric 1-22, then X, Y) and start position."""
    def chromosome_sort_key(chr_str):
        """Convert chromosome to sortable format"""
        chr_clean = str(chr_str).replace('chr', '')
        if chr_clean.isdigit():
            return (0, int(chr_clean))
        elif chr_clean == 'X':
            return (1, 0)
        elif chr_clean == 'Y':
            return (1, 1)
        else:
            return (2, chr_clean)
    
    # Add sort key column
    atlas_df['chr_sort'] = atlas_df['chr'].apply(chromosome_sort_key)
    
    # Sort by chromosome and position
    atlas_df = atlas_df.sort_values(['chr_sort', 'start']).drop('chr_sort', axis=1)
    
    # Reset index
    atlas_df = atlas_df.reset_index(drop=True)
    
    return atlas_df

def main():
    parser = argparse.ArgumentParser(description='Apply control specificity filtering to atlas regions')
    parser.add_argument('--atlas', type=str, required=True, 
                       help='Input atlas file from differential methylation filtering')
    parser.add_argument('--control_signal_file', type=str, required=True,
                       help='Control signal matrix parquet file')
    parser.add_argument('--control_coverage_file', type=str, required=True,
                       help='Control coverage matrix parquet file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output filtered atlas file')
    
    # Coverage thresholds
    parser.add_argument('--xtp_min_coverage', type=int, default=10,
                       help='Minimum coverage for X###/TP### controls')
    parser.add_argument('--gi_min_coverage', type=int, default=4,
                       help='Minimum coverage for GI controls')
    parser.add_argument('--coverage_quorum', type=float, default=0.8,
                       help='Fraction of control samples that must pass coverage')
    
    # Signal thresholds
    parser.add_argument('--xtp_max_signal', type=float, default=0.001,
                       help='Maximum signal allowed in X###/TP### controls')
    parser.add_argument('--gi_max_signal', type=float, default=None,
                       help='Maximum signal allowed in GI controls (optional)')
    parser.add_argument('--other_max_signal', type=float, default=None,
                       help='Maximum signal allowed in other controls (optional)')
    
    args = parser.parse_args()
    
    # Load atlas
    print(f"Loading atlas from {args.atlas}...")
    atlas_df = pd.read_csv(args.atlas, sep='\t')
    print(f"Loaded {len(atlas_df)} regions")
    
    # Analyze initial region lengths
    analyze_region_lengths(atlas_df.copy())
    
    # Load control data
    print(f"\nLoading control signal matrix from {args.control_signal_file}...")
    control_signal = pd.read_parquet(args.control_signal_file)
    
    print(f"Loading control coverage matrix from {args.control_coverage_file}...")
    control_coverage = pd.read_parquet(args.control_coverage_file)
    
    # Ensure we're working with the same regions
    common_indices = atlas_df.index.intersection(control_signal.index).intersection(control_coverage.index)
    print(f"\nFound {len(common_indices)} common regions across all datasets")
    
    if len(common_indices) < len(atlas_df):
        print(f"WARNING: {len(atlas_df) - len(common_indices)} atlas regions not found in control data!")
    
    # Subset to common regions
    atlas_subset = atlas_df.loc[common_indices].copy()
    control_signal_subset = control_signal.loc[common_indices].copy()
    control_coverage_subset = control_coverage.loc[common_indices].copy()
    
    # Add atlas metadata to control signal
    metadata_cols = ['chr', 'start', 'end', 'startCpG', 'endCpG', 'name', 'direction', 'target']
    for col in metadata_cols:
        if col in atlas_subset.columns:
            control_signal_subset[col] = atlas_subset[col]
    
    # Apply coverage filtering
    print("\n" + "="*60)
    print("APPLYING CONTROL COVERAGE FILTER")
    print("="*60)
    filtered_signal, filtered_coverage = apply_control_coverage_filter(
        control_signal_subset, 
        control_coverage_subset,
        xtp_min_coverage=args.xtp_min_coverage,
        gi_min_coverage=args.gi_min_coverage,
        coverage_quorum=args.coverage_quorum
    )
    
    # Apply signal filtering
    print("\n" + "="*60)
    print("APPLYING CONTROL SIGNAL FILTER")
    print("="*60)
    final_filtered = apply_control_signal_filter(
        filtered_signal,
        xtp_max_signal=args.xtp_max_signal,
        gi_max_signal=args.gi_max_signal,
        other_max_signal=args.other_max_signal
    )
    
    # Prepare final atlas
    print("\n" + "="*60)
    print("PREPARING FINAL ATLAS")
    print("="*60)
    
    # Get the atlas rows that passed filtering
    final_indices = final_filtered.index
    final_atlas = atlas_subset.loc[final_indices].copy()
    
    # Sort by chromosome and position
    final_atlas = sort_atlas_by_position(final_atlas)
    
    print(f"\nFinal atlas: {len(final_atlas)} regions")
    
    # Analyze final region lengths
    print("\nFinal region length analysis:")
    analyze_region_lengths(final_atlas.copy())
    
    # Chromosome distribution
    print("\nChromosome distribution:")
    chr_counts = final_atlas['chr'].value_counts()
    for chr_name in sorted(chr_counts.index, key=lambda x: (0, int(x.replace('chr', ''))) if x.replace('chr', '').isdigit() else (1, x)):
        print(f"  {chr_name}: {chr_counts[chr_name]} regions")
    
    # Save final atlas
    final_atlas.to_csv(args.output_file, sep='\t', index=False)
    print(f"\nSaved {len(final_atlas)} filtered regions to {args.output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"Initial regions: {len(atlas_df)}")
    print(f"After coverage filter: {len(filtered_signal)} ({100*len(filtered_signal)/len(atlas_df):.1f}%)")
    print(f"After signal filter: {len(final_atlas)} ({100*len(final_atlas)/len(atlas_df):.1f}%)")

if __name__ == "__main__":
    main()