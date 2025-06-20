import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import glob

def test_broad_filter_thresholds(df):
    """
    Test the broad filter parameters to see how many regions remain
    """
    print("=== BROAD FILTER CASCADE ANALYSIS ===")
    print(f"Starting regions: {len(df):,}")
    
    # Define blood cell types
    blood_cells = ['Granulocytes', 'T-cells', 'B-cells', 'NK-cells', 'Monocytes']
    
    # Add computed columns
    df = df.copy()
    df['region_length'] = df['end'] - df['start']
    df['n_cpgs'] = df['endCpG'] - df['startCpG']
    
    # Calculate median blood background
    df['median_blood_background'] = df[blood_cells].median(axis=1)
    
    # Progressive filtering
    print("\n--- Progressive Filtering ---")
    
    # 1. Cancer signal
    step1 = df[df['OAC'] >= 0.5]
    print(f"After OAC ≥ 0.5: {len(step1):,} ({len(step1)/len(df)*100:.1f}%)")
    
    # 2. Coverage
    step2 = step1[step1['OAC_coverage'] >= 100]
    print(f"After OAC_coverage ≥ 100: {len(step2):,} ({len(step2)/len(df)*100:.1f}%)")
    
    # 3. Region length
    step3 = step2[step2['region_length'] <= 350]
    print(f"After region_length ≤ 350bp: {len(step3):,} ({len(step3)/len(df)*100:.1f}%)")
    
    # 4. CpG count
    step4 = step3[(step3['n_cpgs'] >= 4) & (step3['n_cpgs'] <= 20)]
    print(f"After 4 ≤ CpGs ≤ 20: {len(step4):,} ({len(step4)/len(df)*100:.1f}%)")
    
    # 5. Blood background - test different thresholds
    print("\n--- Blood Background Thresholds ---")
    for threshold in [0.02, 0.015, 0.01, 0.005]:
        step5 = step4[step4['median_blood_background'] < threshold]
        print(f"Median blood < {threshold:.1%}: {len(step5):,} regions")
        
        if len(step5) > 0:
            # Additional statistics
            print(f"  Mean OAC signal: {step5['OAC'].mean():.3f}")
            print(f"  Mean median_blood: {step5['median_blood_background'].mean():.4f}")
            print(f"  Mean region length: {step5['region_length'].mean():.1f}bp")
    
    # Check final selection with 1% threshold
    final = step4[step4['median_blood_background'] < 0.01]
    
    if len(final) > 0:
        print(f"\n--- Final Selection Analysis (blood < 1%) ---")
        print(f"Total regions: {len(final):,}")
        print(f"Chromosomes covered: {final['chr'].nunique()}")
        print(f"Mean signal-to-blood ratio: {(final['OAC'] / (final['median_blood_background'] + 0.001)).mean():.1f}")
        
        # Check individual blood cell contributions
        print("\nBlood cell background distribution:")
        for cell in blood_cells:
            print(f"  {cell}: mean={final[cell].mean():.4f}, >1%: {(final[cell] > 0.01).sum()}")
    
    return final

def analyze_controls_for_filtered_regions(filtered_regions, control_mv, control_cov):
    """
    Analyze control data for the 609K filtered regions
    """
    print("=== CONTROL ANALYSIS FOR FILTERED REGIONS ===")
    # Define control groups and thresholds
    plasma_controls = ['TP277_Ctrl_plasma_md', 'X2881_Ctrl_plasma_md', 
                      'X3161_Ctrl_plasma_md', 'X3421_Ctrl_plasma_md']
    gi_controls = ['GI10873', 'GI10881', 'GI10882', 'GI10887', 'GI10888', 'GI9774']
    all_controls = plasma_controls + gi_controls
    # Get regions that have control data
    merged = filtered_regions.merge(
        control_mv[['name', 'direction'] + all_controls],
        on=['name', 'direction'],
        how='inner'
    )
    # Add coverage data
    merged = merged.merge(
        control_cov[['name', 'direction'] + all_controls],
        on=['name', 'direction'],
        how='left',
        suffixes=('', '_coverage')
    )
    print(f"Filtered regions with control data: {len(merged):,} of {len(filtered_regions):,}")
    # Apply coverage thresholds and set unreliable values to NA
    for control in all_controls:
        cov_col = f'{control}_coverage'
        threshold = 10 if control in plasma_controls else 5
        # Set to NA where coverage is insufficient
        mask = merged[cov_col] < threshold
        merged.loc[mask, control] = np.nan
        # Report coverage statistics
        valid = merged[cov_col] >= threshold
        print(f"{control}: {valid.sum():,} regions with ≥{threshold} reads ({valid.sum()/len(merged)*100:.1f}%)")
    # Calculate control statistics (excluding NA values)
    merged['n_valid_plasma'] = merged[plasma_controls].notna().sum(axis=1)
    merged['n_valid_gi'] = merged[gi_controls].notna().sum(axis=1)
    merged['n_valid_total'] = merged['n_valid_plasma'] + merged['n_valid_gi']
    # Calculate max and mean for valid values only
    merged['max_control_plasma'] = merged[plasma_controls].max(axis=1, skipna=True)
    merged['max_control_gi'] = merged[gi_controls].max(axis=1, skipna=True)
    merged['max_control_all'] = merged[all_controls].max(axis=1, skipna=True)
    merged['mean_control_plasma'] = merged[plasma_controls].mean(axis=1, skipna=True)
    merged['mean_control_gi'] = merged[gi_controls].mean(axis=1, skipna=True)
    merged['mean_control_all'] = merged[all_controls].mean(axis=1, skipna=True)
    # Define tiers
    print("\n=== TIER ANALYSIS ===")
    
    # Tier 1: ≥3 valid controls, all = 0%
    tier1 = merged[
        (merged['n_valid_total'] >= 3) & 
        (merged['max_control_all'] == 0) &
        (merged['OAC'] >= 0.7)
    ]
    print(f"Tier 1 (≥3 valid, all 0%, OAC≥0.7): {len(tier1):,} regions")
    # Tier 2: ≥2 valid controls, all < 0.5%
    tier2 = merged[
        (merged['n_valid_total'] >= 2) & 
        (merged['max_control_all'] < 0.005) &
        (merged['OAC'] >= 0.6)
    ]
    print(f"Tier 2 (≥2 valid, all <0.5%, OAC≥0.6): {len(tier2):,} regions")
    # Tier 3: ≥2 valid controls, all < 1%
    tier3 = merged[
        (merged['n_valid_total'] >= 2) & 
        (merged['max_control_all'] < 0.01) &
        (merged['OAC'] >= 0.5)
    ]
    print(f"Tier 3 (≥2 valid, all <1%, OAC≥0.5): {len(tier3):,} regions")
    # Analysis by control type
    print("\n=== ANALYSIS BY CONTROL TYPE ===")
    # Regions with only plasma data
    plasma_only = merged[
        (merged['n_valid_plasma'] >= 2) & 
        (merged['n_valid_gi'] == 0)
    ]
    print(f"Regions with only plasma data (≥2 valid): {len(plasma_only):,}")
    # Regions with both types
    both_types = merged[
        (merged['n_valid_plasma'] >= 2) & 
        (merged['n_valid_gi'] >= 2)
    ]
    print(f"Regions with both plasma and GI data: {len(both_types):,}")
    # Check concordance when both available
    if len(both_types) > 0:
        concordant = both_types[
            (both_types['max_control_plasma'] < 0.01) & 
            (both_types['max_control_gi'] < 0.01)
        ]
        discordant = both_types[
            ((both_types['max_control_plasma'] < 0.01) & (both_types['max_control_gi'] >= 0.03)) |
            ((both_types['max_control_plasma'] >= 0.03) & (both_types['max_control_gi'] < 0.01))
        ]
        print(f"  Concordant (both <1%): {len(concordant):,}")
        print(f"  Discordant: {len(discordant):,}")
    return merged, tier1, tier2, tier3



def main():
    df = pd.read_parquet(glob.glob("/users/zetzioni/sharedscratch/loyfer_atlas/marker_regions/parts/*OAC*.parquet"))
    df = df.drop_duplicates(["name"])
    
    # Run the analysis
    filtered_regions = test_broad_filter_thresholds(df)
    
    control_mv = pd.read_parquet("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/marker_values.parquet")
    control_cov = pd.read_parquet("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/controls/cfDNA/coverage.parquet")

    # Run the analysis
    merged_data, tier1, tier2, tier3 = analyze_controls_for_filtered_regions(
        filtered_regions, control_mv, control_cov
    )

    # If we need more markers, examine distribution
    if len(tier1) < 400:
        print("\n=== EXPANDING SELECTION ===")
        print(f"Tier 1 has only {len(tier1)} markers, need to use additional tiers")
        
        # Combine tiers as needed
        if len(tier1) + len(tier2) >= 400:
            print(f"Using Tier 1 + best from Tier 2")
        else:
            print(f"Need to use all three tiers")


