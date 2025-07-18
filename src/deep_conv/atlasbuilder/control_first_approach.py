import pandas as pd
import numpy as np
from scipy import stats
import argparse
import glob

def find_control_clean_regions(tumor_dir, control_dir, min_cpgs, max_control_signal=0.01):
    """
    Step 1: Find regions with very low methylation in controls
    Step 2: Among those, find regions that correlate with tumor purity
    """
    print("="*60)
    print("CONTROL-FIRST APPROACH")
    print("="*60)
    
    # Load all data
    print("Loading all chromosomes...")
    all_tumor_mv = []
    all_control_mv = []
    
    for chr_num in range(1, 23):
        try:
            # Use the filtered files that were created by the original script
            tumor_mv = pd.read_parquet(f"{tumor_dir}/l{min_cpgs}_chr{chr_num}_filtered_marker_values.parquet")
            control_mv = pd.read_parquet(f"{control_dir}/l{min_cpgs}_chr{chr_num}_filtered_marker_values.parquet")
            
            # Add chromosome info
            tumor_mv['chr'] = f"chr{chr_num}"
            control_mv['chr'] = f"chr{chr_num}"
            
            all_tumor_mv.append(tumor_mv)
            all_control_mv.append(control_mv)
            
        except FileNotFoundError:
            print(f"Skipping chr{chr_num}")
            continue
    
    # Combine all chromosomes
    print("Combining chromosomes...")
    combined_tumor = pd.concat(all_tumor_mv, ignore_index=True)
    combined_control = pd.concat(all_control_mv, ignore_index=True)
    
    print(f"Total tumor regions: {len(combined_tumor)}")
    print(f"Total control regions: {len(combined_control)}")
    
    # Get sample columns
    tumor_sample_cols = [col for col in combined_tumor.columns if col not in ['name', 'direction', 'chr']]
    control_sample_cols = [col for col in combined_control.columns if col not in ['name', 'direction', 'chr']]
    
    print(f"Tumor samples: {len(tumor_sample_cols)}")
    print(f"Control samples: {len(control_sample_cols)}")
    
    # STEP 1: Find regions with low control methylation
    print(f"\nSTEP 1: Finding regions with control signal ≤ {max_control_signal}")
    
    control_max = combined_control[control_sample_cols].max(axis=1)
    control_clean_mask = control_max <= max_control_signal
    
    print(f"Regions with low control signal: {control_clean_mask.sum():,} ({100*control_clean_mask.sum()/len(control_clean_mask):.1f}%)")
    
    if control_clean_mask.sum() == 0:
        print("No regions with sufficiently low control signal!")
        return None
    
    # Filter to control-clean regions
    clean_tumor = combined_tumor[control_clean_mask].copy()
    clean_control = combined_control[control_clean_mask].copy()
    
    print(f"Proceeding with {len(clean_tumor)} control-clean regions")
    
    # STEP 2: Among control-clean regions, find tumor purity correlation
    print(f"\nSTEP 2: Finding tumor purity correlation among control-clean regions")
    
    # Tumor purity dictionary
    tumor_purity_dict = {
        '069-009_ScrBsl_tumour_cna_corrected': 0.5171,
        '071-011_ScrBsl_tumour_cna_corrected': 0.2361,
        '071-014_ScrBsl_tumour_cna_corrected': 0.07926,
        '071-021_ScrBsl_tumour_cna_corrected': 0.4766,
        '071-022_ScrBsl_tumour_cna_corrected': 0.4108,
        '071-030_ScrBsl_tumour_cna_corrected': 0.0801,
        '071-043_ScrBsl_tumour_cna_corrected': 0.4607,
        '129-001_ScrBsl_tumour_cna_corrected': 0.6921
    }
    
    # Get tumor samples with purity info
    tumor_purity_cols = [col for col in tumor_sample_cols if col in tumor_purity_dict]
    purities = np.array([tumor_purity_dict[col] for col in tumor_purity_cols])
    
    print(f"Tumor purity samples: {len(tumor_purity_cols)}")
    print(f"Purities: {purities}")
    
    # Calculate correlations
    correlations = []
    pvalues = []
    
    print("Calculating tumor purity correlations...")
    for idx in range(len(clean_tumor)):
        if idx % 50000 == 0:
            print(f"  Processed {idx}/{len(clean_tumor)}")
        
        signals = clean_tumor.iloc[idx][tumor_purity_cols].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(signals)
        if valid_mask.sum() < 3:
            correlations.append(np.nan)
            pvalues.append(np.nan)
            continue
        
        valid_signals = signals[valid_mask]
        valid_purities = purities[valid_mask]
        
        # Calculate correlation
        try:
            r, p = stats.pearsonr(valid_purities, valid_signals)
            correlations.append(r)
            pvalues.append(p)
        except:
            correlations.append(np.nan)
            pvalues.append(np.nan)
    
    # Find significant correlations
    min_correlation = 0.7
    significant_mask = (
        (np.array(pvalues) < 0.05) & 
        (np.array(correlations) > min_correlation)
    )
    
    print(f"\nResults:")
    print(f"Control-clean regions: {len(clean_tumor)}")
    print(f"Regions with tumor purity correlation > {min_correlation}: {significant_mask.sum()}")
    
    if significant_mask.sum() == 0:
        print("No regions found with both low control signal and tumor purity correlation!")
        return None
    
    # Create results
    results = clean_tumor[significant_mask].copy()
    results['correlation'] = np.array(correlations)[significant_mask]
    results['pvalue'] = np.array(pvalues)[significant_mask]
    results['control_max'] = control_max[control_clean_mask][significant_mask]
    
    # Add tumor signal statistics
    tumor_signals = results[tumor_purity_cols].mean(axis=1)
    results['tumor_mean'] = tumor_signals
    
    # Sort by correlation
    results = results.sort_values('correlation', ascending=False)
    
    print(f"\nTop 10 regions:")
    for idx, row in results.head(10).iterrows():
        print(f"  {row['name']}: corr={row['correlation']:.3f}, tumor={row['tumor_mean']:.3f}, control={row['control_max']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Control-first approach to find tumor-specific regions')
    parser.add_argument('--tumor_dir', required=True)
    parser.add_argument('--control_dir', required=True)
    parser.add_argument('--min_cpgs', type=int, required=True)
    parser.add_argument('--output_atlas', required=True)
    parser.add_argument('--max_control_signal', type=float, default=0.01)
    
    args = parser.parse_args()
    
    results = find_control_clean_regions(
        args.tumor_dir, args.control_dir, args.min_cpgs, args.max_control_signal
    )
    
    if results is not None and len(results) > 0:
        # Create atlas format
        atlas_df = pd.DataFrame({
            'chr': results['chr'],
            'start': 0,  # Would need to extract from name
            'end': 0,    # Would need to extract from name
            'name': results['name'],
            'direction': results['direction'],
            'startCpG': 0,
            'endCpG': 0,
            'target': 'OAC'
        })
        
        atlas_df.to_csv(args.output_atlas, sep='\t', index=False)
        print(f"\nSaved {len(atlas_df)} tumor-specific regions to {args.output_atlas}")
    else:
        print("No suitable regions found!")

if __name__ == "__main__":
    main()