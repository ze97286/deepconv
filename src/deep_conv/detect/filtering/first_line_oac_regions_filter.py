import pandas as pd
import glob


def refined_stage1_filtering(df):
    """
    Refined Stage 1: Cancer + blood focus + broad GI filter
    """
    
    print("="*60)
    print("REFINED STAGE 1: CANCER + BLOOD + BROAD GI FILTER")
    print("="*60)
    
    # Blood/immune cell columns
    blood_immune_cols = [
        'Granulocytes', 'T-cells', 'B-cells', 'NK-cells', 
        'Monocytes', 'CD34-erythroblasts', 'CD34-megakaryocytes'
    ]
    
    # Calculate max blood/immune signal
    df['max_blood_immune'] = df[blood_immune_cols].max(axis=1)
    
    # Calculate max GI signal
    gi_cols = ['Colon', 'Esophagus', 'Gastric', 'Small-intestine']
    df['max_gi'] = df[gi_cols].max(axis=1)
    
    # Calculate GI vs OAC ratio
    df['gi_to_oac_ratio'] = df['max_gi'] / df['OAC']
    
    # REFINED filtering
    filtered = df[
        # High cancer signal
        (df['OAC'] >= 0.7) &
        (df['OAC_coverage'] >= 80) &
        
        # STRICT on blood/immune
        (df['max_blood_immune'] <= 0.01) &
        (df['median_blood_background'] <= 0.005) &
        
        # BROAD GI filter - two conditions
        (df['max_gi'] <= 0.2) &  # Max 20% in any GI tissue
        (df['gi_to_oac_ratio'] <= 0.25) &  # GI < 1/4 of OAC signal
        
        # Updated cfDNA compatibility
        (df['region_length'] <= 400) &
        (df['n_cpgs'] >= 4) &
        (df['n_cpgs'] <= 30)
    ]
    
    print(f"Regions after refined filtering: {len(filtered):,}")
    
    # Show statistics
    print(f"\nQuality distribution:")
    print(f"  OAC signal: {filtered['OAC'].min():.3f} - {filtered['OAC'].max():.3f}")
    print(f"  Max blood/immune: {filtered['max_blood_immune'].min():.4f} - {filtered['max_blood_immune'].max():.4f}")
    print(f"  Max GI: {filtered['max_gi'].min():.3f} - {filtered['max_gi'].max():.3f}")
    print(f"  GI/OAC ratio: {filtered['gi_to_oac_ratio'].min():.3f} - {filtered['gi_to_oac_ratio'].max():.3f}")
    
    # Individual GI tissue distribution
    for col in gi_cols:
        print(f"  {col}: {filtered[col].min():.3f} - {filtered[col].max():.3f}")
    
    # Fragment characteristics
    print(f"\nFragment characteristics:")
    print(f"  Region length: {filtered['region_length'].min()} - {filtered['region_length'].max()} bp")
    print(f"  CpGs per region: {filtered['n_cpgs'].min()} - {filtered['n_cpgs'].max()}")
    
    # Chromosome distribution
    print(f"\nChromosome distribution:")
    chr_counts = filtered['chr'].value_counts().head(10)
    for chr_name, count in chr_counts.items():
        print(f"  {chr_name}: {count}")
    
    return filtered

def progressive_refined_filtering(df):
    """
    Try different levels of refined filtering
    """
    
    strategies = {
        'strict_refined': {
            'oac_min': 0.75,
            'blood_max': 0.005,
            'gi_max': 0.15,
            'gi_ratio': 0.2,
            'coverage_min': 100
        },
        'moderate_refined': {
            'oac_min': 0.7,
            'blood_max': 0.01,
            'gi_max': 0.2,
            'gi_ratio': 0.25,
            'coverage_min': 80
        },
        'lenient_refined': {
            'oac_min': 0.65,
            'blood_max': 0.015,
            'gi_max': 0.25,
            'gi_ratio': 0.3,
            'coverage_min': 60
        }
    }
    
    # Pre-calculate columns
    blood_immune_cols = [
        'Granulocytes', 'T-cells', 'B-cells', 'NK-cells', 
        'Monocytes', 'CD34-erythroblasts', 'CD34-megakaryocytes'
    ]
    gi_cols = ['Colon', 'Esophagus', 'Gastric', 'Small-intestine']
    
    max_blood_immune = df[blood_immune_cols].max(axis=1)
    max_gi = df[gi_cols].max(axis=1)
    gi_to_oac_ratio = max_gi / df['OAC']
    
    results = {}
    
    for strategy_name, params in strategies.items():
        
        filtered = df[
            # Cancer requirements
            (df['OAC'] >= params['oac_min']) &
            (df['OAC_coverage'] >= params['coverage_min']) &
            
            # Blood/immune
            (max_blood_immune <= params['blood_max']) &
            (df['median_blood_background'] <= params['blood_max']/2) &
            
            # GI filters
            (max_gi <= params['gi_max']) &
            (gi_to_oac_ratio <= params['gi_ratio']) &
            
            # cfDNA compatibility (your updated criteria)
            (df['region_length'] <= 400) &
            (df['n_cpgs'] >= 4) &
            (df['n_cpgs'] <= 30)
        ]
        
        results[strategy_name] = filtered
        print(f"{strategy_name}: {len(filtered):,} regions")
        
        if len(filtered) > 0:
            print(f"  OAC range: {filtered['OAC'].min():.3f}-{filtered['OAC'].max():.3f}")
            print(f"  Max GI: {max_gi[filtered.index].max():.3f}")
            print(f"  GI/OAC ratio: {gi_to_oac_ratio[filtered.index].max():.3f}")
    
    return results


def process(input_dir, output_file):
    df = pd.read_parquet(glob.glob(f"{input_dir}/*OAC*"))
    blood_cells = ['Granulocytes', 'T-cells', 'B-cells', 'NK-cells', 'Monocytes', 'CD34-erythroblasts', 'CD34-megakaryocytes']
    df['median_blood_background'] = df[blood_cells].median(axis=1)
    df['region_length'] = df['end'] - df['start']
    df['n_cpgs'] = df['endCpG'] - df['startCpG']
    filtered = refined_stage1_filtering(df)
    filtered = progressive_refined_filtering(df)
    filtered.to_csv(output_file, index=False, sep="\t")
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description='admix synthetic samples')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input tissue pat files')
    parser.add_argument('--output_file', type=str, required=True, help='Directory containing input control pat files')
    args = parser.parse_args()
    process(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
