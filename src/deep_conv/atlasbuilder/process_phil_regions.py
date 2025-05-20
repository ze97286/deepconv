import pandas as pd

def regions_to_cpgs_indices(df, idx):
    """
    function to process genomic regions in df using CpG indices from idx.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with columns: '#chr', 'start', 'end', 'tbeta', 'ibeta', 'nReads'
    idx : pandas DataFrame
        DataFrame with columns: 'chr', 'start', 'cpg_index'
    
    Returns:
    --------
    pandas DataFrame
        Original df with additional columns: 'startCpg', 'endCpg', 'num_cpgs', 'len'
    """
    import numpy as np
    import pandas as pd
    
    # Create a copy of the input dataframe
    result_df = df.copy()
    
    # Initialize new columns
    result_df['startCpg'] = np.nan
    result_df['endCpg'] = np.nan
    result_df['start_position'] = np.nan
    result_df['end_position'] = np.nan
    result_df['num_cpgs'] = np.nan
    result_df['len'] = np.nan
    
    # Process each chromosome separately
    unique_chroms = np.unique(df['#chr'])
    
    for chrom in unique_chroms:
        # Get data for this chromosome
        chr_mask_df = df['#chr'] == chrom
        chr_mask_idx = idx['chr'] == chrom
        
        if not np.any(chr_mask_df) or not np.any(chr_mask_idx):
            continue
            
        # Extract relevant data
        df_start = df.loc[chr_mask_df, 'start'].values
        df_end = df.loc[chr_mask_df, 'end'].values
        df_indices = np.where(chr_mask_df)[0]
        
        # Sort idx data for binary search
        idx_chrom = idx[chr_mask_idx].sort_values('start')
        idx_starts = idx_chrom['start'].values
        idx_cpg_indices = idx_chrom['cpg_index'].values
        
        # For each region, find first CpG with idx.start >= df.start
        start_positions = np.searchsorted(idx_starts, df_start, side='left')
        # Ensure we don't go out of bounds
        valid_start = start_positions < len(idx_starts)
        
        # For each region, find last CpG with idx.start <= df.end
        end_positions = np.searchsorted(idx_starts, df_end, side='right') - 1
        # Ensure we don't go out of bounds
        valid_end = end_positions >= 0
        
        # Only consider valid entries
        valid_entries = valid_start & valid_end & (start_positions <= end_positions)
        valid_df_indices = df_indices[valid_entries]
        valid_start_pos = start_positions[valid_entries]
        valid_end_pos = end_positions[valid_entries]
        
        # Get the actual values
        if len(valid_df_indices) > 0:
            result_df.loc[valid_df_indices, 'startCpg'] = idx_cpg_indices[valid_start_pos]
            result_df.loc[valid_df_indices, 'endCpg'] = idx_cpg_indices[valid_end_pos]
            result_df.loc[valid_df_indices, 'start_position'] = idx_starts[valid_start_pos]
            result_df.loc[valid_df_indices, 'end_position'] = idx_starts[valid_end_pos]
            
            # Calculate num_cpgs and len
            result_df.loc[valid_df_indices, 'num_cpgs'] = (
                result_df.loc[valid_df_indices, 'endCpg'] - 
                result_df.loc[valid_df_indices, 'startCpg'] + 1
            )
            result_df.loc[valid_df_indices, 'len'] = (
                result_df.loc[valid_df_indices, 'end_position'] - 
                result_df.loc[valid_df_indices, 'start_position'] + 1
            )
    
    return result_df


df = pd.read_csv("/mnt/lustre/users/pxie/nxf/nxf_workflows/pxie/deliver_tmd/tmd_mapBed/final_wgbs_broad_tumour-normal.tsv.gz", sep="\t")
idx = pd.read_csv("/users/zetzioni/sharedscratch/wgbs_tools/references/hg38/CpG.bed.gz", sep="\t", names=['chr','start','cpg_index'])
out = regions_to_cpgs_indices(df, idx)
out.to_csv("/users/zetzioni/sharedscratch/phil_regions.csv", sep="\t", index=False)
