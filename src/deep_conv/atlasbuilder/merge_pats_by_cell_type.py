import pandas as pd
import argparse
from pathlib import Path
import subprocess

cell_type_to_pat = {
    "B-cells": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652316_Blood-B-Z000000TX.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652317_Blood-B-Z000000UB.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652318_Blood-B-Z000000UR.hg38.pat.gz",        
    ],
    "CD4-T-cells": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652279_Blood-T-CD4-Z000000TT.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652280_Blood-T-CD4-Z000000U7.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652281_Blood-T-CD4-Z000000UM.hg38.pat.gz",
    ],
    "CD8-T-cells": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652282_Blood-T-CD8-Z000000TR.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652283_Blood-T-CD8-Z000000U5.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652284_Blood-T-CD8-Z000000UK.hg38.pat.gz",
    ],
    "T-cells": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652282_Blood-T-CD8-Z000000TR.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652283_Blood-T-CD8-Z000000U5.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652284_Blood-T-CD8-Z000000UK.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652279_Blood-T-CD4-Z000000TT.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652280_Blood-T-CD4-Z000000U7.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652281_Blood-T-CD4-Z000000UM.hg38.pat.gz",
    ],
    "Monocytes": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652302_Blood-Monocytes-Z000000TP.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652303_Blood-Monocytes-Z000000U3.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652304_Blood-Monocytes-Z000000UH.hg38.pat.gz",
    ],
    "NK-cells": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652299_Blood-NK-Z000000TM.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652300_Blood-NK-Z000000U1.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652301_Blood-NK-Z000000UF.hg38.pat.gz",
    ],
    "Erythrocyte_progenitors": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652274_Bone_marrow-Erythrocyte_progenitors-Z000000RF.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652275_Bone_marrow-Erythrocyte_progenitors-Z000000RH.hg38.pat.gz"
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652276_Bone_marrow-Erythrocyte_progenitors-Z000000RK.hg38.pat.gz"
    ],
    "Granulocytes": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652313_Blood-Granulocytes-Z000000TZ.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652314_Blood-Granulocytes-Z000000UD.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652315_Blood-Granulocytes-Z000000UT.hg38.pat.gz",
    ],
    "CD34-erythroblasts": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/D30_CD34-erythroblasts_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/D37_CD34-erythroblasts_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/D66_CD34-erythroblasts_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/D70_CD34-erythroblasts_md.pat.gz",
    ],
    "CD34-megakaryocytes": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/D30_CD34-megakaryocytes_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/D66_CD34-megakaryocytes_md.pat.gz",
    ],   
    "OAC": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-011_ScrBsl_tumour_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-021_ScrBsl_tumour_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/129-001_ScrBsl_tumour_md.pat.gz",
    ],
    "Esophagus": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652332_Esophagus-Epithelial-Z000000PZ.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652333_Esophagus-Epithelial-Z00000426.hg38.pat.gz",        
    ],
    "Colon": [
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652370_Colon-Right-Epithelial-Z000000V0.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652371_Colon-Right-Epithelial-Z000000V8.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652372_Colon-Right-Endocrine-Z0000044S.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652373_Colon-Left-Epithelial-Z000000VA.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652374_Colon-Left-Endocrine-Z0000044J.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652375_Colon-Left-Endocrine-Z0000044T.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652376_Colon-Left-Epithelial-Z0000043B.hg38.pat.gz",
        "/mnt/lustre/shared/Loyfer_etal/hg38/GSM5652377_Colon-Left-Epithelial-Z0000043C.hg38.pat.gz",
    ],
   
    
    
    "Pancreas": [
    ],
    "Stomach": [
    ],
    "Duodenum": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/069-004_ScrBsl_duodenum_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-011_ScrBsl_duodenum_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-013_ScrBsl_duodenum_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-015_ScrBsl_duodenum_md.pat.gz",
    ],
}

import subprocess
import os
from pathlib import Path

def merge_pat_files(pat_files, output_path):
    """
    Merge pat files by summing read counts at each position.
    Ensures the output is properly sorted, bgzip compressed, and tabix indexed.
    
    Args:
        pat_files: list of pat file paths to merge
        output_path: where to save merged file (with .gz extension)
    """
    dfs = []
    for pat_file in pat_files:
        df = pd.read_csv(pat_file, sep='\t', compression='gzip', 
                        names=['chr', 'pos', 'pattern', 'count'])
        dfs.append(df)
    combined = pd.concat(dfs)
    merged = combined.groupby(['chr', 'pos', 'pattern'], as_index=False)['count'].sum()
    
    # Custom sorting function for chromosomes
    def chr_sort_key(chrom):
        # Remove 'chr' prefix if present
        chrom = str(chrom)
        if chrom.startswith('chr'):
            chrom = chrom[3:]
        # Handle numbered chromosomes
        if chrom.isdigit():
            return int(chrom)
        # Handle X, Y, MT, etc.
        elif chrom == 'X':
            return 100
        elif chrom == 'Y':
            return 101
        elif chrom == 'M' or chrom == 'MT':
            return 102
        # Handle other cases
        else:
            return 1000 + ord(chrom[0])
    
    # Apply custom sorting
    merged['chr_sort'] = merged['chr'].apply(chr_sort_key)
    merged = merged.sort_values(['chr_sort', 'pos'])
    merged = merged.drop('chr_sort', axis=1)
    
    # Convert output_path to string if it's not already
    output_path_str = str(output_path)
    
    # Create temp output path without .gz extension
    if output_path_str.endswith('.gz'):
        temp_output = output_path_str[:-3]
    else:
        temp_output = output_path_str
        output_path_str = output_path_str + '.gz'
    
    # Save as uncompressed file
    merged.to_csv(temp_output, sep='\t', index=False, header=False)
    
    # Compress with bgzip (creates temp_output.gz)
    subprocess.run(['bgzip', temp_output])
    
    # Index with tabix
    subprocess.run(['tabix', '-s', '1', '-b', '2', '-e', '2', temp_output + '.gz'])
    
    # If the output path is different from temp_output.gz, move it there
    if temp_output + '.gz' != output_path_str:
        os.rename(temp_output + '.gz', output_path_str)
        os.rename(temp_output + '.gz.tbi', output_path_str + '.tbi')
    
    return output_path_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_type', type=str, required=True)
    args = parser.parse_args()
    output_dir = Path("/users/zetzioni/sharedscratch/loyfer_atlas/pat_by_cell_type/")
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_pat_files(cell_type_to_pat[args.cell_type], output_dir / f"{args.cell_type}_merged.pat.gz")

