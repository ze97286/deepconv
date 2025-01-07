import pandas as pd
import argparse
from pathlib import Path

# the purpose of this script is to merge by cell type input pat files by aggregating the coverage per pattern. This is further used downstream. 

cell_type_to_pat = {
    "B-cell": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-001-5_B-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-003-6_B-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-049-3_B-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-140_B-cells_md.pat.gz",
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
    "CD4-T-cells": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-001-5_CD4-T-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-003-6_CD4-T-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-049-3_CD4-T-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-140_CD4-T-cells_md.pat.gz",
    ],
    "CD8-T-cells": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-001-5_CD8-T-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-003-6_CD8-T-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-049-3_CD8-T-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-140_CD8-T-cells_md.pat.gz",
    ],
    "Colon": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD563419_Colon_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD563663_Colon_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD564159_Colon_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD565189_Colon_md.pat.gz",
    ],
    "Eosinophils": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-001-5_Eosinophils_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-003-6_Eosinophils_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-049-3_Eosinophils_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-140_Eosinophils_md.pat.gz",
    ],
    "Esophagus": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD563678_Esophagus_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD564986_Esophagus_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD565136_Esophagus_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD565252_Esophagus_md.pat.gz",
    ],
    "Monocytes": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-001-5_Monocytes_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-003-6_Monocytes_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-049-3_Monocytes_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-140_Monocytes_md.pat.gz",
    ],
    "Neutrophils": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-001-5_Neutrophils_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-003-6_Neutrophils_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-049-3_Neutrophils_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-140_Neutrophils_md.pat.gz",
    ],
    "NK-cells": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-001-5_NK-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-003-6_NK-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-049-3_NK-cells_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/UKVAC-140_NK-cells_md.pat.gz",
    ],
    "OAC": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-011_ScrBsl_tumour_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-021_ScrBsl_tumour_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/129-001_ScrBsl_tumour_md.pat.gz",
    ],
    "Pancreas": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD564011_Pancreas_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD564404_Pancreas_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD564844_Pancreas_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD565341_Pancreas_md.pat.gz",
    ],
    "Stomach": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD563162_Stomach_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD563430_Stomach_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD564596_Stomach_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/CD565042_Stomach_md.pat.gz",
    ],
    "Duodenum": [
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/069-004_ScrBsl_duodenum_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-011_ScrBsl_duodenum_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-013_ScrBsl_duodenum_md.pat.gz",
        "/mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/pat/Tissue/071-015_ScrBsl_duodenum_md.pat.gz",
    ],
}

def merge_pat_files(pat_files, output_path):
    """
    Merge pat files by summing read counts at each position.
    
    Args:
        pat_files: list of pat file paths to merge
        output_path: where to save merged file
    """
    dfs = []
    for pat_file in pat_files:
        df = pd.read_csv(pat_file, sep='\t', compression='gzip', 
                        names=['chr', 'pos', 'pattern', 'count'])
        dfs.append(df)
    combined = pd.concat(dfs)
    merged = combined.groupby(['chr', 'pos', 'pattern'], as_index=False)['count'].sum()
    merged = merged.sort_values(['chr', 'pos'])
    merged.to_csv(output_path, sep='\t', index=False, header=False, compression='gzip')
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_type', type=str, required=True)
    args = parser.parse_args()
    output_dir = Path("/users/zetzioni/sharedscratch/atlas/pat_by_cell_type/")
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_pat_files(cell_type_to_pat[args.cell_type], output_dir / f"{args.cell_type}_merged.pat.gz")

