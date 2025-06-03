import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pyarrow.parquet as pq
from tqdm import tqdm

def load_training_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load marker values, coverage, and ground truth data.
    
    Returns:
        marker_values, coverage, ground_truth DataFrames
    """
    data_path = Path(data_dir)
    
    print(f"Loading data from {data_path}")
    marker_values = pd.read_parquet(data_path / "marker_values.parquet")
    coverage = pd.read_parquet(data_path / "coverage.parquet")
    ground_truth = pd.read_parquet(data_path / "ground_truth_y.parquet")
    
    print(f"Loaded shapes:")
    print(f"  Marker values: {marker_values.shape}")
    print(f"  Coverage: {coverage.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    
    return marker_values, coverage, ground_truth

def load_cancer_markers(atlas_path: str, cell_type: str = "OAC") -> np.ndarray:
    """
    Load atlas and extract cancer marker indices for the specified cell type.
    
    Returns:
        Array of marker indices
    """
    atlas_df = pd.read_csv(atlas_path, sep='\t')
    cancer_markers = atlas_df[atlas_df['target'] == cell_type]
    marker_indices = cancer_markers.index.values
    print(f"Found {len(marker_indices)} cancer markers for {cell_type}")
    return marker_indices

def load_tcga_cna_profiles(tcga_file: str) -> pd.DataFrame:
    """
    Load TCGA CNA profiles for OAC samples.
    
    Returns:
        DataFrame with columns: icgc_sample_id, chr, start, end, cn
    """
    print(f"Loading TCGA CNA profiles from {tcga_file}")
    
    tcga_df = pd.read_csv(tcga_file, sep='\t')
    
    # Extract relevant columns
    cna_profiles = tcga_df[['icgc_sample_id', 'chromosome', 'chromosome_start', 
                           'chromosome_end', 'copy_number']].copy()
    
    # Rename columns
    cna_profiles.columns = ['icgc_sample_id', 'chr', 'start', 'end', 'cn']
    
    # Add 'chr' prefix if needed
    cna_profiles['chr'] = cna_profiles['chr'].apply(
        lambda x: f'chr{x}' if not str(x).startswith('chr') else str(x)
    )
    
    # Clean up
    cna_profiles = cna_profiles.dropna()
    cna_profiles['start'] = cna_profiles['start'].astype(int)
    cna_profiles['end'] = cna_profiles['end'].astype(int)
    cna_profiles['cn'] = cna_profiles['cn'].astype(float)
    
    # Get unique samples
    unique_samples = cna_profiles['icgc_sample_id'].unique()
    print(f"Found {len(unique_samples)} unique TCGA/ICGC samples")
    
    return cna_profiles

def load_marker_regions(atlas_path: str) -> pd.DataFrame:
    """
    Load marker regions with genomic coordinates.
    """
    atlas_df = pd.read_csv(atlas_path, sep='\t')
    return atlas_df[['chr', 'start', 'end']]

def precompute_cna_profiles(
    marker_regions: pd.DataFrame,
    tcga_cna: pd.DataFrame,
    unique_samples: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Precompute CN profiles for all ICGC samples to avoid repeated computation.
    
    Returns:
        Dictionary mapping icgc_sample_id to CN array for all markers
    """
    print("Precomputing CNA profiles for all ICGC samples...")
    cna_profiles = {}
    
    for sample_id in tqdm(unique_samples, desc="Precomputing CNA profiles"):
        sample_cna = tcga_cna[tcga_cna['icgc_sample_id'] == sample_id]
        cn_values = assign_cn_to_markers_vectorized(marker_regions, sample_cna)
        cna_profiles[sample_id] = cn_values
    
    return cna_profiles

def assign_cn_to_markers_vectorized(
    marker_regions: pd.DataFrame,
    cna_segments: pd.DataFrame
) -> np.ndarray:
    """
    Vectorized version of CN assignment for better performance.
    """
    cn_values = np.ones(len(marker_regions)) * 2.0
    
    # Calculate marker midpoints once
    midpoints = (marker_regions['start'].values + marker_regions['end'].values) // 2
    chromosomes = marker_regions['chr'].values
    
    # Group segments by chromosome for faster lookup
    for chrom in np.unique(chromosomes):
        # Get indices of markers on this chromosome
        chr_mask = chromosomes == chrom
        chr_indices = np.where(chr_mask)[0]
        chr_midpoints = midpoints[chr_mask]
        
        # Get segments for this chromosome
        chr_segments = cna_segments[cna_segments['chr'] == chrom]
        if len(chr_segments) == 0:
            continue
        
        # Vectorized interval search
        seg_starts = chr_segments['start'].values
        seg_ends = chr_segments['end'].values
        seg_cns = chr_segments['cn'].values
        
        # For each segment, find all markers within it
        for i in range(len(seg_starts)):
            mask = (chr_midpoints >= seg_starts[i]) & (chr_midpoints < seg_ends[i])
            if mask.any():
                cn_values[chr_indices[mask]] = seg_cns[i]
    
    return cn_values

def apply_cna_to_sample(
    marker_values: np.ndarray,
    coverage: np.ndarray,
    cn_values: np.ndarray,
    cancer_marker_indices: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simpler approach: Just scale the observed signal by CNA.
    
    Rationale: In regions with CN=4, the tumor signal is effectively doubled
    in the mixture, even though the tumor fraction stays the same.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    adj_marker_values = marker_values.copy()
    adj_coverage = coverage.copy().astype(float)
    
    for idx in cancer_marker_indices:
        if coverage[idx] == 0 or np.isnan(marker_values[idx]):
            continue
            
        cn = cn_values[idx]
        if cn == 2.0:
            continue
        
        # Scale factor for signal
        signal_factor = cn / 2.0
        
        # The observed unmethylated proportion increases with amplification
        # but is bounded by the tumor fraction
        # Example: 5% tumor fraction with 80% tumor unmethylated
        # CN=2: 0.05 * 0.80 = 0.04 (4% observed)
        # CN=4: might go up to ~0.08 (8% observed) but not 0.16
        
        # Adjust the marker value
        current_signal = marker_values[idx]
        adjusted_signal = current_signal * signal_factor
        
        # Apply a soft ceiling based on reasonable tumor fraction limits
        # This prevents values from exceeding 1.0
        max_reasonable_signal = min(0.5, current_signal * 3)  # Don't let signal more than triple
        adj_marker_values[idx] = min(adjusted_signal, max_reasonable_signal, 1.0)
        
        # Adjust coverage
        adj_coverage[idx] = probabilistic_round(coverage[idx] * signal_factor, rng)
        
        if adj_coverage[idx] == 0:
            adj_marker_values[idx] = np.nan
    
    return adj_marker_values, adj_coverage.astype(int)
def probabilistic_round(value: float, rng: np.random.Generator) -> int:
    """Probabilistic rounding to preserve expected values."""
    if value == 0:
        return 0
    
    integer_part = int(value)
    fractional_part = value - integer_part
    
    if rng.random() < fractional_part:
        return integer_part + 1
    else:
        return integer_part

def calculate_augmentations_needed(
    ground_truth: pd.DataFrame,
    cell_type: str,
    min_tf_threshold: float = 5e-4,
    target_samples_per_tf_range: Dict[Tuple[float, float], int] = None
) -> pd.Series:
    """
    Calculate how many augmentations are needed for each sample based on TF.
    
    Returns:
        Series with number of augmentations per sample index
    """
    if target_samples_per_tf_range is None:
        target_samples_per_tf_range = {
            (0, 0): 0,  
            (0.0001, 0.001): 2, 
            (0.001, 0.01): 3,
            (0.01, 0.1): 4,
            (0.1, 0.2): 4,
            (0.2, 0.8): 4
        }
    
    tf_values = ground_truth[cell_type].values
    augmentations = np.zeros(len(ground_truth), dtype=int)
    
    for i, tf in enumerate(tf_values):
        if tf < min_tf_threshold:
            augmentations[i] = 0  # No augmentation for very low TF
            continue
            
        # Find which range this TF falls into
        for (low, high), n_aug in target_samples_per_tf_range.items():
            if low <= tf < high or (high == 0.4 and tf >= high):
                augmentations[i] = n_aug
                break
    
    return pd.Series(augmentations, index=ground_truth.index)

def augment_dataset(
    data_dir: str,
    atlas_path: str,
    tcga_cna_file: str,
    output_dir: str,
    cell_type: str = "OAC",
    cell_type_index: int = 9,
    min_tf_threshold: float = 5e-4,
    seed: int = 42
):
    """
    Main function to augment the dataset with CNA profiles.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    marker_values, coverage, ground_truth = load_training_data(data_dir)
    
    # Load cancer marker indices
    cancer_marker_indices = load_cancer_markers(atlas_path, cell_type)
    
    # Load marker regions
    marker_regions = load_marker_regions(atlas_path)
    
    # Load TCGA CNA profiles
    tcga_cna = load_tcga_cna_profiles(tcga_cna_file)
    unique_samples = tcga_cna['icgc_sample_id'].unique()
    # Precompute CNA profiles for all ICGC samples
    cna_profiles_dict = precompute_cna_profiles(marker_regions, tcga_cna, unique_samples)
    
    # Calculate augmentations needed per sample
    augmentations_per_sample = calculate_augmentations_needed(
        ground_truth, cell_type, min_tf_threshold
    )
    
    total_augmentations = augmentations_per_sample.sum()
    print(f"\nPlanning to create {total_augmentations} augmented samples")
    print(f"Original samples to augment: {(augmentations_per_sample > 0).sum()}")
    
    # Initialize lists to store augmented data
    aug_marker_values_list = []
    aug_coverage_list = []
    aug_ground_truth_list = []
    aug_sample_info = []
    
    # Keep track of original sample indices
    original_indices = []
    
    # First, add all original samples
    # Get sample columns (excluding metadata columns like 'name' and 'direction')
    metadata_cols = ['name', 'direction']
    sample_columns = [col for col in marker_values.columns if col not in metadata_cols]
    
    print(f"Processing {len(sample_columns)} samples...")
    
    for i, col in enumerate(tqdm(sample_columns, desc="Processing samples")):
        # Add original sample
        aug_marker_values_list.append(marker_values[col].values)
        aug_coverage_list.append(coverage[col].values)
        aug_ground_truth_list.append(ground_truth.iloc[i])
        aug_sample_info.append({
            'original_index': i,
            'icgc_sample_id': 'original',
            'augmentation_num': 0
        })
        
        # Check if we need to augment this sample
        n_augmentations = augmentations_per_sample.iloc[i]
        
        if n_augmentations > 0:
            # Get TF for this sample
            tf = ground_truth.iloc[i][cell_type]
            
            # Randomly select ICGC samples for augmentation
            rng = np.random.default_rng(seed + i)
            selected_icgc = rng.choice(unique_samples, size=n_augmentations, replace=True)
            
            for aug_num, icgc_id in enumerate(selected_icgc, 1):
                # Get precomputed CNA profile
                cn_values = cna_profiles_dict[icgc_id]
                
                # Apply CNA to sample
                aug_values, aug_cov = apply_cna_to_sample(
                    marker_values[col].values,
                    coverage[col].values,
                    cn_values,
                    cancer_marker_indices,
                    seed=seed + i + aug_num
                )
                
                # Store augmented data
                aug_marker_values_list.append(aug_values)
                aug_coverage_list.append(aug_cov)
                aug_ground_truth_list.append(ground_truth.iloc[i])  # Same ground truth
                aug_sample_info.append({
                    'original_index': i,
                    'icgc_sample_id': icgc_id,
                    'augmentation_num': aug_num
                })
    
    print(f"\nCreated {len(aug_marker_values_list)} total samples "
          f"({len(sample_columns)} original + {len(aug_marker_values_list) - len(sample_columns)} augmented)")
    
    # Create output DataFrames
    # Marker values
    aug_marker_df = pd.DataFrame(
        np.column_stack(aug_marker_values_list),
        columns=[f'sample_{i}' for i in range(len(aug_marker_values_list))]
    )
    # Add metadata columns if they exist
    if 'name' in marker_values.columns:
        aug_marker_df.insert(0, 'name', marker_values['name'])
    if 'direction' in marker_values.columns:
        aug_marker_df.insert(1, 'direction', marker_values['direction'])
    
    # Coverage
    aug_coverage_df = pd.DataFrame(
        np.column_stack(aug_coverage_list),
        columns=[f'sample_{i}' for i in range(len(aug_coverage_list))]
    )
    if 'name' in coverage.columns:
        aug_coverage_df.insert(0, 'name', coverage['name'])
    if 'direction' in coverage.columns:
        aug_coverage_df.insert(1, 'direction', coverage['direction'])
    
    # Ground truth
    aug_ground_truth_df = pd.DataFrame(aug_ground_truth_list)
    
    # Sample info
    sample_info_df = pd.DataFrame(aug_sample_info)
    
    # Save outputs
    print("\nSaving augmented data...")
    aug_marker_df.to_parquet(output_path / "aug_marker_values.parquet",  index=False)
    aug_coverage_df.to_parquet(output_path / "aug_coverage.parquet",  index=False)
    aug_ground_truth_df.to_parquet(output_path / "aug_ground_truth_y.parquet",  index=False)
    sample_info_df.to_parquet(output_path / "aug_sample_info.parquet",  index=False)
    
    print(f"Augmented data saved to {output_path}")
    
    # Print summary statistics
    print("\nAugmentation summary:")
    print(f"Total samples: {len(aug_marker_values_list)}")
    print(f"Original samples: {len(sample_columns)}")
    print(f"Augmented samples: {len(aug_marker_values_list) - len(sample_columns)}")
    print(f"Unique ICGC profiles used: {sample_info_df['icgc_sample_id'].nunique() - 1}")  # -1 for 'original'
    
    # TF distribution in augmented data
    tf_dist = aug_ground_truth_df[cell_type].value_counts(bins=[0, 5e-4, 1e-3, 1e-2, 1e-1, 2e-1, 1.0]).sort_index()
    print("\nTF distribution in augmented data:")
    print(tf_dist)
    
    return aug_marker_df, aug_coverage_df, aug_ground_truth_df, sample_info_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augment dataset with CNA data adjusted samples")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to marker values, coverage, and ground truth data")
    parser.add_argument("--atlas_path", type=str, default=None, help="Path to atlas file")
    args = parser.parse_args()

    augment_dataset(
        data_dir=args.data_dir,
        atlas_path=args.atlas_path,
        tcga_cna_file="/mnt/lustre/shared/ICGC/ESAD-UK/copy_number_somatic_mutation.ESAD-UK.tsv.gz",
        output_dir=args.data_dir,
        cell_type="OAC",
        cell_type_index=9,
        min_tf_threshold=5e-4,
        seed=42
    )

    # augment_dataset(
    #     data_dir="/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/aug_test/",
    #     atlas_path="/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed",
    #     tcga_cna_file="/mnt/lustre/shared/ICGC/ESAD-UK/copy_number_somatic_mutation.ESAD-UK.tsv.gz",
    #     output_dir="/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/aug_test/",
    #     cell_type="OAC",
    #     cell_type_index=9,
    #     min_tf_threshold=5e-4,
    #     seed=42
    # )
