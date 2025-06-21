import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import gzip
from collections import defaultdict
import random
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
from tqdm import tqdm
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticMixtureGenerator:
    """
    Generate synthetic cfDNA mixtures for deep learning model training with massive parallelization.
    """
    
    def __init__(self, 
                 tumor_pat_files: List[str],
                 control_pat_files: List[str],
                 cna_profiles_file: str,
                 output_dir: str,
                 n_workers: int = None):
        """
        Initialize the mixture generator.
        
        Args:
            tumor_pat_files: List of paths to tumor PAT files (diploid normalized)
            control_pat_files: List of paths to control cfDNA PAT files
            cna_profiles_file: Path to TSV file with all CNA profiles
            output_dir: Directory to save synthetic mixtures
            n_workers: Number of parallel workers (default: CPU count)
        """
        self.tumor_pat_files = tumor_pat_files
        self.control_pat_files = control_pat_files
        self.output_dir = output_dir
        self.n_workers = n_workers or cpu_count()
        
        # Load CNA profiles
        logger.info(f"Loading CNA profiles from {cna_profiles_file}")
        self.cna_df = pd.read_csv(cna_profiles_file, sep='\t')
        self.cna_profiles = self.cna_df['icgc_sample_id'].unique()
        logger.info(f"Loaded {len(self.cna_profiles)} CNA profiles")
        
        # Pre-load tumor and control data into memory for efficiency
        logger.info("Pre-loading PAT files into memory...")
        self.tumor_data = [self.load_pat_file(f) for f in tumor_pat_files]
        self.control_data = [self.load_pat_file(f) for f in control_pat_files]
        logger.info("PAT files loaded")
        
        # Define sampling distribution
        self.tf_ranges = [
            (0.0, 0.0, 30000),      # True negatives
            (0.0005, 0.001, 30000), # 0.05-0.1%
            (0.001, 0.005, 60000),  # 0.1-0.5%
            (0.005, 0.01, 60000),   # 0.5-1%
            (0.01, 0.02, 40000),    # 1-2%
            (0.02, 0.05, 40000),    # 2-5%
            (0.05, 0.10, 20000),    # 5-10%
            (0.10, 0.20, 15000),    # 10-20%
            (0.20, 0.40, 5000),     # 20-40%
        ]
        
        # Coverage distributions
        self.coverage_options = [
            (3, 0.2),   # 3x - 20%
            (5, 0.3),   # 5x - 30%
            (10, 0.2),  # 10x - 20%
            (15, 0.15), # 15x - 15%
            (20, 0.1),  # 20x - 10%
            (30, 0.05), # 30x - 5%
        ]
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_pat_file(self, filepath: str) -> Dict[Tuple[str, int, str], int]:
        """
        Load PAT file into memory-efficient structure.
        Handles both gzipped and regular files.
        """
        pat_data = {}
        
        # Determine if file is gzipped
        open_func = gzip.open if filepath.endswith('.gz') else open
        mode = 'rt' if filepath.endswith('.gz') else 'r'
        
        with open_func(filepath, mode) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    chrom = parts[0]
                    pos = int(parts[1])
                    pattern = parts[2]
                    count = int(parts[3])
                    pat_data[(chrom, pos, pattern)] = count
        
        return pat_data
    
    def get_cna_for_position(self, cna_profile_id: str, chrom: str, pos: int) -> float:
        """
        Get copy number for a specific position from CNA profile.
        Uses efficient filtering on pre-loaded dataframe.
        """
        # Filter CNA data for this profile and position
        mask = (
            (self.cna_df['icgc_sample_id'] == cna_profile_id) &
            (self.cna_df['chromosome'] == chrom) &
            (self.cna_df['chromosome_start'] <= pos) &
            (self.cna_df['chromosome_end'] >= pos)
        )
        
        matches = self.cna_df[mask]
        
        if len(matches) > 0:
            return matches.iloc[0]['copy_number']
        else:
            return 2.0  # Default diploid
    
    def generate_sample_params(self, n_samples: int, seed: int) -> List[Dict]:
        """
        Generate all sample parameters upfront for parallel processing.
        """
        np.random.seed(seed)
        random.seed(seed)
        
        params = []
        sample_counter = 0
        
        n_tumors = len(self.tumor_pat_files)
        n_controls = len(self.control_pat_files)
        n_cnas = len(self.cna_profiles)
        
        # Generate samples according to TF distribution
        for tf_min, tf_max, n_tf_samples in self.tf_ranges:
            for i in range(n_tf_samples):
                # Sample parameters
                tumor_idx = np.random.randint(n_tumors)
                control_idx = np.random.randint(n_controls)
                cna_idx = np.random.randint(n_cnas)
                
                # Sample TF
                if tf_min == tf_max:  # True negatives
                    tf = 0.0
                else:
                    tf = np.random.uniform(tf_min, tf_max)
                
                # Sample coverage
                coverages, probs = zip(*self.coverage_options)
                coverage = np.random.choice(coverages, p=probs)
                
                params.append({
                    'sample_id': f"sample_{sample_counter:06d}",
                    'tumor_idx': tumor_idx,
                    'control_idx': control_idx,
                    'cna_profile_id': self.cna_profiles[cna_idx],
                    'tumor_fraction': tf,
                    'target_coverage': coverage,
                    'seed': seed + sample_counter  # Unique seed per sample
                })
                
                sample_counter += 1
                
                if sample_counter >= n_samples:
                    break
            
            if sample_counter >= n_samples:
                break
        
        return params[:n_samples]
    
    @staticmethod
    def process_single_sample(params: Dict, generator: 'SyntheticMixtureGenerator') -> Dict:
        """
        Static method to process a single sample. Used for parallel processing.
        """
        # Set random seed for this sample
        np.random.seed(params['seed'])
        
        # Get data references
        tumor_data = generator.tumor_data[params['tumor_idx']]
        control_data = generator.control_data[params['control_idx']]
        cna_profile_id = params['cna_profile_id']
        tf = params['tumor_fraction']
        target_coverage = params['target_coverage']
        
        # Merge patterns with CNA adjustment
        merged = defaultdict(int)
        tumor_read_count = 0  # Track tumor-derived reads
        
        # Process tumor patterns
        for (chrom, pos, pattern), count in tumor_data.items():
            if count > 0:
                # Get copy number for this position
                cn = generator.get_cna_for_position(cna_profile_id, chrom, pos)
                effective_tf = min(tf * (cn / 2.0), 1.0)
                
                # Subsample reads
                if effective_tf > 0:
                    sampled_count = np.random.binomial(count, effective_tf)
                    if sampled_count > 0:  # Only include if we got reads
                        merged[(chrom, pos, pattern)] += sampled_count
                        tumor_read_count += sampled_count
        
        # Process control patterns
        control_read_count = 0  # Track control-derived reads
        if tf < 1.0:  # Only if there's control contribution
            control_fraction = 1.0 - tf
            for (chrom, pos, pattern), count in control_data.items():
                if count > 0:
                    sampled_count = np.random.binomial(count, control_fraction)
                    if sampled_count > 0:  # Only include if we got reads
                        merged[(chrom, pos, pattern)] += sampled_count
                        control_read_count += sampled_count
        
        # Convert to dict (from defaultdict)
        merged = dict(merged)
        
        # Calculate current coverage
        position_reads = defaultdict(int)
        for (chrom, pos, pattern), count in merged.items():
            position_reads[(chrom, pos)] += count
        
        if len(position_reads) == 0:
            current_coverage = 0
        else:
            current_coverage = np.mean(list(position_reads.values()))
        
        # Track reads before downsampling
        total_reads_before_downsample = tumor_read_count + control_read_count
        
        # Downsample to target coverage if needed
        if current_coverage > target_coverage and current_coverage > 0:
            downsample_factor = target_coverage / current_coverage
            downsampled = {}
            
            # Apply downsampling and track the actual reads
            actual_tumor_reads = 0
            actual_total_reads = 0
            
            for key, count in merged.items():
                new_count = np.random.binomial(count, downsample_factor)
                if new_count > 0:  # Only keep patterns with reads
                    downsampled[key] = new_count
                    actual_total_reads += new_count
            
            # Estimate tumor reads after downsampling
            # (proportional to the original tumor fraction in the merged data)
            if total_reads_before_downsample > 0:
                tumor_fraction_in_merged = tumor_read_count / total_reads_before_downsample
                actual_tumor_reads = int(actual_total_reads * tumor_fraction_in_merged)
            
            merged = downsampled
            final_coverage = target_coverage
            final_tumor_reads = actual_tumor_reads
            final_total_reads = actual_total_reads
        else:
            final_coverage = current_coverage
            final_tumor_reads = tumor_read_count
            final_total_reads = total_reads_before_downsample
        
        # Calculate actual tumor fraction
        actual_tumor_fraction = final_tumor_reads / final_total_reads if final_total_reads > 0 else 0.0
        
        # Save PAT file
        output_file = os.path.join(generator.output_dir, f"{params['sample_id']}.pat.gz")
        with gzip.open(output_file, 'wt') as f:
            # Sort by chromosome and position
            sorted_keys = sorted(merged.keys(), key=lambda x: (x[0], x[1]))
            for (chrom, pos, pattern) in sorted_keys:
                count = merged[(chrom, pos, pattern)]
                f.write(f"{chrom}\t{pos}\t{pattern}\t{count}\n")
        
        # Save true concentrations file
        concentrations_file = os.path.join(generator.output_dir, f"{params['sample_id']}_true_concentrations.csv")
        cell_types = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", "Colon", 
                      "Esophagus", "Gastric", "Granulocytes", "Monocytes", "NK-cells", 
                      "OAC", "Small-intestine", "T-cells"]
        
        concentrations = pd.DataFrame({
            'cell_type': cell_types,
            'concentration': [0.0] * len(cell_types)
        })
        # Set OAC concentration to actual tumor fraction
        concentrations.loc[concentrations['cell_type'] == 'OAC', 'concentration'] = actual_tumor_fraction
        
        # Save as CSV (not compressed for easy reading)
        concentrations.to_csv(concentrations_file, index=False, header=False)
        
        # Return metadata
        return {
            'sample_id': params['sample_id'],
            'tumor_idx': params['tumor_idx'],
            'control_idx': params['control_idx'],
            'cna_profile_id': params['cna_profile_id'],
            'tumor_fraction': params['tumor_fraction'],
            'actual_tumor_fraction': actual_tumor_fraction,
            'target_coverage': params['target_coverage'],
            'final_coverage': final_coverage,
            'n_patterns': len(merged),
            'tumor_reads': final_tumor_reads,
            'total_reads': final_total_reads,
            'output_file': output_file
        }
    
    def generate_dataset(self, n_samples: int = 300000, seed: int = 42) -> pd.DataFrame:
        """
        Generate full dataset with massive parallelization.
        """
        logger.info(f"Generating {n_samples} samples using {self.n_workers} workers")
        
        # Generate all parameters
        logger.info("Generating sample parameters...")
        all_params = self.generate_sample_params(n_samples, seed)
        
        # Create partial function with generator
        process_func = partial(self.process_single_sample, generator=self)
        
        # Process in parallel with progress bar
        logger.info("Processing samples in parallel...")
        metadata = []
        
        with Pool(self.n_workers) as pool:
            # Process in chunks for progress tracking
            chunk_size = 1000
            n_chunks = (len(all_params) + chunk_size - 1) // chunk_size
            
            with tqdm(total=len(all_params), desc="Generating samples") as pbar:
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(all_params))
                    chunk_params = all_params[start_idx:end_idx]
                    
                    # Process chunk
                    chunk_results = pool.map(process_func, chunk_params)
                    metadata.extend(chunk_results)
                    
                    # Update progress
                    pbar.update(len(chunk_results))
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata)
        metadata_file = os.path.join(self.output_dir, 'metadata.csv.gz')
        metadata_df.to_csv(metadata_file, index=False, compression='gzip')
        
        # Print summary statistics
        logger.info(f"\nGeneration complete! Generated {len(metadata_df)} samples")
        logger.info("\nTF distribution:")
        tf_dist = metadata_df.groupby(
            pd.cut(metadata_df['tumor_fraction'], 
                   bins=[0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 1.0])
        ).size()
        for interval, count in tf_dist.items():
            logger.info(f"  {interval}: {count}")
        
        logger.info("\nCoverage distribution:")
        cov_dist = metadata_df['target_coverage'].value_counts().sort_index()
        for cov, count in cov_dist.items():
            logger.info(f"  {cov}x: {count}")
        
        logger.info(f"\nMetadata saved to: {metadata_file}")
        
        return metadata_df

# python -m deep_conv.atlasbuilder.admixer --tissue_input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.l4/train_tissue/ --control_input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.l4/controls/cfDNA/ --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.l4/train/ --cna_profiles_file /mnt/lustre/shared/ICGC/ESAD-UK/copy_number_somatic_mutation.ESAD-UK.tsv.gz --n_samples 100 --n_workers 8
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='admix synthetic samples')
    parser.add_argument('--tissue_input_dir', type=str, required=True, help='Directory containing input tissue pat files')
    parser.add_argument('--control_input_dir', type=str, required=True, help='Directory containing input control pat files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save admixed pat files')
    parser.add_argument('--cna_profiles_file', type=str, required=True, help="Directory to save admixed pat files")
    parser.add_argument('--n_samples', type=int, required=True, help="Number of samples to generate")
    parser.add_argument('--n_workers', type=int, default=None, help="Number of workers to use for parallelization")
    args = parser.parse_args()

    # Initialize generator
    generator = SyntheticMixtureGenerator(
        tumor_pat_files=glob.glob(args.tissue_input_dir + "/*.pat.gz"),
        control_pat_files=glob.glob(args.control_input_dir + "/*.pat.gz"),
        cna_profiles_file=args.cna_profiles_file,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
    )

    # Generate dataset
    metadata = generator.generate_dataset(n_samples=args.n_samples)
