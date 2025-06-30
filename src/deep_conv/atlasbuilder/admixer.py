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
import pickle
import h5py
from intervaltree import IntervalTree, Interval
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticMixtureGenerator:
    """
    Highly optimized synthetic cfDNA mixture generator.
    """
    
    def __init__(self, 
                 tumor_pat_files: List[str],
                 control_pat_files: List[str],
                 cna_profiles_file: str,
                 output_dir: str,
                 n_workers: int = None,
                 batch_output: bool = True):
        """
        Initialize the mixture generator with optimizations.
        
        Args:
            batch_output: If True, save samples in batches to HDF5 instead of individual files
        """
        self.tumor_pat_files = tumor_pat_files
        self.control_pat_files = control_pat_files
        self.output_dir = output_dir
        self.n_workers = n_workers or cpu_count()
        self.batch_output = batch_output
        
        # Load and optimize CNA profiles
        logger.info(f"Loading and optimizing CNA profiles from {cna_profiles_file}")
        self.cna_df = pd.read_csv(cna_profiles_file, sep='\t')
        self.cna_profiles = self.cna_df['icgc_sample_id'].unique()
        
        # Create interval trees for fast CNA lookups
        self._build_cna_interval_trees()
        
        # Pre-load and optimize PAT data
        logger.info("Pre-loading and optimizing PAT files...")
        self.tumor_data = [self._load_and_optimize_pat(f) for f in tumor_pat_files]
        self.control_data = [self._load_and_optimize_pat(f) for f in control_pat_files]
        logger.info("PAT files loaded and optimized")
        
        # Define sampling distribution as proportions
        self.tf_ranges = [
            (0.0, 0.0, 0.10),      # True negatives - 10%
            (0.0005, 0.001, 0.10), # 0.05-0.1% - 10%
            (0.001, 0.005, 0.20),  # 0.1-0.5% - 20%
            (0.005, 0.01, 0.20),   # 0.5-1% - 20%
            (0.01, 0.02, 0.133),   # 1-2% - 13.3%
            (0.02, 0.05, 0.133),   # 2-5% - 13.3%
            (0.05, 0.10, 0.067),   # 5-10% - 6.7%
            (0.10, 0.20, 0.05),    # 10-20% - 5%
            (0.20, 0.40, 0.017),   # 20-40% - 1.7%
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
    
    def _build_cna_interval_trees(self):
        """
        Build interval trees for each CNA profile for O(log n) lookups.
        """
        self.cna_trees = {}
        
        for profile_id in self.cna_profiles:
            profile_data = self.cna_df[self.cna_df['icgc_sample_id'] == profile_id]
            
            # Create trees per chromosome
            trees = {}
            for chrom in profile_data['chromosome'].unique():
                tree = IntervalTree()
                chrom_data = profile_data[profile_data['chromosome'] == chrom]
                
                for _, row in chrom_data.iterrows():
                    start = row['chromosome_start']
                    end = row['chromosome_end']
                    
                    # Handle zero-width intervals by extending by 1bp
                    if start >= end:
                        end = start + 1
                    
                    # Add interval with copy number as data
                    tree[start:end] = row['copy_number']
                
                trees[chrom] = tree
            
            self.cna_trees[profile_id] = trees
    
    def _load_and_optimize_pat(self, filepath: str) -> Dict:
        """
        Load PAT file and organize by chromosome for faster processing.
        """
        pat_data = defaultdict(list)
        
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
                    # Store as tuple for efficient processing
                    pat_data[chrom].append((pos, pattern, count))
        
        # Convert to regular dict and sort by position
        pat_dict = {}
        for chrom, data in pat_data.items():
            # Sort by position for cache efficiency
            pat_dict[chrom] = sorted(data, key=lambda x: x[0])
        
        return pat_dict
    
    def get_cna_batch(self, profile_id: str, chrom: str, positions: List[int]) -> np.ndarray:
        """
        Get copy numbers for multiple positions at once.
        """
        if profile_id not in self.cna_trees or chrom not in self.cna_trees[profile_id]:
            return np.full(len(positions), 2.0)  # Default diploid
        
        tree = self.cna_trees[profile_id][chrom]
        copy_numbers = []
        
        for pos in positions:
            intervals = tree[pos]
            if intervals:
                # Take first interval (should only be one)
                copy_numbers.append(next(iter(intervals)).data)
            else:
                copy_numbers.append(2.0)
        
        return np.array(copy_numbers)
    
    @staticmethod
    def process_sample_batch(params_batch: List[Dict], generator: 'SyntheticMixtureGenerator') -> List[Dict]:
        """
        Process a batch of samples together for efficiency.
        """
        results = []
        
        for params in params_batch:
            # Set random seed
            np.random.seed(params['seed'])
            
            # Get data references
            tumor_data = generator.tumor_data[params['tumor_idx']]
            control_data = generator.control_data[params['control_idx']]
            cna_profile_id = params['cna_profile_id']
            tf = params['tumor_fraction']
            target_coverage = params['target_coverage']
            
            # Process by chromosome for cache efficiency
            merged = defaultdict(int)
            tumor_read_count = 0
            control_read_count = 0
            
            # Process tumor data ONLY if tf > 0
            if tf > 0:
                for chrom, chrom_data in tumor_data.items():
                    if not chrom_data:
                        continue
                    
                    # Extract positions and counts
                    positions = [x[0] for x in chrom_data]
                    patterns = [x[1] for x in chrom_data]
                    counts = np.array([x[2] for x in chrom_data])
                    
                    # Get all CNAs for this chromosome at once
                    cnas = generator.get_cna_batch(cna_profile_id, chrom, positions)
                    effective_tfs = np.minimum(tf * (cnas / 2.0), 1.0)
                    
                    # Vectorized sampling for all positions
                    mask = (counts > 0) & (effective_tfs > 0)
                    if np.any(mask):
                        # Sample all at once WITH CNA effects
                        sampled = np.random.binomial(counts[mask], effective_tfs[mask])
                        
                        # Add to merged
                        for i, (pos, pattern, sampled_count) in enumerate(
                            zip(np.array(positions)[mask], 
                                np.array(patterns)[mask], 
                                sampled)):
                            if sampled_count > 0:
                                merged[(chrom, pos, pattern)] += sampled_count
                                tumor_read_count += sampled_count
            
            # Process control data ONLY if tf < 1.0
            if tf < 1.0:
                control_fraction = 1.0 - tf
                
                # IMPORTANT: When tf=0, we should NOT use the CNA profile at all
                # The sample should be purely diploid control
                for chrom, chrom_data in control_data.items():
                    if not chrom_data:
                        continue
                    
                    # Vectorized processing
                    positions = [x[0] for x in chrom_data]
                    patterns = [x[1] for x in chrom_data]
                    counts = np.array([x[2] for x in chrom_data])
                    
                    mask = counts > 0
                    if np.any(mask):
                        # Sample control reads - no CNA effects ever!
                        # Control is always diploid regardless of tumor fraction
                        sampled = np.random.binomial(counts[mask], control_fraction)
                        
                        for i, (pos, pattern, sampled_count) in enumerate(
                            zip(np.array(positions)[mask], 
                                np.array(patterns)[mask], 
                                sampled)):
                            if sampled_count > 0:
                                merged[(chrom, pos, pattern)] += sampled_count
                                control_read_count += sampled_count
            
            # Convert to regular dict
            merged = dict(merged)
            
            # Calculate coverage
            position_reads = defaultdict(int)
            for (chrom, pos, pattern), count in merged.items():
                position_reads[(chrom, pos)] += count
            
            if len(position_reads) == 0:
                current_coverage = 0
                final_tumor_reads = 0
                final_total_reads = 0
                tumor_fraction_actual = 0.0
            else:
                current_coverage = np.mean(list(position_reads.values()))
                
                # For actual tumor fraction, we just use the intended TF
                tumor_fraction_actual = tf
                
                # Downsample if needed
                total_reads = tumor_read_count + control_read_count
                
                if current_coverage > target_coverage and current_coverage > 0:
                    downsample_factor = target_coverage / current_coverage
                    
                    # Downsample all reads at once
                    all_counts = np.array(list(merged.values()))
                    downsampled_counts = np.random.binomial(all_counts, downsample_factor)
                    
                    # Rebuild merged with downsampled counts
                    new_merged = {}
                    for i, (key, new_count) in enumerate(zip(merged.keys(), downsampled_counts)):
                        if new_count > 0:
                            new_merged[key] = new_count
                    
                    merged = new_merged
                    
                    # Calculate final reads
                    final_total_reads = sum(downsampled_counts)
                    if total_reads > 0:
                        tumor_fraction_in_merged = tumor_read_count / total_reads
                        final_tumor_reads = int(final_total_reads * tumor_fraction_in_merged)
                    else:
                        final_tumor_reads = 0
                else:
                    final_tumor_reads = tumor_read_count
                    final_total_reads = total_reads
            
            # Calculate post-CNA tumor read fraction
            tumor_read_fraction = final_tumor_reads / final_total_reads if final_total_reads > 0 else 0.0
            
            # Store results
            result = {
                'sample_id': params['sample_id'],
                'tumor_idx': params['tumor_idx'],
                'control_idx': params['control_idx'],
                'cna_profile_id': params['cna_profile_id'] if tf > 0 else 'diploid',
                'tumor_fraction_intended': params['tumor_fraction'],
                'tumor_fraction_actual': tumor_fraction_actual,
                'tumor_read_fraction': tumor_read_fraction,
                'target_coverage': params['target_coverage'],
                'final_coverage': target_coverage if current_coverage > target_coverage else current_coverage,
                'n_patterns': len(merged),
                'tumor_reads': final_tumor_reads,
                'total_reads': final_total_reads,
                'pat_data': merged
            }
            
            results.append(result)
        
        return results

    def save_batch_to_hdf5(self, results: List[Dict], batch_id: int):
        """
        Save a batch of results to HDF5 format for efficient storage.
        """
        h5_file = os.path.join(self.output_dir, f'batch_{batch_id:05d}.h5')
        
        with h5py.File(h5_file, 'w') as f:
            for result in results:
                sample_group = f.create_group(result['sample_id'])
                
                # Save metadata
                for key, value in result.items():
                    if key != 'pat_data':
                        sample_group.attrs[key] = value
                
                # Save PAT data efficiently
                if result['pat_data']:
                    # Convert to arrays for HDF5 storage
                    chroms = []
                    positions = []
                    patterns = []
                    counts = []
                    
                    for (chrom, pos, pattern), count in result['pat_data'].items():
                        chroms.append(chrom)
                        positions.append(pos)
                        patterns.append(pattern)
                        counts.append(count)
                    
                    # Store as datasets
                    sample_group.create_dataset('chromosomes', data=np.array(chroms, dtype='S10'))
                    sample_group.create_dataset('positions', data=np.array(positions))
                    sample_group.create_dataset('patterns', data=np.array(patterns, dtype='S20'))
                    sample_group.create_dataset('counts', data=np.array(counts))
                
                # Save true concentrations
                cell_types = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                             "Colon", "Esophagus", "Gastric", "Granulocytes", 
                             "Monocytes", "NK-cells", "OAC", "Small-intestine", "T-cells"]
                concentrations = np.zeros(len(cell_types))
                oac_idx = cell_types.index("OAC")
                concentrations[oac_idx] = result['tumor_fraction_actual']  # Use pre-CNA actual TF
                
                sample_group.create_dataset('cell_types', data=np.array(cell_types, dtype='S30'))
                sample_group.create_dataset('concentrations', data=concentrations)
    
    def save_individual_files(self, results: List[Dict]):
        """
        Save results as individual PAT and concentration files.
        """
        for result in results:
            # Save PAT file
            pat_file = os.path.join(self.output_dir, f"{result['sample_id']}.pat.gz")
            with gzip.open(pat_file, 'wt') as f:
                # Sort by chromosome and position
                sorted_keys = sorted(result['pat_data'].keys(), key=lambda x: (x[0], x[1]))
                for (chrom, pos, pattern) in sorted_keys:
                    count = result['pat_data'][(chrom, pos, pattern)]
                    f.write(f"{chrom}\t{pos}\t{pattern}\t{count}\n")
            
            # Save concentrations
            conc_file = os.path.join(self.output_dir, f"{result['sample_id']}_true_concentrations.csv")
            cell_types = ["B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", 
                         "Colon", "Esophagus", "Gastric", "Granulocytes", 
                         "Monocytes", "NK-cells", "OAC", "Small-intestine", "T-cells"]
            
            with open(conc_file, 'w') as f:
                for cell_type in cell_types:
                    conc = result['tumor_fraction_actual'] if cell_type == "OAC" else 0.0  # Use pre-CNA actual TF
                    f.write(f"{cell_type},{conc}\n")
    
    def generate_dataset(self, n_samples: int = 300000, seed: int = 42, 
                        batch_size: int = 100) -> pd.DataFrame:
        """
        Generate dataset with batched processing.
        """
        logger.info(f"Generating {n_samples} samples using {self.n_workers} workers")
        logger.info(f"Batch size: {batch_size}, Output mode: {'HDF5' if self.batch_output else 'Individual files'}")
        
        # Generate all parameters
        all_params = self.generate_sample_params(n_samples, seed)
        
        # Process in batches
        metadata = []
        process_func = partial(self.process_sample_batch, generator=self)
        
        with Pool(self.n_workers) as pool:
            # Split work into chunks for workers
            n_batches = (len(all_params) + batch_size - 1) // batch_size
            
            with tqdm(total=len(all_params), desc="Generating samples") as pbar:
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(all_params))
                    batch_params = all_params[start_idx:end_idx]
                    
                    # Split batch among workers
                    worker_batch_size = max(1, len(batch_params) // self.n_workers)
                    worker_batches = [
                        batch_params[i:i+worker_batch_size] 
                        for i in range(0, len(batch_params), worker_batch_size)
                    ]
                    
                    # Process in parallel
                    worker_results = pool.map(process_func, worker_batches)
                    
                    # Flatten results
                    batch_results = []
                    for worker_batch in worker_results:
                        batch_results.extend(worker_batch)
                    
                    # Save results
                    if self.batch_output:
                        self.save_batch_to_hdf5(batch_results, batch_idx)
                    else:
                        self.save_individual_files(batch_results)
                    
                    # Extract metadata
                    for result in batch_results:
                        metadata.append({k: v for k, v in result.items() if k != 'pat_data'})
                    
                    pbar.update(len(batch_results))
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_file = os.path.join(self.output_dir, 'metadata.csv.gz')
        metadata_df.to_csv(metadata_file, index=False, compression='gzip')
        
        logger.info(f"\nGeneration complete! Generated {len(metadata_df)} samples")
        return metadata_df
    
    def generate_sample_params(self, n_samples: int, seed: int) -> List[Dict]:
        """Generate all sample parameters upfront."""
        np.random.seed(seed)
        random.seed(seed)
        
        params = []
        sample_counter = 0
        
        n_tumors = len(self.tumor_pat_files)
        n_controls = len(self.control_pat_files)
        n_cnas = len(self.cna_profiles)
        
        for tf_min, tf_max, tf_proportion in self.tf_ranges:
            n_tf_samples = int(n_samples * tf_proportion)
            
            for i in range(n_tf_samples):
                tumor_idx = np.random.randint(n_tumors)
                control_idx = np.random.randint(n_controls)
                cna_idx = np.random.randint(n_cnas)
                
                if tf_min == tf_max:
                    tf = 0.0
                else:
                    tf = np.random.uniform(tf_min, tf_max)
                
                coverages, probs = zip(*self.coverage_options)
                coverage = np.random.choice(coverages, p=probs)
                
                params.append({
                    'sample_id': f"sample_{sample_counter:06d}",
                    'tumor_idx': tumor_idx,
                    'control_idx': control_idx,
                    'cna_profile_id': self.cna_profiles[cna_idx],
                    'tumor_fraction': tf,
                    'target_coverage': coverage,
                    'seed': seed + sample_counter
                })
                
                sample_counter += 1
                if sample_counter >= n_samples:
                    break
            
            if sample_counter >= n_samples:
                break
        
        return params[:n_samples]
    
# python -m deep_conv.atlasbuilder.admixer --tissue_input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.l4/train_tissue/ --control_input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.l4/controls/cfDNA/ --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.l4/train/ --cna_profiles_file /mnt/lustre/shared/ICGC/ESAD-UK/copy_number_somatic_mutation.ESAD-UK.tsv.gz --n_samples 100 --n_workers 32
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
