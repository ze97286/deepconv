#!/usr/bin/env python3
"""
Patient-specific tumour-guided cfDNA detection framework.
Uses matched tumour-normal pairs to identify patient-specific methylation markers,
then searches for these in cfDNA to estimate tumour fraction.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import gzip
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PatientSpecificDetector:
    """
    Implements patient-specific tumour fraction detection in cfDNA.
    """
    
    def __init__(self, 
                 min_region_coverage: int = 10,
                 min_differential: float = 0.3,
                 max_normal_variation: float = 0.1,
                 region_size: int = 150,  # Match cfDNA fragment size
                 min_cpgs: int = 3,
                 fragment_aware: bool = True,
                 edge_trim: int = 10,  # Account for 5bp clipping each side
                 n_workers: Optional[int] = None):
        """
        Initialize the detector.
        
        Args:
            min_region_coverage: Minimum coverage required in all samples
            min_differential: Minimum methylation difference between tumour and normal
            max_normal_variation: Maximum allowed variation in normal tissue
            region_size: Size of genomic regions to analyze (default 150bp for cfDNA)
            min_cpgs: Minimum CpGs per region
            fragment_aware: Whether to account for cfDNA fragmentation patterns
            edge_trim: Base pairs to exclude from edges (accounts for clipping)
            n_workers: Number of parallel workers
        """
        self.min_region_coverage = min_region_coverage
        self.min_differential = min_differential
        self.max_normal_variation = max_normal_variation
        self.region_size = region_size
        self.min_cpgs = min_cpgs
        self.fragment_aware = fragment_aware
        self.edge_trim = edge_trim
        self.n_workers = n_workers or cpu_count()
        
    def load_pat_file(self, pat_file: str) -> Dict[str, List[Tuple]]:
        """
        Load a PAT file and organize by genomic regions.
        Format: chromosome    cpg_index    pattern    read_count
        
        Returns:
            Dictionary mapping region_id to list of (cpg_pos, methylation_pattern, count)
        """
        logger.info(f"Loading PAT file: {pat_file}")
        regions = defaultdict(list)
        
        with gzip.open(pat_file, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 4:
                    chrom = parts[0]
                    pos = int(parts[1])
                    pattern = parts[2]
                    count = int(parts[3])
                    
                    # Assign to region
                    region_start = (pos // self.region_size) * self.region_size
                    region_id = f"{chrom}:{region_start}-{region_start + self.region_size}"
                    
                    regions[region_id].append((pos, pattern, count))
        
        return dict(regions)
    
    def calculate_region_methylation(self, region_data: List[Tuple], 
                                   region_start: int = None,
                                   is_cfDNA: bool = False) -> Tuple[float, int, float]:
        """
        Calculate methylation statistics for a region.
        
        Args:
            region_data: List of (position, pattern, count) tuples
            region_start: Start position of the region (for edge trimming)
            is_cfDNA: Whether this is cfDNA data (applies edge trimming)
        
        Returns:
            (unmethylation_proportion, total_coverage, std_dev)
        """
        if not region_data:
            return 0.0, 0, 0.0
        
        total_unmethylated = 0
        total_reads = 0
        
        for pos, pattern, count in region_data:
            # Apply edge trimming for cfDNA if fragment-aware mode
            if self.fragment_aware and is_cfDNA and region_start is not None:
                # Calculate position within region
                pos_in_region = pos - region_start
                
                # Skip CpGs too close to edges (accounting for clipping)
                if pos_in_region < self.edge_trim or pos_in_region > (self.region_size - self.edge_trim):
                    continue
            
            # Count unmethylated CpGs (represented as C in pattern)
            unmethylated_cpgs = pattern.count('C')
            total_cpgs = len(pattern)
            
            if total_cpgs > 0:
                total_unmethylated += (unmethylated_cpgs / total_cpgs) * count
                total_reads += count
        
        if total_reads == 0:
            return 0.0, 0, 0.0
        
        unmeth_prop = total_unmethylated / total_reads
        
        # Calculate standard deviation if multiple reads
        if total_reads > 1:
            # Approximate std dev based on binomial variance
            std_dev = np.sqrt(unmeth_prop * (1 - unmeth_prop) / total_reads)
        else:
            std_dev = 0.5  # Maximum uncertainty
        
        return unmeth_prop, total_reads, std_dev
    
    def identify_informative_regions(self,
                                   tumour_regions: Dict[str, List[Tuple]],
                                   normal_regions: Dict[str, List[Tuple]],
                                   background_normal_regions: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Identify patient-specific informative regions.
        
        Args:
            tumour_regions: tumour PAT data by region
            normal_regions: Matched normal PAT data by region
            background_normal_regions: Optional list of other normal samples for background estimation
            
        Returns:
            DataFrame with informative regions and their statistics
        """
        logger.info("Identifying patient-specific informative regions...")
        
        # Find regions present in both tumour and normal with sufficient coverage
        common_regions = set(tumour_regions.keys()) & set(normal_regions.keys())
        
        informative_regions = []
        
        for region_id in tqdm(common_regions, desc="Analyzing regions"):
            # Extract region start position
            try:
                chrom, pos_range = region_id.split(':')
                region_start = int(pos_range.split('-')[0])
            except:
                region_start = None
            
            # Calculate methylation for tumour and normal
            # Don't apply edge trimming to tissue samples
            tumour_meth, tumour_cov, tumour_std = self.calculate_region_methylation(
                tumour_regions[region_id], region_start, is_cfDNA=False
            )
            normal_meth, normal_cov, normal_std = self.calculate_region_methylation(
                normal_regions[region_id], region_start, is_cfDNA=False
            )
            
            # Check coverage requirements
            if tumour_cov < self.min_region_coverage or normal_cov < self.min_region_coverage:
                continue
            
            # Check minimum CpGs
            if len(tumour_regions[region_id]) < self.min_cpgs:
                continue
            
            # Calculate differential
            differential = abs(tumour_meth - normal_meth)
            
            if differential < self.min_differential:
                continue
            
            # Calculate background variation if other normals provided
            background_var = 0.0
            background_mean = normal_meth  # Default to patient normal
            
            if background_normal_regions:
                background_values = []
                for bg_normal in background_normal_regions:
                    if region_id in bg_normal:
                        bg_meth, bg_cov, _ = self.calculate_region_methylation(
                            bg_normal[region_id], region_start, is_cfDNA=False
                        )
                        if bg_cov >= self.min_region_coverage:
                            background_values.append(bg_meth)
                
                if len(background_values) >= 2:
                    background_mean = np.mean(background_values)
                    background_var = np.std(background_values)
                    
                    # Skip if too much variation in normal samples
                    if background_var > self.max_normal_variation:
                        continue
            
            # Calculate informativeness score
            # High score = high differential, high coverage, low background variation
            score = (differential / (1 + background_var)) * np.sqrt(min(tumour_cov, normal_cov))
            
            informative_regions.append({
                'region_id': region_id,
                'tumour_meth': tumour_meth,
                'normal_meth': normal_meth,
                'background_meth': background_mean,
                'differential': differential,
                'tumour_cov': tumour_cov,
                'normal_cov': normal_cov,
                'background_var': background_var,
                'informativeness_score': score,
                'direction': 'hyper' if tumour_meth > normal_meth else 'hypo'
            })
        
        df = pd.DataFrame(informative_regions)
        
        if len(df) > 0:
            df = df.sort_values('informativeness_score', ascending=False)
            logger.info(f"Found {len(df)} informative regions")
            logger.info(f"Top differential: {df['differential'].max():.3f}")
            logger.info(f"Regions with >50% differential: {(df['differential'] > 0.5).sum()}")
        else:
            logger.warning("No informative regions found!")
        
        return df
    
    def estimate_tumour_fraction_mle(self,
                                  cfDNA_regions: Dict[str, List[Tuple]],
                                  informative_regions: pd.DataFrame,
                                  use_top_n: Optional[int] = None) -> Dict:
        """
        Estimate tumour fraction using Maximum Likelihood Estimation.
        
        Args:
            cfDNA_regions: cfDNA PAT data by region
            informative_regions: DataFrame of informative regions
            use_top_n: Use only top N most informative regions
            
        Returns:
            Dictionary with MLE estimate and statistics
        """
        if use_top_n:
            regions_to_use = informative_regions.head(use_top_n)
        else:
            regions_to_use = informative_regions
        
        # Collect cfDNA measurements for informative regions
        observed_data = []
        
        for _, region in regions_to_use.iterrows():
            region_id = region['region_id']
            
            if region_id in cfDNA_regions:
                # Extract region start position
                try:
                    chrom, pos_range = region_id.split(':')
                    region_start = int(pos_range.split('-')[0])
                except:
                    region_start = None
                
                # Apply edge trimming to cfDNA
                cfDNA_meth, cfDNA_cov, _ = self.calculate_region_methylation(
                    cfDNA_regions[region_id], region_start, is_cfDNA=True
                )
                
                if cfDNA_cov >= self.min_region_coverage:
                    observed_data.append({
                        'cfDNA_meth': cfDNA_meth,
                        'cfDNA_cov': cfDNA_cov,
                        'tumour_meth': region['tumour_meth'],
                        'background_meth': region['background_meth'],
                        'differential': region['differential']
                    })
        
        if len(observed_data) < 10:
            logger.warning(f"Only {len(observed_data)} regions with sufficient coverage in cfDNA")
            return {'tumour_fraction': 0.0, 'confidence_interval': (0.0, 1.0), 'n_regions': len(observed_data)}
        
        logger.info(f"Using {len(observed_data)} regions for MLE")
        
        # Debug output to understand what the model is seeing
        logger.info("Sample of observed data (first 10 regions):")
        for i, obs in enumerate(observed_data[:10]):
            logger.info(f"  Region {i}: cfDNA={obs['cfDNA_meth']:.3f}, tumor={obs['tumour_meth']:.3f}, "
                       f"bg={obs['background_meth']:.3f}, diff={obs['differential']:.3f}, "
                       f"coverage={obs['cfDNA_cov']}")
        
        # Check if cfDNA is closer to tumor or background
        closer_to_tumor = 0
        for obs in observed_data:
            dist_to_tumor = abs(obs['cfDNA_meth'] - obs['tumour_meth'])
            dist_to_bg = abs(obs['cfDNA_meth'] - obs['background_meth'])
            if dist_to_tumor < dist_to_bg:
                closer_to_tumor += 1
        
        logger.info(f"cfDNA closer to tumor in {closer_to_tumor}/{len(observed_data)} regions "
                   f"({100*closer_to_tumor/len(observed_data):.1f}%)")
        
        # Define negative log-likelihood function
        def neg_log_likelihood(theta):
            if theta < 0 or theta > 1:
                return np.inf
            
            log_like = 0
            
            for obs in observed_data:
                # Expected methylation under mixture model
                expected = theta * obs['tumour_meth'] + (1 - theta) * obs['background_meth']
                
                # Binomial likelihood
                # Using normal approximation for computational efficiency
                variance = expected * (1 - expected) / obs['cfDNA_cov']
                
                if variance > 0:
                    log_like += -0.5 * np.log(2 * np.pi * variance)
                    log_like += -0.5 * ((obs['cfDNA_meth'] - expected) ** 2) / variance
            
            return -log_like
        
        # Find MLE
        result = minimize_scalar(neg_log_likelihood, bounds=(0, 1), method='bounded')
        theta_mle = result.x
        
        # Bootstrap for confidence intervals
        n_bootstrap = 1000
        bootstrap_estimates = []
        
        for _ in range(n_bootstrap):
            # Resample regions with replacement
            indices = np.random.choice(len(observed_data), len(observed_data), replace=True)
            bootstrap_data = [observed_data[i] for i in indices]
            
            # Re-estimate with bootstrap sample
            def bootstrap_neg_log_like(theta):
                if theta < 0 or theta > 1:
                    return np.inf
                
                log_like = 0
                for obs in bootstrap_data:
                    expected = theta * obs['tumour_meth'] + (1 - theta) * obs['background_meth']
                    variance = expected * (1 - expected) / obs['cfDNA_cov']
                    
                    if variance > 0:
                        log_like += -0.5 * ((obs['cfDNA_meth'] - expected) ** 2) / variance
                
                return -log_like
            
            bs_result = minimize_scalar(bootstrap_neg_log_like, bounds=(0, 1), method='bounded')
            bootstrap_estimates.append(bs_result.x)
        
        # Calculate 95% confidence interval
        ci_lower = np.percentile(bootstrap_estimates, 2.5)
        ci_upper = np.percentile(bootstrap_estimates, 97.5)
        
        return {
            'tumour_fraction': theta_mle,
            'confidence_interval': (ci_lower, ci_upper),
            'n_regions': len(observed_data),
            'log_likelihood': -neg_log_likelihood(theta_mle)
        }
    
    def estimate_tumour_fraction_bayesian(self,
                                       cfDNA_regions: Dict[str, List[Tuple]],
                                       informative_regions: pd.DataFrame,
                                       prior_alpha: float = 1.0,
                                       prior_beta: float = 10.0,
                                       use_top_n: Optional[int] = None) -> Dict:
        """
        Estimate tumour fraction using Bayesian inference.
        
        Args:
            cfDNA_regions: cfDNA PAT data by region
            informative_regions: DataFrame of informative regions
            prior_alpha: Beta distribution alpha parameter (favors low TF by default)
            prior_beta: Beta distribution beta parameter
            use_top_n: Use only top N most informative regions
            
        Returns:
            Dictionary with posterior statistics
        """
        # Similar setup as MLE
        if use_top_n:
            regions_to_use = informative_regions.head(use_top_n)
        else:
            regions_to_use = informative_regions
        
        observed_data = []
        for _, region in regions_to_use.iterrows():
            region_id = region['region_id']
            
            if region_id in cfDNA_regions:
                # Extract region start position
                try:
                    chrom, pos_range = region_id.split(':')
                    region_start = int(pos_range.split('-')[0])
                except:
                    region_start = None
                
                # Apply edge trimming to cfDNA
                cfDNA_meth, cfDNA_cov, _ = self.calculate_region_methylation(
                    cfDNA_regions[region_id], region_start, is_cfDNA=True
                )
                
                if cfDNA_cov >= self.min_region_coverage:
                    observed_data.append({
                        'cfDNA_meth': cfDNA_meth,
                        'cfDNA_cov': cfDNA_cov,
                        'tumour_meth': region['tumour_meth'],
                        'background_meth': region['background_meth']
                    })
        
        if len(observed_data) < 10:
            return {
                'posterior_mean': 0.0,
                'posterior_median': 0.0,
                'credible_interval': (0.0, 1.0),
                'n_regions': len(observed_data)
            }
        
        # Grid approximation for posterior
        theta_grid = np.linspace(0, 1, 1000)
        log_posterior = np.zeros_like(theta_grid)
        
        for i, theta in enumerate(theta_grid):
            # Prior
            log_prior = (prior_alpha - 1) * np.log(theta + 1e-10) + \
                       (prior_beta - 1) * np.log(1 - theta + 1e-10)
            
            # Likelihood
            log_like = 0
            for obs in observed_data:
                expected = theta * obs['tumour_meth'] + (1 - theta) * obs['background_meth']
                variance = expected * (1 - expected) / obs['cfDNA_cov']
                
                if variance > 0:
                    log_like += -0.5 * ((obs['cfDNA_meth'] - expected) ** 2) / variance
            
            log_posterior[i] = log_prior + log_like
        
        # Normalize posterior
        log_posterior -= np.max(log_posterior)  # For numerical stability
        posterior = np.exp(log_posterior)
        posterior /= np.trapz(posterior, theta_grid)
        
        # Calculate posterior statistics
        posterior_mean = np.trapz(posterior * theta_grid, theta_grid)
        
        # Find median
        cumulative = np.cumsum(posterior) * (theta_grid[1] - theta_grid[0])
        posterior_median = theta_grid[np.argmin(np.abs(cumulative - 0.5))]
        
        # 95% credible interval
        ci_lower = theta_grid[np.argmin(np.abs(cumulative - 0.025))]
        ci_upper = theta_grid[np.argmin(np.abs(cumulative - 0.975))]
        
        return {
            'posterior_mean': posterior_mean,
            'posterior_median': posterior_median,
            'credible_interval': (ci_lower, ci_upper),
            'n_regions': len(observed_data),
            'posterior': (theta_grid, posterior)  # For plotting if needed
        }
    
    def run_analysis(self,
                    tumour_pat_file: str,
                    normal_pat_file: str,
                    cfDNA_pat_file: str,
                    background_normal_pat_files: Optional[List[str]] = None,
                    use_top_n_regions: Optional[int] = 500) -> Dict:
        """
        Run complete patient-specific analysis pipeline.
        
        Args:
            tumour_pat_file: Path to tumour PAT file
            normal_pat_file: Path to matched normal PAT file
            cfDNA_pat_file: Path to cfDNA PAT file
            background_normal_pat_files: Optional list of other normal PAT files
            use_top_n_regions: Number of top regions to use for estimation
            
        Returns:
            Dictionary with all results
        """
        # Load data
        logger.info("Loading tumour data...")
        tumour_regions = self.load_pat_file(tumour_pat_file)
        
        logger.info("Loading normal data...")
        normal_regions = self.load_pat_file(normal_pat_file)
        
        logger.info("Loading cfDNA data...")
        cfDNA_regions = self.load_pat_file(cfDNA_pat_file)
        
        # Load background normals if provided
        background_regions = None
        if background_normal_pat_files:
            logger.info(f"Loading {len(background_normal_pat_files)} background normal samples...")
            background_regions = [self.load_pat_file(f) for f in background_normal_pat_files]
        
        # Identify informative regions
        informative_regions = self.identify_informative_regions(
            tumour_regions, normal_regions, background_regions
        )
        
        if len(informative_regions) == 0:
            return {
                'status': 'failed',
                'error': 'No informative regions found',
                'mle_estimate': None,
                'bayesian_estimate': None
            }
        
        # Run MLE estimation
        logger.info("Running MLE estimation...")
        mle_result = self.estimate_tumour_fraction_mle(
            cfDNA_regions, informative_regions, use_top_n_regions
        )
        
        # Run Bayesian estimation
        logger.info("Running Bayesian estimation...")
        bayesian_result = self.estimate_tumour_fraction_bayesian(
            cfDNA_regions, informative_regions, use_top_n=use_top_n_regions
        )
        
        return {
            'status': 'success',
            'n_informative_regions': len(informative_regions),
            'n_regions_used': mle_result['n_regions'],
            'mle_estimate': mle_result,
            'bayesian_estimate': bayesian_result,
            'informative_regions': informative_regions,
            'summary': {
                'tumour_fraction_mle': mle_result['tumour_fraction'],
                'tumour_fraction_bayesian': bayesian_result['posterior_median'],
                'confidence_interval_mle': mle_result['confidence_interval'],
                'credible_interval_bayesian': bayesian_result['credible_interval']
            }
        }


def main():
    """Example usage of the patient-specific detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Patient-specific tumour-guided cfDNA analysis')
    parser.add_argument('--tumour', required=True, help='tumour PAT file')
    parser.add_argument('--normal', required=True, help='Normal PAT file')
    parser.add_argument('--cfdna', required=True, help='cfDNA PAT file')
    parser.add_argument('--background-normals', nargs='*', help='Background normal PAT files')
    parser.add_argument('--output', required=True, help='Output file for results')
    parser.add_argument('--min-coverage', type=int, default=10, help='Minimum coverage')
    parser.add_argument('--min-differential', type=float, default=0.3, help='Minimum methylation differential')
    parser.add_argument('--top-n-regions', type=int, default=500, help='Number of top regions to use')
    parser.add_argument('--region-size', type=int, default=150, help='Region size (default 150bp for cfDNA)')
    parser.add_argument('--no-fragment-aware', action='store_true', help='Disable fragment-aware mode')
    parser.add_argument('--edge-trim', type=int, default=10, help='Base pairs to trim from edges')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PatientSpecificDetector(
        min_region_coverage=args.min_coverage,
        min_differential=args.min_differential,
        region_size=args.region_size,
        fragment_aware=not args.no_fragment_aware,
        edge_trim=args.edge_trim
    )
    
    # Run analysis
    results = detector.run_analysis(
        tumour_pat_file=args.tumour,
        normal_pat_file=args.normal,
        cfDNA_pat_file=args.cfdna,
        background_normal_pat_files=args.background_normals,
        use_top_n_regions=args.top_n_regions
    )
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        # Convert numpy types for JSON serialization
        summary = results['summary']
        summary_json = {
            'status': results['status'],
            'n_informative_regions': results.get('n_informative_regions', 0),
            'n_regions_used': results.get('n_regions_used', 0),
            'tumour_fraction_mle': float(summary['tumour_fraction_mle']),
            'tumour_fraction_bayesian': float(summary['tumour_fraction_bayesian']),
            'confidence_interval_mle': [float(x) for x in summary['confidence_interval_mle']],
            'credible_interval_bayesian': [float(x) for x in summary['credible_interval_bayesian']]
        }
        json.dump(summary_json, f, indent=2)
    
    # Print summary
    print(f"\nAnalysis complete!")
    print(f"Informative regions found: {results.get('n_informative_regions', 0)}")
    print(f"Regions used for estimation: {results.get('n_regions_used', 0)}")
    print(f"\ntumour fraction estimates:")
    print(f"  MLE: {summary['tumour_fraction_mle']:.3f} ({summary['confidence_interval_mle'][0]:.3f}-{summary['confidence_interval_mle'][1]:.3f})")
    print(f"  Bayesian: {summary['tumour_fraction_bayesian']:.3f} ({summary['credible_interval_bayesian'][0]:.3f}-{summary['credible_interval_bayesian'][1]:.3f})")


if __name__ == '__main__':
    main()