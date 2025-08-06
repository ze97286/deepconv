#!/usr/bin/env python3
"""
Validation framework for patient-specific tumor detection.
Tests the approach using synthetic mixtures with known tumor fractions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import gzip
from typing import Dict, List, Tuple
import logging
from patient_specific_detector import PatientSpecificDetector
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientSpecificValidator:
    """Validates patient-specific detection using known mixtures."""
    
    def __init__(self, detector: PatientSpecificDetector):
        self.detector = detector
        
    def create_synthetic_mixture(self,
                               tumor_regions: Dict[str, List[Tuple]],
                               normal_regions: Dict[str, List[Tuple]],
                               tumor_fraction: float,
                               target_coverage: int = 20,
                               output_file: str = None) -> Dict[str, List[Tuple]]:
        """
        Create a synthetic cfDNA mixture with known tumor fraction.
        
        Args:
            tumor_regions: Tumor PAT data
            normal_regions: Normal PAT data
            tumor_fraction: Desired tumor fraction (0-1)
            target_coverage: Target coverage depth
            output_file: Optional file to save mixture
            
        Returns:
            Synthetic cfDNA regions
        """
        logger.info(f"Creating synthetic mixture with TF={tumor_fraction:.3f}")
        
        # Find common regions
        common_regions = set(tumor_regions.keys()) & set(normal_regions.keys())
        
        synthetic_regions = {}
        
        for region_id in common_regions:
            tumor_data = tumor_regions[region_id]
            normal_data = normal_regions[region_id]
            
            # Skip if no data
            if not tumor_data or not normal_data:
                continue
            
            # Merge CpG positions from both
            all_positions = {}
            
            # Add tumor positions
            for pos, pattern, count in tumor_data:
                all_positions[pos] = {'tumor_pattern': pattern, 'tumor_count': count, 
                                    'normal_pattern': None, 'normal_count': 0}
            
            # Add normal positions
            for pos, pattern, count in normal_data:
                if pos in all_positions:
                    all_positions[pos]['normal_pattern'] = pattern
                    all_positions[pos]['normal_count'] = count
                else:
                    all_positions[pos] = {'tumor_pattern': None, 'tumor_count': 0,
                                        'normal_pattern': pattern, 'normal_count': count}
            
            # Create mixture for this region
            region_mixture = []
            
            for pos, data in all_positions.items():
                # Sample from tumor and normal based on fraction
                tumor_reads = 0
                normal_reads = 0
                
                if data['tumor_count'] > 0 and tumor_fraction > 0:
                    tumor_reads = np.random.binomial(
                        int(target_coverage * tumor_fraction), 
                        min(1.0, data['tumor_count'] / max(1, data['tumor_count'] + data['normal_count']))
                    )
                
                if data['normal_count'] > 0 and (1 - tumor_fraction) > 0:
                    normal_reads = np.random.binomial(
                        int(target_coverage * (1 - tumor_fraction)),
                        min(1.0, data['normal_count'] / max(1, data['tumor_count'] + data['normal_count']))
                    )
                
                # Combine patterns (simplified - in reality would need proper merging)
                if tumor_reads > 0 and data['tumor_pattern']:
                    region_mixture.append((pos, data['tumor_pattern'], tumor_reads))
                if normal_reads > 0 and data['normal_pattern']:
                    # If same position, need to merge patterns - here we simplify
                    if tumor_reads == 0:
                        region_mixture.append((pos, data['normal_pattern'], normal_reads))
            
            if region_mixture:
                synthetic_regions[region_id] = region_mixture
        
        # Save if requested
        if output_file:
            self.save_synthetic_pat(synthetic_regions, output_file)
        
        return synthetic_regions
    
    def save_synthetic_pat(self, regions: Dict[str, List[Tuple]], output_file: str):
        """Save synthetic regions to PAT file format."""
        with gzip.open(output_file, 'wt') as f:
            # Group by chromosome
            by_chrom = {}
            for region_id, data in regions.items():
                chrom = region_id.split(':')[0]
                if chrom not in by_chrom:
                    by_chrom[chrom] = []
                by_chrom[chrom].extend(data)
            
            # Write each chromosome
            for chrom in sorted(by_chrom.keys()):
                f.write(f">{chrom}\n")
                # Sort by position
                positions = sorted(by_chrom[chrom], key=lambda x: x[0])
                for pos, pattern, count in positions:
                    f.write(f"{pos}\t{pattern}\t{count}\n")
    
    def run_validation_series(self,
                            tumor_pat_file: str,
                            normal_pat_file: str,
                            tumor_fractions: List[float] = None,
                            n_replicates: int = 3) -> pd.DataFrame:
        """
        Run validation across multiple tumor fractions.
        
        Args:
            tumor_pat_file: Tumor PAT file
            normal_pat_file: Normal PAT file  
            tumor_fractions: List of tumor fractions to test
            n_replicates: Number of replicates per fraction
            
        Returns:
            DataFrame with validation results
        """
        if tumor_fractions is None:
            tumor_fractions = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        
        # Load data once
        logger.info("Loading tumor and normal data...")
        tumor_regions = self.detector.load_pat_file(tumor_pat_file)
        normal_regions = self.detector.load_pat_file(normal_pat_file)
        
        # Find informative regions once
        logger.info("Identifying informative regions...")
        informative_regions = self.detector.identify_informative_regions(
            tumor_regions, normal_regions
        )
        
        if len(informative_regions) == 0:
            logger.error("No informative regions found!")
            return pd.DataFrame()
        
        logger.info(f"Found {len(informative_regions)} informative regions")
        
        results = []
        
        for tf in tumor_fractions:
            for rep in range(n_replicates):
                logger.info(f"Testing TF={tf}, replicate {rep+1}/{n_replicates}")
                
                # Create synthetic mixture
                synthetic = self.create_synthetic_mixture(
                    tumor_regions, normal_regions, tf, target_coverage=20
                )
                
                # Estimate tumor fraction
                mle_result = self.detector.estimate_tumor_fraction_mle(
                    synthetic, informative_regions, use_top_n=500
                )
                
                bayesian_result = self.detector.estimate_tumor_fraction_bayesian(
                    synthetic, informative_regions, use_top_n=500
                )
                
                results.append({
                    'true_tf': tf,
                    'replicate': rep,
                    'mle_estimate': mle_result['tumor_fraction'],
                    'mle_ci_lower': mle_result['confidence_interval'][0],
                    'mle_ci_upper': mle_result['confidence_interval'][1],
                    'bayesian_estimate': bayesian_result['posterior_median'],
                    'bayesian_ci_lower': bayesian_result['credible_interval'][0],
                    'bayesian_ci_upper': bayesian_result['credible_interval'][1],
                    'n_regions_used': mle_result['n_regions']
                })
        
        return pd.DataFrame(results)
    
    def plot_validation_results(self, results_df: pd.DataFrame, output_dir: str = '.'):
        """Create validation plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Setup plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Accuracy plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MLE accuracy
        ax1.errorbar(results_df.groupby('true_tf')['true_tf'].first(),
                    results_df.groupby('true_tf')['mle_estimate'].mean(),
                    yerr=results_df.groupby('true_tf')['mle_estimate'].std(),
                    fmt='o', capsize=5, label='MLE estimate')
        ax1.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5, label='Perfect estimation')
        ax1.set_xlabel('True Tumor Fraction')
        ax1.set_ylabel('Estimated Tumor Fraction (MLE)')
        ax1.set_title('MLE Estimation Accuracy')
        ax1.legend()
        ax1.set_xlim(-0.02, 0.52)
        ax1.set_ylim(-0.02, 0.52)
        
        # Bayesian accuracy
        ax2.errorbar(results_df.groupby('true_tf')['true_tf'].first(),
                    results_df.groupby('true_tf')['bayesian_estimate'].mean(),
                    yerr=results_df.groupby('true_tf')['bayesian_estimate'].std(),
                    fmt='o', capsize=5, label='Bayesian estimate', color='orange')
        ax2.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5, label='Perfect estimation')
        ax2.set_xlabel('True Tumor Fraction')
        ax2.set_ylabel('Estimated Tumor Fraction (Bayesian)')
        ax2.set_title('Bayesian Estimation Accuracy')
        ax2.legend()
        ax2.set_xlim(-0.02, 0.52)
        ax2.set_ylim(-0.02, 0.52)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Log-scale plot for low fractions
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Filter for non-zero true fractions
        nonzero_df = results_df[results_df['true_tf'] > 0]
        
        ax.errorbar(nonzero_df.groupby('true_tf')['true_tf'].first(),
                   nonzero_df.groupby('true_tf')['mle_estimate'].mean(),
                   yerr=nonzero_df.groupby('true_tf')['mle_estimate'].std(),
                   fmt='o', capsize=5, label='MLE')
        ax.errorbar(nonzero_df.groupby('true_tf')['true_tf'].first(),
                   nonzero_df.groupby('true_tf')['bayesian_estimate'].mean(),
                   yerr=nonzero_df.groupby('true_tf')['bayesian_estimate'].std(),
                   fmt='s', capsize=5, label='Bayesian')
        
        ax.plot([0.0001, 1], [0.0001, 1], 'k--', alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('True Tumor Fraction')
        ax.set_ylabel('Estimated Tumor Fraction')
        ax.set_title('Low Fraction Detection Performance')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_low_fraction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Calculate performance metrics
        metrics = []
        for tf in results_df['true_tf'].unique():
            tf_data = results_df[results_df['true_tf'] == tf]
            
            # Detection rate (estimate > 0.5 * true value)
            if tf > 0:
                mle_detected = (tf_data['mle_estimate'] > 0.5 * tf).mean()
                bayesian_detected = (tf_data['bayesian_estimate'] > 0.5 * tf).mean()
            else:
                # For true negatives, check if estimate < 0.001
                mle_detected = (tf_data['mle_estimate'] < 0.001).mean()
                bayesian_detected = (tf_data['bayesian_estimate'] < 0.001).mean()
            
            metrics.append({
                'true_tf': tf,
                'mle_mean': tf_data['mle_estimate'].mean(),
                'mle_std': tf_data['mle_estimate'].std(),
                'mle_detection_rate': mle_detected,
                'bayesian_mean': tf_data['bayesian_estimate'].mean(),
                'bayesian_std': tf_data['bayesian_estimate'].std(),
                'bayesian_detection_rate': bayesian_detected,
                'n_regions': tf_data['n_regions_used'].mean()
            })
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(output_dir / 'validation_metrics.csv', index=False)
        
        # Print summary
        print("\nValidation Summary:")
        print("==================")
        print(f"Total informative regions: {results_df['n_regions_used'].iloc[0]}")
        print("\nMLE Performance:")
        print(f"  False positive rate (TF=0): {100*(1-metrics_df[metrics_df['true_tf']==0]['mle_detection_rate'].iloc[0]):.1f}%")
        print(f"  Detection rate at 1%: {100*metrics_df[metrics_df['true_tf']==0.01]['mle_detection_rate'].iloc[0]:.1f}%")
        print(f"  Mean absolute error: {np.mean(np.abs(metrics_df['true_tf'] - metrics_df['mle_mean'])):.4f}")
        
        print("\nBayesian Performance:")
        print(f"  False positive rate (TF=0): {100*(1-metrics_df[metrics_df['true_tf']==0]['bayesian_detection_rate'].iloc[0]):.1f}%")
        print(f"  Detection rate at 1%: {100*metrics_df[metrics_df['true_tf']==0.01]['bayesian_detection_rate'].iloc[0]:.1f}%")
        print(f"  Mean absolute error: {np.mean(np.abs(metrics_df['true_tf'] - metrics_df['bayesian_mean'])):.4f}")
        
        return metrics_df


def main():
    """Run validation on patient-specific detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate patient-specific tumor detection')
    parser.add_argument('--tumor', required=True, help='Tumor PAT file')
    parser.add_argument('--normal', required=True, help='Normal PAT file')
    parser.add_argument('--output-dir', default='validation_results', help='Output directory')
    parser.add_argument('--n-replicates', type=int, default=5, help='Replicates per fraction')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PatientSpecificDetector(
        min_region_coverage=10,
        min_differential=0.3
    )
    
    # Initialize validator
    validator = PatientSpecificValidator(detector)
    
    # Run validation
    results = validator.run_validation_series(
        args.tumor,
        args.normal,
        n_replicates=args.n_replicates
    )
    
    # Plot results
    validator.plot_validation_results(results, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()