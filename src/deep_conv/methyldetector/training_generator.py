import random 
import numpy as np

class MethylationPatternSimulator:
    def __init__(self, reference_methylomes):
        """
        Initialize simulator with reference methylomes
        
        Args:
            reference_methylomes: Dictionary mapping cell types to their 
                                 methylation profiles
        """
        self.reference_methylomes = reference_methylomes
        self.cell_type_models = {}
        
        # Build statistical models for each cell type
        for cell_type, methylome in reference_methylomes.items():
            self.cell_type_models[cell_type] = self._build_model(methylome)
    
    def _build_model(self, methylome):
        """
        Build statistical model of methylation patterns
        
        This could use:
        1. Markov models for CpG correlation structure
        2. Beta distributions for methylation levels
        3. Position-specific methylation probabilities
        """
        regions = methylome.keys()
        model = {}
        
        for region in regions:
            # Extract patterns from this region
            patterns = methylome[region]
            
            # Create position-specific methylation probabilities
            cpg_positions = len(patterns[0])
            methylation_probs = np.zeros(cpg_positions)
            
            for pattern in patterns:
                methylation_probs += np.array(pattern)
            
            methylation_probs /= len(patterns)
            
            # Calculate correlation matrix between positions
            correlation_matrix = np.corrcoef(np.array(patterns).T)
            
            # Store in model
            model[region] = {
                'methylation_probs': methylation_probs,
                'correlation_matrix': correlation_matrix,
                'patterns': patterns  # Keep original for resampling
            }
        
        return model
    
    def generate_synthetic_sample(self, tumor_type, normal_type, tumor_fraction, 
                                  num_reads_per_region=100, regions=None):
        """
        Generate synthetic methylation data with specified tumor fraction
        
        Args:
            tumor_type: Tumor cell type to simulate
            normal_type: Normal cell type to simulate
            tumor_fraction: Desired tumor fraction
            num_reads_per_region: Number of reads to generate per region
            regions: Specific regions to include (None for all)
            
        Returns:
            Dictionary mapping regions to simulated methylation patterns
        """
        tumor_model = self.cell_type_models[tumor_type]
        normal_model = self.cell_type_models[normal_type]
        
        # Use intersection of regions if not specified
        if regions is None:
            regions = set(tumor_model.keys()) & set(normal_model.keys())
        
        synthetic_sample = {}
        
        for region in regions:
            # Determine number of tumor and normal reads
            tumor_reads = int(num_reads_per_region * tumor_fraction)
            normal_reads = num_reads_per_region - tumor_reads
            
            # Generate tumor reads
            tumor_patterns = self._generate_patterns(
                tumor_model[region], 
                tumor_reads
            )
            
            # Generate normal reads
            normal_patterns = self._generate_patterns(
                normal_model[region], 
                normal_reads
            )
            
            # Combine patterns
            synthetic_sample[region] = tumor_patterns + normal_patterns
        
        return synthetic_sample
    
    def _generate_patterns(self, region_model, num_patterns):
        """
        Generate methylation patterns for a specific region
        """
        # Method 1: Simple resampling with noise
        if len(region_model['patterns']) > 100:
            # If we have enough reference patterns, resample with small modifications
            patterns = []
            for _ in range(num_patterns):
                # Sample a random reference pattern
                base_pattern = random.choice(region_model['patterns'])
                
                # Add random noise (flip some CpGs)
                noise_level = 0.05  # 5% of CpGs might flip
                noisy_pattern = []
                for cpg in base_pattern:
                    if random.random() < noise_level:
                        noisy_pattern.append(1 - cpg)  # Flip the methylation state
                    else:
                        noisy_pattern.append(cpg)
                
                patterns.append(noisy_pattern)
            
            return patterns
        
        # Method 2: Generate from probability model
        else:
            patterns = []
            for _ in range(num_patterns):
                pattern = []
                for i in range(len(region_model['methylation_probs'])):
                    # Sample based on position-specific probability
                    if random.random() < region_model['methylation_probs'][i]:
                        pattern.append(1)  # Methylated
                    else:
                        pattern.append(0)  # Unmethylated
                patterns.append(pattern)
            
            # Apply correlation structure (more complex implementation would be needed)
            # This is a simplified version
            return patterns


class MethylationDataAugmenter:
    def __init__(self, seed_data):
        """
        Augment methylation data to create more training examples
        
        Args:
            seed_data: Dictionary of real methylation data to augment
        """
        self.seed_data = seed_data
    
    def augment_by_region_swapping(self, num_samples=100, 
                                  tumor_fractions=None):
        """
        Create new synthetic samples by swapping regions between samples
        
        Args:
            num_samples: Number of synthetic samples to create
            tumor_fractions: List of desired tumor fractions
            
        Returns:
            List of synthetic samples with known tumor fractions
        """
        if tumor_fractions is None:
            tumor_fractions = np.linspace(0.001, 1.0, num_samples)
        
        # Get high-TF tumor samples and control samples
        tumor_samples = [s for s in self.seed_data if s['tumor_fraction'] > 0.5]
        control_samples = [s for s in self.seed_data if s['tumor_fraction'] < 0.01]
        
        if not tumor_samples or not control_samples:
            raise ValueError("Need both tumor and control samples")
        
        synthetic_samples = []
        
        for target_tf in tumor_fractions:
            # Select random tumor and control samples
            tumor_sample = random.choice(tumor_samples)
            control_sample = random.choice(control_samples)
            
            # Create synthetic sample
            synthetic = {
                'tumor_fraction': target_tf,
                'methylation_data': {}
            }
            
            # Determine region allocation
            all_regions = set(tumor_sample['methylation_data'].keys()) & \
                          set(control_sample['methylation_data'].keys())
            
            # Calculate how many regions to take from tumor sample
            num_tumor_regions = int(len(all_regions) * target_tf)
            tumor_regions = random.sample(list(all_regions), num_tumor_regions)
            control_regions = all_regions - set(tumor_regions)
            
            # Assign regions
            for region in tumor_regions:
                synthetic['methylation_data'][region] = \
                    tumor_sample['methylation_data'][region]
            
            for region in control_regions:
                synthetic['methylation_data'][region] = \
                    control_sample['methylation_data'][region]
            
            synthetic_samples.append(synthetic)
        
        return synthetic_samples
    
    def augment_by_pattern_mixing(self, num_samples=100,
                                 tumor_fractions=None):
        """
        Create synthetic samples by mixing reads at the pattern level
        
        This preserves the read-level characteristics better than just
        swapping entire regions.
        """
        if tumor_fractions is None:
            tumor_fractions = np.linspace(0.001, 1.0, num_samples)
        
        # Similar implementation as above, but mix at pattern level
        # instead of region level
        
        return synthetic_samples
    
    def augment_with_noise_injection(self, base_samples, noise_levels=None):
        """
        Augment existing samples by adding varying levels of noise
        
        This simulates technical variation in sequencing.
        """
        if noise_levels is None:
            noise_levels = [0.01, 0.02, 0.05, 0.1]
        
        augmented_samples = []
        
        for sample in base_samples:
            for noise_level in noise_levels:
                # Create noisy copy
                noisy_sample = copy.deepcopy(sample)
                
                # Add noise to patterns
                for region, patterns in noisy_sample['methylation_data'].items():
                    noisy_patterns = []
                    for pattern in patterns:
                        noisy_pattern = []
                        for cpg in pattern:
                            if random.random() < noise_level:
                                noisy_pattern.append(1 - cpg)  # Flip state
                            else:
                                noisy_pattern.append(cpg)
                        noisy_patterns.append(noisy_pattern)
                    
                    noisy_sample['methylation_data'][region] = noisy_patterns
                
                augmented_samples.append(noisy_sample)
        
        return augmented_samples
        

def create_comprehensive_training_dataset(real_samples, 
                                         target_size=10000,
                                         tumor_fraction_distribution='exponential'):
    """
    Create comprehensive training dataset combining real and synthetic data
    
    Args:
        real_samples: Real samples with known TF to seed generation
        target_size: Desired number of training examples
        tumor_fraction_distribution: Distribution for TF values
        
    Returns:
        Dataset suitable for training TF estimation models
    """
    # 1. Extract statistical models from real data
    methylation_simulator = MethylationPatternSimulator(
        extract_reference_methylomes(real_samples)
    )
    
    # 2. Create augmentations of real data
    augmenter = MethylationDataAugmenter(real_samples)
    augmented_samples = augmenter.augment_by_region_swapping(num_samples=target_size//5)
    
    # 3. Generate fully synthetic samples
    # Generate tumor fractions with emphasis on low values
    if tumor_fraction_distribution == 'exponential':
        # More low TF values, fewer high TF values
        tumor_fractions = np.random.exponential(scale=0.05, size=target_size//2)
        tumor_fractions = np.clip(tumor_fractions, 0.0001, 0.99)
    else:
        # Uniform distribution
        tumor_fractions = np.random.uniform(0.0001, 0.99, size=target_size//2)
    
    synthetic_samples = []
    for tf in tumor_fractions:
        # Select random tumor and normal types
        tumor_type = random.choice(list(TUMOR_TYPES))
        normal_type = random.choice(list(NORMAL_TYPES))
        
        # Generate synthetic sample
        sample = methylation_simulator.generate_synthetic_sample(
            tumor_type=tumor_type,
            normal_type=normal_type,
            tumor_fraction=tf
        )
        
        synthetic_samples.append({
            'methylation_data': sample,
            'tumor_fraction': tf,
            'tumor_type': tumor_type,
            'normal_type': normal_type,
            'is_synthetic': True
        })
    
    # 4. Add noise variations
    noisy_samples = augmenter.augment_with_noise_injection(
        real_samples + augmented_samples[:100],
        noise_levels=[0.01, 0.02, 0.05]
    )
    
    # 5. Combine all datasets
    combined_dataset = real_samples + augmented_samples + synthetic_samples + noisy_samples
    
    # 6. Ensure balanced representation across TF ranges
    # Group by TF range
    tf_ranges = [
        (0, 0.001),
        (0.001, 0.01),
        (0.01, 0.05),
        (0.05, 0.1),
        (0.1, 0.5),
        (0.5, 1.0)
    ]
    
    balanced_dataset = []
    target_per_range = target_size // len(tf_ranges)
    
    for low, high in tf_ranges:
        range_samples = [s for s in combined_dataset 
                        if low <= s['tumor_fraction'] < high]
        
        # If we don't have enough, generate more
        if len(range_samples) < target_per_range:
            needed = target_per_range - len(range_samples)
            
            # Generate more for this specific range
            new_tfs = np.random.uniform(low, high, size=needed)
            for tf in new_tfs:
                # Generate synthetic sample
                sample = methylation_simulator.generate_synthetic_sample(
                    tumor_type=random.choice(list(TUMOR_TYPES)),
                    normal_type=random.choice(list(NORMAL_TYPES)),
                    tumor_fraction=tf
                )
                
                range_samples.append({
                    'methylation_data': sample,
                    'tumor_fraction': tf,
                    'is_synthetic': True
                })
        
        # Sample from available (or take all if fewer than target)
        if len(range_samples) <= target_per_range:
            balanced_dataset.extend(range_samples)
        else:
            balanced_dataset.extend(random.sample(range_samples, target_per_range))
    
    # Shuffle final dataset
    random.shuffle(balanced_dataset)
    
    return balanced_dataset