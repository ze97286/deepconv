# DeepConv: Deep Learning for Cell-Free DNA Cell-type Deconvolution

## Overview
DeepConv is a deep learning approach for deconvoluting cell-type proportions from cell-free DNA methylation data. The model learns to estimate the relative contributions of different cell types in a mixture by analysing methylation patterns across genomic markers.

## Background

### Cell-free DNA Methylation
Cell-free DNA (cfDNA) in blood plasma consists of DNA fragments released by cells throughout the body during natural cell death (apoptosis) or other cellular processes. These fragments retain the methylation patterns of their cells of origin, providing a "signature" that can be used to identify their source tissue.

### The Deconvolution Problem
The fundamental question in cfDNA deconvolution is: Given a mixture of DNA from multiple cell types, can we determine what proportion came from each type? Mathematically, this can be formulated as:

M = R × P

Where each matrix represents:

#### Reference Matrix (R)
- Shape: (regions × cell_types)
- Each column represents a cell type's methylation profile
- Each row is a genomic region (marker)
- Values are between 0-1 representing methylation level:
  * 0: Completely unmethylated
  * 1: Completely methylated
- Obtained from reference samples of pure cell types
- Usually sparse, as regions are selected to be differentially methylated across cell types

#### Proportion Matrix (P)
- Shape: (cell_types × samples)
- Each column represents one sample's composition
- Each row is a cell type
- Values represent the fraction of DNA from each cell type
- Subject to biological constraints:
  * Non-negative: All values ≥ 0
  * Sum-to-one: Each column sums to 1
- This is what we're trying to estimate

#### Methylation Matrix (M)
- Shape: (regions × samples)
- Each column represents one mixed sample
- Each row is a genomic region matching R
- Values are between 0-1 representing observed methylation level
- In practice, these values are derived from sequencing data:
  * Number of methylated reads / Total number of reads
- Quality depends on sequencing coverage

### Additional Practical Considerations

#### Coverage Information
- Not all regions are covered equally in sequencing
- Coverage can vary from 0 to hundreds of reads
- Low coverage regions have less reliable methylation estimates
- Coverage information is crucial for:
  * Weighting reliable measurements more heavily
  * Handling missing or low-confidence data

#### Technical Biases
- Sequencing errors
- PCR amplification biases
- DNA fragmentation patterns
- Batch effects

#### Biological Complexity
- Cell type similarities
- Tissue-specific methylation patterns
- Biological variation within cell types
- Rare cell types (<1% of mixture)

This complex interplay of factors makes the deconvolution problem challenging for traditional optimisation approaches, motivating our deep learning solution.

This formulation is subject to two key biological constraints:
1. Non-negativity: Proportions cannot be negative (P ≥ 0)
2. Sum-to-one: Proportions must sum to 1 (ΣP = 1)

### Traditional Approaches
Most existing methods solve this problem using Non-negative Least Squares (NNLS), which minimises ||M - RP||² subject to P ≥ 0. While effective, these approaches:
- Assume linear relationships
- May not fully capture complex interactions
- Can be sensitive to noise and missing data
- Often struggle with rare cell types (<1% proportion)

## Model Architecture

### Input Features
- Methylation values (0-1) for each genomic region
- Coverage information (read counts) for each region

### Design Rationale

1. **Parallel Encoders**
   - Separate processing paths for methylation and coverage
   - Allows the model to learn different feature patterns:
     * Methylation encoder: Pattern recognition in methylation signals
     * Coverage encoder: Quality and confidence assessment
   - Each encoder can specialize in its domain

2. **Dimensionality Choices**
   - Initial expansion to 512 dimensions:
     * Allows learning of rich feature representations
     * Captures complex interactions between regions
   - Reduction to 256 dimensions:
     * Compresses information to most relevant features
     * Reduces risk of overfitting

3. **Regularization Strategy**
   - Heavy dropout (0.4):
     * Prevents over-reliance on specific markers
     * Improves robustness to missing data
   - Batch normalization:
     * Stabilizes training
     * Handles varying scales of methylation and coverage

4. **Feature Fusion**
   - Concatenation rather than addition:
     * Preserves distinct information from both streams
     * Allows model to learn optimal combination
   - No additional processing:
     * Lets final layer learn direct mapping to proportions

5. **Output Design**
   - Single linear layer to proportions:
     * Simple mapping from learned features
     * Avoids overfitting in final stages
   - Softmax activation:
     * Ensures biological constraints
     * Naturally handles proportion requirements

### Architecture motivation

1. **Biological Motivation**
   - Mirrors the two key aspects of methylation data:
     * Signal (methylation values)
     * Confidence (coverage)
   - Handles sparsity and noise in real data

2. **Technical Advantages**
   - End-to-end differentiable
   - Relatively simple to train
   - Computationally efficient
   - Easy to interpret feature importance

3. **Practical Benefits**
   - Can process variable-length inputs
   - Robust to missing data
   - Scales well with number of markers
   - Easily adaptable to different reference panels

### Network Structure
1. Parallel encoders for methylation and coverage data:
   - Process methylation patterns
   - Account for varying coverage depths
2. Feature fusion through concatenation
3. Final layer producing cell type proportions

### Non-negativity and Sum-to-one Constraints
The model enforces biological constraints through its architecture:
1. Softmax activation in the final layer ensures:
   - All proportions are positive (0-1 range)
   - Proportions sum to 1
   - Differentiable end-to-end training

This architectural choice is superior to post-processing normalisation because:
- It incorporates constraints during training
- Allows the model to learn within the constrained space
- Maintains differentiability for gradient-based optimisation

### Key Components
- Batch normalisation: Stabilises training with varying methylation levels
- Dropout (0.4): Prevents overfitting to specific methylation patterns
- ReLU activation: Introduces non-linearity while maintaining non-negativity

## Training Strategy

### Data Generation
- Synthetic mixtures created from reference methylation profiles
- Realistic coverage simulation using negative binomial distribution
- Technical noise simulation

### Training Process
- KL divergence loss function
- Adam optimiser with learning rate scheduling
- Early stopping based on validation loss
- L2 regularisation to prevent overfitting

## Usage

### Installation
```bash
git clone https://github.com/username/deepconv
cd deepconv
```

### Training a model 
```
cd deepconv/src
python -m deep_conv.deconvolution.train --batch_size 32 --n_train 100000 --n_val 20000 --atlas_path /mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed --model_save_path /users/zetzioni/sharedscratch/deepconv/src/deep_conv/saved_models/
```

### Evaluating a model using diluted admixtures
```
cd deepconv/src
python -m deep_conv.deconvolution.estimate_cell_type \
--model_path /users/zetzioni/sharedscratch/deconvolution_model.pt \
--cell_type CD4-T-cells \
--atlas_path /mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/Atlas_blood+gi+tum.U100.l4.bed \
--wgbs_tools_exec_path /users/zetzioni/sharedscratch/wgbs_tools/wgbstools \
--pats_path /mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/Benchmark/pat/blood+gi+tum.U100/Song/mixed/CD4 \
--output_path /users/zetzioni/sharedscratch/cd4 
```

### Requirements
Python 3.9+
PyTorch
NumPy
Pandas
scikit-learn
wgbs_tools
plotly
kaleido



