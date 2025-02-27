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
- Marker values: Methylation values for each marker region (can contain NaNs where coverage=0)
- Coverage: Number of reads covering each marker value

### Core Architecture

1. Feature Extraction:

Uses a two-layer network to transform each marker value into a rich feature representation
The non-linear LeakyReLU activation allows capturing complex methylation patterns


2. Cell Type-Specific Aggregation:

Each marker is assigned to exactly one target cell type (via target_ids)
Features from markers targeting the same cell type are aggregated
Features are weighted by coverage, giving more reliable markers more influence
Effective zero coverage handling ensures NaN markers don't contribute


3. Proportion Prediction (Encoder):

The aggregated features for each cell type are processed through a neural network
The output is transformed via sigmoid and normalised to ensure proportions sum to 1


4. Marker Reconstruction (Decoder):

For interpretability, the model can reconstruct the original marker values
This helps ensure the predicted proportions explain the observed methylation patterns


### Loss function - Enhanced Weighted Approach for Cell-Type Deconvolution
The loss function implements a specialised approach to tackle the challenge of low SNR cell types like CD4/CD8:

#### Core Components

1. Concentration-Weighted Loss:
Applies importance weights based on true cell type concentrations
Creates progressively stronger penalties for higher CD4/CD8 concentrations
For CD4/CD8 cells at 10% concentration, the weight is ~8x stronger than baseline

2. Asymmetric Error Penalties:
Differentiates between underestimation and overestimation
Applies an additional 1.5x penalty to CD4/CD8 underestimation
Effectively prioritises reducing false negatives for these critical cell types

1. Coverage-Weighted Reconstruction:
Weights reconstruction errors by read coverage
Places more emphasis on markers with higher confidence (more reads)
Ignores markers with zero coverage (NaN values)


4. Combined Loss With Balance Control:
Uses alpha/beta parameters to balance proportion prediction vs. reconstruction
Typically weights proportion prediction much higher (alpha=0.999)
Maintains reconstruction as a regularising constraint (beta=0.001)

#### Design Rationale
The loss function's design addresses key challenges in methylation-based deconvolution:

Low SNR Compensation: CD4/CD8 cells have 8x lower SNR than OAC, requiring special handling
Concentration-Dependent Scaling: Higher concentration predictions need higher accuracy
Penalty Asymmetry: Underestimation has worse clinical implications than overestimation
Coverage Integration: Leverages sequencing depth as confidence measure

This approach effectively focuses the model's learning on the most challenging aspects of the problem, improving performance on low-SNR cell types while maintaining overall accuracy.

## Usage

### Installation
```bash
git clone https://github.com/username/deepconv
cd deepconv
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



