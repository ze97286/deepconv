# Training and Validation Data Overview

This document provides a detailed overview of the training and validation datasets used for the cell type deconvolution model, including their quantities (sizes), qualities (characteristics), augmentation strategy, coverage matching, the inclusion of negative examples, and visualization of the data distributions.

## Training Dataset

### Quantities
- **Original Dataset**:
  - **Low Coverage Tier**:
    - Number of files: 5
    - Samples per file: 100,000
    - Total: \( 5 \times 100,000 = 500,000 \) samples
  - **Medium Coverage Tier**:
    - Number of files: 4
    - Samples per file: 100,000
    - Total: \( 4 \times 100,000 = 400,000 \) samples
  - **High Coverage Tier**:
    - Number of files: 4
    - Samples per file: 100,000
    - Total: \( 4 \times 100,000 = 400,000 \) samples
  - **Combined Original Size**:
    \[
    500,000 + 400,000 + 400,000 = 1,300,000 \text{ samples}
    \]

- **Enhanced with Negative Examples**:
  - **Sample Fraction**: 0.01 (1% of eligible samples per cell type)
  - **Number of Cell Types**: Assumed to be 10 (adjust if different)
  - **Eligible Samples per Cell Type**:
    - Assuming each cell type is absent in 50% of samples:
      \[
      0.5 \times 1,300,000 = 650,000 \text{ samples}
      \]
  - **Selected Samples per Cell Type**:
    \[
    0.01 \times 650,000 = 6,500 \text{ samples}
    \]
  - **Added Samples per Cell Type**: 6,500 (one low-coverage variant per selected sample)
  - **Total Added Samples**:
    \[
    6,500 \times 10 = 65,000 \text{ samples}
    \]
  - **Total Enhanced Size**:
    \[
    1,300,000 + 65,000 = 1,365,000 \text{ samples}
    \]

- **Training DataLoader**:
  - Batch size: 64
  - Number of batches per epoch:
    \[
    \frac{1,365,000}{64} \approx 21,328
    \]

### Qualities
- **Composition**:
  - Low coverage tier: 500,000 samples (36.6% of original dataset)
  - Medium coverage tier: 400,000 samples (29.3% of original dataset)
  - High coverage tier: 400,000 samples (29.3% of original dataset)
  - Negative examples: 65,000 samples (4.8% of enhanced dataset)
  - The dataset is balanced across coverage tiers, with a slight emphasis on low coverage, aligning with the goal of targeting low coverage scenarios.

- **Augmentation**:
  - **Probability**: 0.8
  - **Expected Augmented Samples per Epoch**:
    \[
    0.8 \times 1,365,000 = 1,092,000
    \]
  - **Expected Non-Augmented Samples per Epoch**:
    \[
    0.2 \times 1,365,000 = 273,000
    \]
  - **Coverage Matching**:
    - Uses `coverage_matched_augmentation` to match the target clinical distribution:
      - Mean: ~9.0-9.5
      - Spread: ~5.0 to ~12.0
      - Fraction at 0: ~3%
      - Fraction >5: ~62%
    - This ensures that 80% of samples seen by the model follow the clinical distribution, focusing on low coverage scenarios.
  - **Original Distributions** (for non-augmented samples):
    - Low tier: Mean ~10.0, zero_rate=0.03
    - Medium tier: Mean ~25.0, zero_rate=0.004
    - High tier: Mean ~70.0, zero_rate=0.004
  - The 20% non-augmented samples provide diversity across different coverage regimes, while the 80% augmented samples focus on the clinical distribution.

- **Negative Examples**:
  - Added via `enhanced_negative_examples` to improve learning of absent cell types.
  - For each cell type, selects 1% of samples where the cell type is absent (`y[:, cell_idx] < 0.001`).
  - Creates one low-coverage variant per selected sample (50% coverage reduction).
  - These examples are also augmented with `coverage_matched_augmentation`, ensuring consistency with the target clinical distribution.

- **Sampling**:
  - Uses the full enhanced dataset (1,365,000 samples) with `shuffle=True`.
  - Ensures the model sees a diverse and representative set of samples each epoch, reducing the risk of overfitting to a subset.

- **Consistency**:
  - All samples (original and negative examples) are augmented consistently using `coverage_matched_augmentation` when selected for augmentation, aligning with the clinical distribution.

## Validation Dataset

### Quantities
- **Original Sizes (Per Tier)**:
  - **Tier1**: 100,000 samples
  - **T-cells**: 70,000 samples (7 types × 10,000 samples each)
  - **OAC**: 10,000 samples (12 sets × 1,000 samples each)
  - **Total per Tier**:
    \[
    100,000 + 70,000 + 10,000 = 180,000
    \]
  - **Total Across All Tiers** (4 tiers: `high`, `med`, `low`, `clinical`):
    \[
    180,000 \times 4 = 720,000
    \]

- **Subsampled Sizes (Per Tier)**:
  - **Tier1**:
    - Original: 100,000 samples
    - Target size: 20,000 samples
    - Block size: 50,000
    - Number of blocks:
      \[
      \frac{100,000}{50,000} = 2
      \]
    - Samples per block:
      \[
      \frac{20,000}{2} = 10,000
      \]
    - Subsampled size: 20,000 samples
  - **T-cells**:
    - Original: 70,000 samples
    - Target size: 15,000 samples
    - Block size: 10,000 (each type is a block)
    - Number of blocks:
      \[
      \frac{70,000}{10,000} = 7
      \]
    - Samples per block:
      \[
      \frac{15,000}{7} \approx 2,143
      \]
    - Subsampled size: 15,001 samples (2,143 × 7 = 14,998, adjusted to 15,000)
  - **OAC**:
    - Original: 10,000 samples
    - Target size: 2,000 samples
    - Block size: 1,000 (each set is a block)
    - Number of blocks:
      \[
      \frac{10,000}{1,000} = 12
      \]
    - Samples per block:
      \[
      \frac{2,000}{12} \approx 167
      \]
    - Subsampled size: 2,004 samples (167 × 12 = 2,004, adjusted to 2,000)
  - **Total per Tier**:
    \[
    20,000 + 15,000 + 2,000 = 37,000
    \]
  - **Total Across All Tiers**:
    \[
    37,000 \times 4 = 148,000
    \]

- **Validation DataLoader (Per Loader)**:
  - Batch size: 512
  - Number of batches per epoch:
    - `tier1_*`: \( \frac{20,000}{512} \approx 39 \)
    - `t-cells_*`: \( \frac{15,000}{512} \approx 29 \)
    - `oac_*`: \( \frac{2,000}{512} \approx 4 \)
  - Total batches across all loaders:
    \[
    (39 + 29 + 4) \times 4 = 288
    \]

### Qualities
- **Composition**:
  - **Tier1**: General validation set, subsampled to 20,000 samples with block-based sampling (10,000 samples per block, 2 blocks).
  - **T-cells**: Structured dataset with 7 types, subsampled to 15,000 samples (2,143 samples per type), preserving the balance across types.
  - **OAC**: Structured dataset with 12 sets (dilutions), subsampled to 2,000 samples (167 samples per set), preserving the balance across sets.
  - The validation sets provide a comprehensive evaluation across different coverage tiers (`high`, `med`, `low`, `clinical`) and subsets, with a focus on T-cells for the primary evaluation metric (R² > 0.91).

- **Augmentation**:
  - **Probability**: 0.3 (light augmentation)
  - **Coverage Matching**:
    - Uses `coverage_matched_augmentation` to match the target clinical distribution (mean ~9.0-9.5, spread ~5.0 to ~12.0, 3% at 0, ~62% >5).
    - Applied to 30% of samples, reducing the mismatch with the training data (which uses augmentation probability 0.8) while keeping the validation task manageable.
  - **Original Distributions** (for non-augmented samples):
    - High tier: Mean ~70.0, zero_rate=0.004
    - Medium tier: Mean ~25.0, zero_rate=0.004
    - Low tier: Mean ~10.0, zero_rate=0.03
    - Clinical tier: Mean ~5.0, zero_rate=0.2
  - The light augmentation ensures that the validation data distribution is closer to the training data, improving evaluation consistency.

- **Sampling**:
  - Uses block-based subsampling to preserve the structure of the datasets:
    - `T-cells`: 7 blocks (types), each sampled proportionally.
    - `OAC`: 12 blocks (sets), each sampled proportionally.
    - `Tier1`: 2 blocks, each sampled proportionally.
  - After subsampling, the `DataLoader` uses `shuffle=True` to randomly sample the subsampled dataset each epoch, ensuring a representative evaluation.

- **Consistency**:
  - The light augmentation (probability 0.3) aligns the validation data distribution more closely with the training data (probability 0.8), reducing the mismatch while keeping the evaluation task manageable.
  - The block-based subsampling ensures that the structured nature of `T-cells` and `OAC` is preserved, maintaining representativeness.

## Impact on Training and Validation
- **Training**:
  - The model sees the full enhanced dataset (1,365,000 samples) each epoch, ensuring exposure to all samples and reducing the risk of overfitting to a subset.
  - The focus on low coverage (via the low tier, negative examples, and 80% augmentation to the clinical distribution) aligns with the goal of targeting low coverage scenarios.
  - The inclusion of medium and high coverage tiers provides diversity in the 20% non-augmented samples, potentially improving generalization across different coverage regimes.
  - Consistent augmentation ensures the training data matches the target clinical distribution for 80% of samples, critical for real-world generalization.

- **Validation**:
  - The validation sets provide a comprehensive evaluation across different coverage tiers and subsets, with structured subsampling preserving the balance of types/sets in `T-cells` and `OAC`.
  - Light augmentation aligns the validation data more closely with the training data, making the evaluation more meaningful while keeping the task manageable.
  - Shuffling ensures a representative evaluation each epoch, improving robustness.

## Data Visualization
To verify that the training and validation datasets match expectations, a plotting script (`plot_data_distributions.py`) is provided to visualize key distributions using Plotly. The script generates the following plots, saved in the `plots/` directory as both interactive HTML files (for exploration) and static PNG files (for documentation):

### Training Data Plots
- **Coverage Distribution** (`train_coverage.html/png`):
  - Histogram of coverage values across a sample of 10,000 training samples.
  - Expected: A mix of the clinical distribution (mean ~9.0-9.5, ~80% of samples) and the original distributions (`low`, `med`, `high`, ~20% of samples).
- **Marker Value Distribution** (`train_marker_values.html/png`):
  - Histogram of marker values (methylation fractions) across the same sample.
  - Expected: Values between 0 and 1, with a distribution reflecting the effect of augmentation.
- **Ground Truth Proportions** (`train_proportions.html/png`):
  - Boxplot of ground truth proportions for each cell type.
  - Expected: Balanced distribution of cell type proportions, reflecting the training data composition.
- **Augmented vs. Non-Augmented Coverage** (`train_augmented_vs_non_augmented_coverage.html/png`):
  - Histogram comparing coverage for augmented vs. non-augmented samples.
  - Expected: ~80% of samples (augmented) should follow the clinical distribution, while ~20% (non-augmented) reflect the original `low`, `med`, and `high` distributions.

### Validation Data Plots (Per DataLoader)
For each validation DataLoader in `validation_dls` (e.g., `tier1_high`, `t-cells_low`, etc.):
- **Coverage Distribution** (`val_{name}_coverage.html/png`):
  - Histogram of coverage values across a sample of 5,000 validation samples.
  - Expected: A mix of the clinical distribution (mean ~9.0-9.5, ~30% of samples) and the original distributions (`high`, `med`, `low`, `clinical`, ~70% of samples). Coverage should vary across tiers.
- **Marker Value Distribution** (`val_{name}_marker_values.html/png`):
  - Histogram of marker values across the same sample.
  - Expected: Values between 0 and 1, similar to the training distribution but with slight differences due to the 0.3 augmentation probability.
- **Ground Truth Proportions** (`val_{name}_proportions.html/png`):
  - Boxplot of ground truth proportions for each cell type.
  - Expected: Similar to the training distribution, confirming that subsampling preserved the balance of cell types.

### Running the Plotting Script
1. Ensure the `enhanced_train_dl` and `validation_dls` DataLoaders are loaded (run the main data loading code).
2. Ensure the `cell_types` list is defined (`list(atlas.columns[8:])`).
3. Install Plotly and Kaleido (for PNG export):
   ```bash
   pip install plotly kaleido
   ```
4. Run the script:
   ```bash
   python plot_data_distributions.py
   ```
5. Check the generated plots in the `plots/` directory:
   - Open the `.html` files in a browser for interactive exploration.
   - View the `.png` files for static documentation.

## Notes
- The training dataset size (1,365,000 samples) is significantly larger than the validation set (148,000 samples, ~10.8% of training), which is a good balance to prevent overfitting to the validation data.
- The block-based subsampling approach ensures that the structured nature of the `T-cells` and `OAC` datasets is preserved, making the evaluation more representative of the full dataset.
- The augmentation strategy (0.8 for training, 0.3 for validation) balances the need for clinical relevance with the need for a manageable evaluation task.
- Retaining the medium and high coverage tiers in the training data provides diversity in the non-augmented samples, which may help the model generalize across different coverage regimes, even though the majority of samples are augmented to the clinical distribution.
- The Plotly-based plotting script provides interactive visualizations, allowing for detailed exploration of the data distributions, alongside static PNG files for documentation.
- **Bug Fix in Validation Data Loading**:
  - A bug in `get_validation_set_with_augmentation` was fixed where `val_dataset` was overwritten with a `Subset` object, causing `val_dataset.set_training(True)` to fail. The fix moves the `set_training(True)` call before subsampling, ensuring it is applied to the `AugmentedTissueDataset` object.