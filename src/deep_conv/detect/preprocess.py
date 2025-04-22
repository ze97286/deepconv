import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split

class cfDNAMethylationDataset(Dataset):
    """
    Dataset for cfDNA methylation data with cell type specific markers
    """
    def __init__(self, marker_values, coverage, y_true):
        """
        Args:
            marker_values: np.ndarray of shape [num_samples, num_markers]
            coverage: np.ndarray of shape [num_samples, num_markers]
            y_true: np.ndarray of shape [num_samples]
        """
        self.marker_values = torch.tensor(marker_values, dtype=torch.float32)
        self.coverage = torch.tensor(coverage, dtype=torch.float32)
        self.y_true = torch.tensor(y_true, dtype=torch.float32).unsqueeze(1)  # Shape: [num_samples, 1]
        
    def __len__(self):
        return len(self.y_true)
    
    def __getitem__(self, idx):
        return self.marker_values[idx], self.coverage[idx], self.y_true[idx]


def load_and_preprocess_data(
    marker_values_path, 
    coverage_path, 
    ground_truth_path, 
    atlas_path, 
    target_cell_type="OAC", 
    target_cell_idx=9,
    test_size=0.2, 
    val_size=0.2, 
    random_state=42
):
    """
    Load and preprocess cfDNA methylation data
    
    Args:
        marker_values_path: Path to marker values parquet file
        coverage_path: Path to coverage parquet file
        ground_truth_path: Path to ground truth parquet file
        atlas_path: Path to atlas CSV/TSV file
        target_cell_type: Target cell type name in atlas
        target_cell_idx: Index of target cell type in ground truth
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        random_state: Random seed for splitting
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        num_markers: Number of markers used
    """
    print(f"Loading data from {marker_values_path}, {coverage_path}, {ground_truth_path}...")
    
    # Load marker values and coverage data
    marker_values_df = pd.read_parquet(marker_values_path)
    coverage_df = pd.read_parquet(coverage_path)
    
    # Load ground truth
    ground_truth_df = pd.read_parquet(ground_truth_path)
    y_true = ground_truth_df.iloc[:, target_cell_idx].values
    
    # Load atlas and extract relevant markers
    print(f"Loading atlas from {atlas_path} and extracting markers for {target_cell_type}...")
    atlas = pd.read_csv(atlas_path, sep="\t")
    target_markers = atlas[atlas.target == target_cell_type]
    target_marker_indices = target_markers.index.values
    
    print(f"Found {len(target_marker_indices)} markers for {target_cell_type}")
    
    # Extract relevant markers from data
    marker_values = marker_values_df.iloc[target_marker_indices][marker_values_df.columns[2:]].values.T
    coverage = coverage_df.iloc[target_marker_indices][coverage_df.columns[2:]].values.T  

    # Data summary
    print(f"Marker values shape: {marker_values.shape}")
    print(f"Coverage shape: {coverage.shape}")
    print(f"Ground truth shape: {y_true.shape}")
    
    # Check for missing values
    nan_pct = np.isnan(marker_values).mean() * 100
    zero_cov_pct = (coverage == 0).mean() * 100
    print(f"Percentage of NaN marker values: {nan_pct:.2f}%")
    print(f"Percentage of zero coverage: {zero_cov_pct:.2f}%")
    
    # Split data into train, validation and test sets
    # First split into train+val and test
    train_val_indices, test_indices = train_test_split(
        np.arange(len(y_true)), 
        test_size=test_size, 
        random_state=random_state,
        stratify=np.digitize(y_true, bins=np.linspace(0, 1, 5))  # Stratify by binned concentration
    )
    
    # Then split train+val into train and val
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size/(1-test_size),
        random_state=random_state,
        stratify=np.digitize(y_true[train_val_indices], bins=np.linspace(0, 1, 5))
    )
    
    # Create datasets
    train_dataset = cfDNAMethylationDataset(
        marker_values[train_indices], 
        coverage[train_indices], 
        y_true[train_indices]
    )
    
    val_dataset = cfDNAMethylationDataset(
        marker_values[val_indices], 
        coverage[val_indices], 
        y_true[val_indices]
    )
    
    test_dataset = cfDNAMethylationDataset(
        marker_values[test_indices], 
        coverage[test_indices], 
        y_true[test_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Created data loaders with {len(train_dataset)} training, "
          f"{len(val_dataset)} validation, and {len(test_dataset)} test samples")
    
    return train_loader, val_loader, test_loader, marker_values.shape[1]


def analyze_data_characteristics(train_loader, val_loader):
    """
    Analyze data characteristics to inform model design
    """
    # Initialize accumulators
    total_samples = 0
    nan_count = 0
    zero_cov_count = 0
    marker_value_sum = 0
    marker_value_sq_sum = 0
    coverage_sum = 0
    coverage_sq_sum = 0
    y_true_sum = 0
    y_true_sq_sum = 0
    
    # Process all batches
    for loader in [train_loader, val_loader]:
        for marker_values, coverage, y_true in loader:
            batch_size = marker_values.size(0)
            total_samples += batch_size
            
            # Count NaNs and zeros
            nan_count += torch.isnan(marker_values).sum().item()
            zero_cov_count += (coverage == 0).sum().item()
            
            # Replace NaNs with zeros for statistics calculation
            marker_values_clean = torch.nan_to_num(marker_values, nan=0.0)
            
            # Update sums for mean and std calculation
            marker_value_sum += marker_values_clean.sum().item()
            marker_value_sq_sum += (marker_values_clean ** 2).sum().item()
            
            coverage_sum += coverage.sum().item()
            coverage_sq_sum += (coverage ** 2).sum().item()
            
            y_true_sum += y_true.sum().item()
            y_true_sq_sum += (y_true ** 2).sum().item()
    
    # Calculate total elements
    total_elements = total_samples * marker_values.size(1)
    
    # Calculate statistics
    marker_value_mean = marker_value_sum / total_elements
    marker_value_std = np.sqrt(marker_value_sq_sum / total_elements - marker_value_mean ** 2)
    
    coverage_mean = coverage_sum / total_elements
    coverage_std = np.sqrt(coverage_sq_sum / total_elements - coverage_mean ** 2)
    
    y_true_mean = y_true_sum / total_samples
    y_true_std = np.sqrt(y_true_sq_sum / total_samples - y_true_mean ** 2)
    
    # Print results
    print("\nData Characteristics Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"NaN percentage: {nan_count / total_elements * 100:.2f}%")
    print(f"Zero coverage percentage: {zero_cov_count / total_elements * 100:.2f}%")
    print(f"Marker value - Mean: {marker_value_mean:.4f}, Std: {marker_value_std:.4f}")
    print(f"Coverage - Mean: {coverage_mean:.4f}, Std: {coverage_std:.4f}")
    print(f"Cancer concentration - Mean: {y_true_mean:.4f}, Std: {y_true_std:.4f}")
    
    return {
        "marker_value_mean": marker_value_mean,
        "marker_value_std": marker_value_std,
        "coverage_mean": coverage_mean,
        "coverage_std": coverage_std,
        "y_true_mean": y_true_mean,
        "y_true_std": y_true_std,
        "nan_percentage": nan_count / total_elements * 100,
        "zero_cov_percentage": zero_cov_count / total_elements * 100
    }


# Main execution function
def prepare_data_for_training(
    data_dir,
    atlas_path=None,
    target_cell_type="OAC",
    target_cell_idx=9,
):
    """
    Prepare data for training
    
    Args:
        data_dir: Directory containing parquet files
        atlas_path: Path to atlas file (if None, use default path)
        target_cell_type: Target cell type
        target_cell_idx: Target cell index in ground truth
        batch_size: Batch size for data loaders
        
    Returns:
        train_loader, val_loader, test_loader, num_markers, data_stats
    """
    # Define file paths
    marker_values_path = os.path.join(data_dir, "marker_values.parquet")
    coverage_path = os.path.join(data_dir, "coverage.parquet")
    ground_truth_path = os.path.join(data_dir, "ground_truth_y.parquet")
    
    # Load and preprocess data
    train_loader, val_loader, test_loader, num_markers = load_and_preprocess_data(
        marker_values_path=marker_values_path,
        coverage_path=coverage_path,
        ground_truth_path=ground_truth_path,
        atlas_path=atlas_path,
        target_cell_type=target_cell_type,
        target_cell_idx=target_cell_idx,
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )
    
    # Analyze data characteristics
    data_stats = analyze_data_characteristics(train_loader, val_loader)
    
    return train_loader, val_loader, test_loader, num_markers, data_stats


if __name__ == "__main__":
    data_dir = "/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval_single_cell_clinical/OAC/"
    atlas_path = "/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed"

    train_loader, val_loader, test_loader, num_markers, data_stats = prepare_data_for_training(
        data_dir=data_dir,
        atlas_path=atlas_path,
        target_cell_type="OAC",
        target_cell_idx=9
    )
    
    print(f"Prepared data with {num_markers} markers")
    print(f"Data statistics: {data_stats}")