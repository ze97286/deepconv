import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from tqdm import tqdm
from deep_conv.deconvolution.deepconv import *

# Ensure plots directory exists
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)


# Function to sample data from a DataLoader
def sample_from_dataloader(dataloader, num_samples):
    all_X, all_coverage, all_y = [], [], []
    all_indices = np.arange(len(dataloader.dataset))
    np.random.shuffle(all_indices)
    sampled_indices = all_indices[:num_samples]

    for idx in tqdm(sampled_indices, desc="Sampling data"):
        batch = dataloader.dataset[idx]
        all_X.append(batch["X"].numpy())
        all_coverage.append(batch["coverage"].numpy())
        all_y.append(batch["y"].numpy())

    return np.stack(all_X), np.stack(all_coverage), np.stack(all_y)

# Function to plot coverage distribution
def plot_coverage_distribution(coverage, title, filename):
    fig = px.histogram(
        x=coverage.flatten(),
        nbins=100,
        histnorm="density",
        title=title,
        labels={"x": "Coverage", "y": "Density"},
        range_x=[0, 100],
    )
    fig.update_layout(xaxis_title="Coverage", yaxis_title="Density", bargap=0.1)
    fig.write_html(os.path.join(plots_dir, f"{filename}.html"))
    fig.write_image(os.path.join(plots_dir, f"{filename}.png"))

# Function to plot marker value distribution
def plot_marker_value_distribution(X, title, filename):
    fig = px.histogram(
        x=X.flatten(),
        nbins=100,
        histnorm="density",
        title=title,
        labels={"x": "Marker Value (Methylation Fraction)", "y": "Density"},
        range_x=[0, 1],
    )
    fig.update_layout(
        xaxis_title="Marker Value (Methylation Fraction)",
        yaxis_title="Density",
        bargap=0.1,
    )
    fig.write_html(os.path.join(plots_dir, f"{filename}.html"))
    fig.write_image(os.path.join(plots_dir, f"{filename}.png"))

# Function to plot ground truth proportions
def plot_ground_truth_proportions(y, cell_types, title, filename):
    fig = go.Figure()
    for i, cell_type in enumerate(cell_types):
        fig.add_trace(go.Box(x=y[:, i], name=cell_type, orientation="h"))

    fig.update_layout(
        title=title,
        xaxis_title="Proportion",
        yaxis_title="Cell Type",
        xaxis=dict(range=[0, 1]),
    )
    fig.write_html(os.path.join(plots_dir, f"{filename}.html"))
    fig.write_image(os.path.join(plots_dir, f"{filename}.png"))

# Updated function to plot augmented vs non-augmented mean coverage
def plot_augmented_vs_non_augmented_coverage(dataloader, num_samples, title, filename):
    # Ensure augmentation is enabled
    dataloader.dataset.set_training(True)
    
    # Collect augmented and non-augmented coverage values
    augmented_coverage = []
    non_augmented_coverage = []

    all_indices = np.arange(len(dataloader.dataset))
    np.random.shuffle(all_indices)
    sampled_indices = all_indices[:num_samples]

    for idx in tqdm(sampled_indices, desc="Sampling augmented/non-augmented"):
        batch = dataloader.dataset[idx]
        coverage = batch["coverage"].numpy()
        is_augmented = batch["is_augmented"]
        
        # Compute mean coverage per sample
        mean_coverage = np.mean(coverage)
        
        if is_augmented:
            augmented_coverage.append(mean_coverage)
        else:
            non_augmented_coverage.append(mean_coverage)

    # Convert to arrays
    augmented_coverage = np.array(augmented_coverage)
    non_augmented_coverage = np.array(non_augmented_coverage)

    # Log the proportions for debugging
    total_augmented = len(augmented_coverage)
    total_non_augmented = len(non_augmented_coverage)
    total_samples = total_augmented + total_non_augmented
    print(f"Augmented samples: {total_augmented}, Non-Augmented samples: {total_non_augmented}")
    print(f"Proportion augmented: {total_augmented / total_samples:.3f}")

    # Create histogram with two traces, using counts
    fig = go.Figure()
    if augmented_coverage.size > 0:
        fig.add_trace(
            go.Histogram(
                x=augmented_coverage,
                name="Augmented",
                nbinsx=100,
                opacity=0.5,
                marker_color="blue",
            )
        )
    if non_augmented_coverage.size > 0:
        fig.add_trace(
            go.Histogram(
                x=non_augmented_coverage,
                name="Non-Augmented",
                nbinsx=100,
                opacity=0.5,
                marker_color="orange",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Mean Coverage",
        yaxis_title="Count",
        xaxis=dict(range=[0, 25]),  # Adjusted range for mean coverage
        barmode="overlay",
        bargap=0.1,
    )
    fig.write_html(os.path.join(plots_dir, f"{filename}.html"))
    fig.write_image(os.path.join(plots_dir, f"{filename}.png"))


clinical_dist_params = {
    "low": {
        "mean": 10.0,
        "std": 6.0,
        "log_params": {"mean": 2.08, "std": 0.8},
        "quantiles": {"5%": 1.0, "25%": 4.0, "50%": 8.0, "75%": 12.0, "95%": 20.0},
        "zero_rate": 0.03,
    },
    "med": {
        "mean": 25.0,
        "std": 10.0,
        "log_params": {"mean": 3.0, "std": 0.7},
        "quantiles": {"5%": 5.0, "25%": 12.0, "50%": 20.0, "75%": 30.0, "95%": 50.0},
        "zero_rate": 0.004,
    },
    "high": {
        "mean": 70.0,
        "std": 20.0,
        "log_params": {"mean": 4.2, "std": 0.6},
        "quantiles": {"5%": 30.0, "25%": 50.0, "50%": 67.0, "75%": 85.0, "95%": 120.0},
        "zero_rate": 0.004,
    },
    "clinical": {
        "mean": 5.0,
        "std": 4.0,
        "log_params": {"mean": 1.4, "std": 0.9},
        "quantiles": {"5%": 0.5, "25%": 2.0, "50%": 4.0, "75%": 7.0, "95%": 12.0},
        "zero_rate": 0.2,
    },
}


def plot_distributions(train_pat_dir, eval_pat_dir, atlas_path):
    atlas = pd.read_csv(atlas_path, sep="\t")
    names = set(atlas.name.unique())
    train_dl_low = load_training_with_augmentation(
        f"{train_pat_dir}_low",
        atlas,
        names,
        num_files=5,
        enable_augmentation=True,
        target_dist_params=clinical_dist_params["low"],
        augmentation_probability=0.8,
    )
    train_dl_med = load_training_with_augmentation(
        f"{train_pat_dir}_med",
        atlas,
        names,
        num_files=4,
        enable_augmentation=True,
        target_dist_params=clinical_dist_params["med"],
        augmentation_probability=0.8,
    )
    train_dl_high = load_training_with_augmentation(
        f"{train_pat_dir}_high",
        atlas,
        names,
        num_files=4,
        enable_augmentation=True,
        target_dist_params=clinical_dist_params["high"],
        augmentation_probability=0.8,
    )

    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(
        [train_dl_low.dataset, train_dl_med.dataset, train_dl_high.dataset]
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        persistent_workers=True,
    )
    validation_dls = {}
    for cov in ["high", "med", "low", "clinical"]:
        tier1_dl, t1_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "tier1"),
            atlas,
            names,
            block_size=50_000,  # Block size for Tier1
            target_dist_params=clinical_dist_params[cov],
            enable_augmentation=True,
            target_size=20_000,
        )
        print(f"Validation set {cov} tier1 length={len(t1_yval)}")
        tcells_dl, tcells_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "T-cells"),
            atlas,
            names,
            block_size=10_000,  # Block size matches the 7 types (10,000 each)
            target_dist_params=clinical_dist_params[cov],
            enable_augmentation=True,
            target_size=15_000,
        )
        print(f"Validation set {cov} tcells length={len(tcells_yval)}")
        oac_dl, oac_yval = get_validation_set_with_augmentation(
            str(Path(eval_pat_dir + "_" + cov) / "OAC"),
            atlas,
            names,
            block_size=1_000,  # Block size matches the 12 sets (1,000 each)
            target_dist_params=clinical_dist_params[cov],
            enable_augmentation=True,
            target_size=2_000,
        )
        print(f"Validation set {cov} oac length={len(oac_yval)}")
        validation_dls[f"tier1_{cov}"] = tier1_dl
        validation_dls[f"t-cells_{cov}"] = tcells_dl
        validation_dls[f"oac_{cov}"] = oac_dl
    # 5) Enhance negative examples
    cell_types = list(atlas.columns[8:])
    enhanced_train_dl = enhanced_negative_examples(
        train_dl,
        cell_types,
        atlas,
        sample_fraction=0.01,
        target_dist_params=clinical_dist_params['clinical'],  # Use clinical distribution
        augmentation_probability=0.8,
        enable_augmentation=True
    )
    train_sample_X, train_sample_coverage, train_sample_y = sample_from_dataloader(
        enhanced_train_dl, num_samples=10_000
    )
    plot_coverage_distribution(
        train_sample_coverage,
        "Training Data Coverage Distribution",
        "train_coverage.png",
    )
    plot_marker_value_distribution(
        train_sample_X,
        "Training Data Marker Value Distribution",
        "train_marker_values.png",
    )
    plot_ground_truth_proportions(
        train_sample_y,
        cell_types,
        "Training Data Ground Truth Proportions",
        "train_proportions.png",
    )
    plot_augmented_vs_non_augmented_coverage(
        enhanced_train_dl,
        num_samples=10_000,
        title="Training Data: Augmented vs Non-Augmented Coverage",
        filename="train_augmented_vs_non_augmented_coverage.png",
    )
    for name, dl in validation_dls.items():
        print(f"Plotting validation data distributions for {name}...")
        val_sample_X, val_sample_coverage, val_sample_y = sample_from_dataloader(
            dl, num_samples=5_000
        )
        plot_coverage_distribution(
            val_sample_coverage,
            f"Validation Data ({name}) Coverage Distribution",
            f"val_{name}_coverage.png",
        )
        plot_marker_value_distribution(
            val_sample_X,
            f"Validation Data ({name}) Marker Value Distribution",
            f"val_{name}_marker_values.png",
        )
        plot_ground_truth_proportions(
            val_sample_y,
            cell_types,
            f"Validation Data ({name}) Ground Truth Proportions",
            f"val_{name}_proportions.png",
        )
        print(f"Plotting validation data distributions for {name}...")
        val_sample_X, val_sample_coverage, val_sample_y = sample_from_dataloader(
            dl, num_samples=5_000
        )

        plot_coverage_distribution(
            val_sample_coverage,
            f"Validation Data ({name}) Coverage Distribution",
            f"val_{name}_coverage",
        )
        plot_marker_value_distribution(
            val_sample_X,
            f"Validation Data ({name}) Marker Value Distribution",
            f"val_{name}_marker_values",
        )
        plot_ground_truth_proportions(
            val_sample_y,
            cell_types,
            f"Validation Data ({name}) Ground Truth Proportions",
            f"val_{name}_proportions",
        )
        plot_augmented_vs_non_augmented_coverage(
            dl,
            num_samples=5_000,
            title=f"Validation Data ({name}): Augmented vs Non-Augmented Coverage",
            filename=f"val_{name}_augmented_vs_non_augmented_coverage",
        )


atlas_path = (
    "/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed"
)
train_pat_dir = (
    "/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/train"
)
eval_pat_dir = (
    "/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval"
)
plot_distributions(train_pat_dir, eval_pat_dir, atlas_path)
