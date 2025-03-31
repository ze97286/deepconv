from deep_conv.deconvolution.deepconv import *
import json 

def analyse_model_performance(model, val_loaders, cell_types, device):    
    diagnostics = {}
    # Run with post-processing disabled first to see raw output
    model.post_processing_enabled = False
    for val_name, val_loader in val_loaders.items():
        print(f"Analysing {val_name}...")
        val_diagnostics = {}
        # Collect predictions, true values, coverage, and quality scores
        all_preds = []
        all_true = []
        all_coverages = []
        all_qualities = []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                y_true = batch['y'].to(device)
                # Get predictions and quality scores
                props, _, _, quality = model(X, coverage)
                # Store everything
                all_preds.append(props.cpu().numpy())
                all_true.append(y_true.cpu().numpy())
                all_coverages.append(coverage.mean(dim=1).cpu().numpy())
                all_qualities.append(quality.cpu().numpy())
        # Combine results
        all_preds = np.vstack(all_preds)
        all_true = np.vstack(all_true)
        all_coverages = np.concatenate(all_coverages)
        all_qualities = np.vstack(all_qualities)
        # 1. Sparsity analysis
        val_diagnostics['true_zeros_percentage'] = np.mean(all_true < 0.001) * 100
        val_diagnostics['pred_zeros_percentage'] = np.mean(all_preds < 0.001) * 100
        # 2. Cell detection analysis
        true_cells_per_sample = np.sum(all_true > 0.001, axis=1)
        pred_cells_per_sample = np.sum(all_preds > 0.001, axis=1)
        val_diagnostics['avg_true_cells_per_sample'] = np.mean(true_cells_per_sample)
        val_diagnostics['avg_pred_cells_per_sample'] = np.mean(pred_cells_per_sample)
        # 3. Coverage analysis
        coverage_bins = [0, 2, 5, 10, 20, float('inf')]
        for i in range(len(coverage_bins)-1):
            low, high = coverage_bins[i], coverage_bins[i+1]
            bin_mask = (all_coverages >= low) & (all_coverages < high)
            if np.any(bin_mask):
                bin_name = f"cov_{low}_to_{high}"
                val_diagnostics[f"{bin_name}_count"] = np.sum(bin_mask)
                val_diagnostics[f"{bin_name}_r2"] = r2_score(
                    all_true[bin_mask].flatten(), 
                    all_preds[bin_mask].flatten()
                )
                val_diagnostics[f"{bin_name}_sparsity_diff"] = (
                    np.mean(all_preds[bin_mask] < 0.001) - 
                    np.mean(all_true[bin_mask] < 0.001)
                )
        # 4. Quality analysis
        val_diagnostics['avg_quality'] = np.mean(all_qualities)
        # Calculate error vs quality correlation
        errors = np.abs(all_preds - all_true)
        error_quality_corr = []
        for c in range(all_qualities.shape[1]):
            # Correlation between error and quality for each cell type
            corr = np.corrcoef(errors[:, c], all_qualities[:, c])[0, 1]
            error_quality_corr.append(corr)
        val_diagnostics['error_quality_correlation'] = np.mean(error_quality_corr)
        # 5. Per-cell-type analysis
        for c, cell_type in enumerate(cell_types):
            # Get cell-specific metrics
            true_c = all_true[:, c]
            pred_c = all_preds[:, c]
            quality_c = all_qualities[:, c]
            # Calculate metrics
            val_diagnostics[f"{cell_type}_r2"] = r2_score(true_c, pred_c)
            val_diagnostics[f"{cell_type}_precision"] = precision_score(
                true_c > 0.001, 
                pred_c > 0.001, 
                zero_division=0
            )
            val_diagnostics[f"{cell_type}_recall"] = recall_score(
                true_c > 0.001, 
                pred_c > 0.001, 
                zero_division=0
            )
            val_diagnostics[f"{cell_type}_false_positive_rate"] = np.sum(
                (pred_c > 0.001) & (true_c <= 0.001)
            ) / max(1, np.sum(true_c <= 0.001))
            val_diagnostics[f"{cell_type}_avg_quality"] = np.mean(quality_c)
        # 6. Test different thresholds
        for threshold in [0.001, 0.002, 0.005, 0.01, 0.02]:
            proc_preds = all_preds.copy()
            proc_preds[proc_preds < threshold] = 0
            # Renormalize
            row_sums = np.sum(proc_preds, axis=1, keepdims=True)
            valid_rows = (row_sums > 0).flatten()
            if np.any(valid_rows):
                proc_preds[valid_rows] /= row_sums[valid_rows]
            val_diagnostics[f"threshold_{threshold}_r2"] = r2_score(
                all_true.flatten(), 
                proc_preds.flatten()
            )
            val_diagnostics[f"threshold_{threshold}_false_positive_rate"] = np.mean(
                (proc_preds > 0) & (all_true <= 0.001)
            )
            val_diagnostics[f"threshold_{threshold}_false_negative_rate"] = np.mean(
                (proc_preds == 0) & (all_true > 0.001)
            )
        diagnostics[val_name] = val_diagnostics
    return diagnostics


clinical_dist_params = {
        'mean': 5.0,
        'std': 4.0,
        'log_params': {
            'mean': 1.2,
            'std': 0.8
        },
        'quantiles': {
            '5%': 0.5,
            '25%': 2.0,
            '50%': 4.0,
            '75%': 7.0,
            '95%': 12.0
        },
        'zero_rate': 0.1
    }
atlas_path = "/users/zetzioni/sharedscratch/loyfer_atlas/atlas/atlas_oac.blood+gi+tum.l4.bed"
atlas = pd.read_csv(atlas_path, sep="\t")
names = set(atlas.name.unique())
eval_pat_dir = "/users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval"
validation_dls = {}
clinical_validation_dls = {}  # New clinical-like validation set
y_vals = {}
for cov in ['high','med','low']:
    # For each validation set, get both standard and clinical variants
    tier1_dl, tier1_clinical_dl, t1_yval = get_validation_set_with_augmentation(
        str(Path(eval_pat_dir+"_"+cov) / "tier1"), 
        atlas, names, 
        target_dist_params=clinical_dist_params,
        block_size=100_000,
    )
    print(f"validation set for {cov} tier1 length={len(t1_yval)}")
    tcells_dl, tcells_clinical_dl, tcells_yval = get_validation_set_with_augmentation(
        str(Path(eval_pat_dir+"_"+cov) / "T-cells"), 
        atlas, names,
        target_dist_params=clinical_dist_params,
        block_size=10_000,
    )
    print(f"validation set for {cov} tcells length={len(tcells_yval)}")
    oac_dl, oac_clinical_dl, oac_yval = get_validation_set_with_augmentation(
        str(Path(eval_pat_dir+"_"+cov) / "OAC"), 
        atlas, names,
        target_dist_params=clinical_dist_params,
        block_size=1_000,
    )
    print(f"validation set for {cov} oac length={len(oac_yval)}")
    # Store both standard and clinical variants
    validation_dls[f"tier1_{cov}"] = tier1_dl
    validation_dls[f"t-cells_{cov}"] = tcells_dl
    validation_dls[f"oac_{cov}"] = oac_dl
    clinical_validation_dls[f"tier1_{cov}_clinical"] = tier1_clinical_dl
    clinical_validation_dls[f"t-cells_{cov}_clinical"] = tcells_clinical_dl
    clinical_validation_dls[f"oac_{cov}_clinical"] = oac_clinical_dl
    y_vals[f"tier1_{cov}"] = t1_yval
    y_vals[f"t-cells_{cov}"] = tcells_yval
    y_vals[f"oac_{cov}"] = oac_yval
    # Use the same ground truth for clinical variants
    y_vals[f"tier1_{cov}_clinical"] = t1_yval
    y_vals[f"t-cells_{cov}_clinical"] = tcells_yval
    y_vals[f"oac_{cov}_clinical"] = oac_yval

cell_types = list(atlas.columns[8:])
target_ids = atlas["target"].map(lambda x: cell_types.index(x)).to_numpy()   
model = CellTypeDeconvolutionModel(
    num_markers=len(atlas),
    num_cell_types=len(cell_types),
    target_ids=target_ids,
    cell_types=cell_types,
)
checkpoint = torch.load(f"/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/deepconv/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
device = next(model.parameters()).device
print("\nRunning comprehensive diagnostics...")
diagnostics = analyse_model_performance(
    model, 
    {**validation_dls, **clinical_validation_dls}, 
    cell_types, 
    device
)

# Print key findings
print("\nKey Diagnostic Findings:")
for val_name, metrics in diagnostics.items():
    print(f"\n--- {val_name} ---")
    print(f"True zeros: {metrics['true_zeros_percentage']:.1f}% | Predicted zeros: {metrics['pred_zeros_percentage']:.1f}%")
    print(f"Avg true cells per sample: {metrics['avg_true_cells_per_sample']:.2f} | Predicted: {metrics['avg_pred_cells_per_sample']:.2f}")
    
    # Top 3 problematic cell types (highest false positive rate)
    cell_fps = [(cell, metrics[f"{cell}_false_positive_rate"]) 
                for cell in cell_types]
    cell_fps.sort(key=lambda x: x[1], reverse=True)
    print("Top 3 problematic cell types (false positives):")
    for cell, fp_rate in cell_fps[:3]:
        print(f"  {cell}: {fp_rate:.1%}")
    
    # Best threshold
    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02]
    best_r2 = -np.inf
    best_threshold = None
    for t in thresholds:
        r2 = metrics[f"threshold_{t}_r2"]
        if r2 > best_r2:
            best_r2 = r2
            best_threshold = t
    print(f"Best threshold: {best_threshold} (R² = {best_r2:.3f})")
    
    # Coverage analysis
    print("Performance by coverage:")
    for cov in ["cov_0_to_2", "cov_2_to_5", "cov_5_to_10", "cov_10_to_20"]:
        if f"{cov}_count" in metrics:
            count = metrics[f"{cov}_count"]
            r2 = metrics[f"{cov}_r2"]
            print(f"  {cov}: count={count}, R²={r2:.3f}")

# Save diagnostics for later analysis
import json
output_path = "/users/zetzioni/sharedscratch/loyfer_atlas/saved_models/deepconv/"
with open(f"{output_path}/model_diagnostics.json", 'w') as f:
    # Convert numpy values to python native types
    serializable_diagnostics = {}
    for val_name, metrics in diagnostics.items():
        serializable_diagnostics[val_name] = {
            k: float(v) if isinstance(v, (np.float32, np.float64)) else v
            for k, v in metrics.items()
        }
    json.dump(serializable_diagnostics, f, indent=2)
