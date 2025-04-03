from deep_conv.presence.presence import *

def evaluate_with_standardized_metrics(model, dataloaders, device=None):
    """
    Evaluates model performance with standardized metrics across coverage levels.
    
    Args:
        model: The trained model to evaluate
        dataloaders: Dictionary of dataloaders for validation sets
        device: Device to run evaluation on
    
    Returns:
        Dictionary of standardized metrics
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    results = {}
    for dataset_name, dataloader in dataloaders.items():
        print(f"\nEvaluating {dataset_name}...")
        # Store raw predictions and labels for later analysis
        all_probs = []
        all_preds = []
        all_labels = []
        all_coverages = []
        # Track standard confusion matrix stats
        tp, fp, tn, fn = 0, 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                X = batch['X'].to(device)
                coverage = batch['coverage'].to(device)
                labels = batch['label'].to(device).view(-1, 1)
                # Forward pass
                logits, _ = model(X, coverage)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                # Store for later analysis
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_coverages.append(coverage.mean(dim=1).cpu().numpy())
                # Update confusion matrix
                tp += torch.sum((preds == 1) & (labels == 1)).item()
                fp += torch.sum((preds == 1) & (labels == 0)).item()
                tn += torch.sum((preds == 0) & (labels == 0)).item()
                fn += torch.sum((preds == 0) & (labels == 1)).item()
        # Combine all data
        all_probs = np.concatenate(all_probs).flatten()
        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()
        all_coverages = np.concatenate(all_coverages).flatten()
        # Calculate overall class prevalence
        prevalence = np.mean(all_labels)
        # Calculate standard metrics
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        balanced_acc = (recall + specificity) / 2
        # Calculate PR-AUC
        from sklearn.metrics import precision_recall_curve, auc
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall_curve, precision_curve)
        # Calculate standardized error rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        expected_balanced_error = 0.5 * fpr + 0.5 * fnr
        actual_weighted_error = prevalence * fnr + (1 - prevalence) * fpr
        # Store overall results
        dataset_results = {
            'overall': {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'balanced_acc': balanced_acc,
                'pr_auc': pr_auc,
                'prevalence': prevalence,
                'fpr': fpr,
                'fnr': fnr,
                'expected_balanced_error': expected_balanced_error,
                'actual_weighted_error': actual_weighted_error,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
        }
        # Analyze by coverage level
        coverage_thresholds = [5, 10, 20, 30, 50]
        for threshold in coverage_thresholds:
            high_cov_mask = all_coverages >= threshold
            low_cov_mask = all_coverages < threshold
            for mask_name, mask in [('high_cov', high_cov_mask), ('low_cov', low_cov_mask)]:
                if np.sum(mask) > 0:
                    masked_labels = all_labels[mask]
                    masked_preds = all_preds[mask]
                    masked_probs = all_probs[mask]
                    # Calculate prevalence
                    mask_prevalence = np.mean(masked_labels)
                    # Calculate metrics
                    mask_tp = np.sum((masked_preds == 1) & (masked_labels == 1))
                    mask_fp = np.sum((masked_preds == 1) & (masked_labels == 0))
                    mask_tn = np.sum((masked_preds == 0) & (masked_labels == 0))
                    mask_fn = np.sum((masked_preds == 0) & (masked_labels == 1))
                    if mask_tp + mask_fp > 0:
                        mask_precision = mask_tp / (mask_tp + mask_fp)
                    else:
                        mask_precision = 0
                    if mask_tp + mask_fn > 0:
                        mask_recall = mask_tp / (mask_tp + mask_fn)
                    else:
                        mask_recall = 0
                    if mask_tn + mask_fp > 0:
                        mask_specificity = mask_tn / (mask_tn + mask_fp)
                    else:
                        mask_specificity = 0
                    if mask_precision + mask_recall > 0:
                        mask_f1 = 2 * mask_precision * mask_recall / (mask_precision + mask_recall)
                    else:
                        mask_f1 = 0
                    mask_balanced_acc = (mask_recall + mask_specificity) / 2
                    # Calculate PR-AUC if possible
                    if len(np.unique(masked_labels)) > 1:
                        p_curve, r_curve, _ = precision_recall_curve(masked_labels, masked_probs)
                        mask_pr_auc = auc(r_curve, p_curve)
                    else:
                        mask_pr_auc = 0
                    # Calculate standardized error rates
                    mask_fpr = mask_fp / (mask_fp + mask_tn) if (mask_fp + mask_tn) > 0 else 0
                    mask_fnr = mask_fn / (mask_fn + mask_tp) if (mask_fn + mask_tp) > 0 else 0
                    mask_expected_balanced_error = 0.5 * mask_fpr + 0.5 * mask_fnr
                    mask_actual_weighted_error = mask_prevalence * mask_fnr + (1 - mask_prevalence) * mask_fpr
                    # Store results
                    key = f'{mask_name}_{threshold}'
                    dataset_results[key] = {
                        'precision': mask_precision,
                        'recall': mask_recall,
                        'specificity': mask_specificity,
                        'f1': mask_f1,
                        'balanced_acc': mask_balanced_acc,
                        'pr_auc': mask_pr_auc,
                        'prevalence': mask_prevalence,
                        'fpr': mask_fpr,
                        'fnr': mask_fnr,
                        'expected_balanced_error': mask_expected_balanced_error,
                        'actual_weighted_error': mask_actual_weighted_error,
                        'sample_count': np.sum(mask),
                        'tp': mask_tp,
                        'fp': mask_fp,
                        'tn': mask_tn,
                        'fn': mask_fn
                    }
        # Create matched subsets for fair comparison
        pos_indices = np.where(all_labels == 1)[0]
        neg_indices = np.where(all_labels == 0)[0]
        # Ensure we have enough samples in each group
        min_samples = min(len(pos_indices), len(neg_indices)) // 2
        if min_samples >= 100:
            np.random.seed(42)  # For reproducibility
            # Select random samples
            sampled_pos = np.random.choice(pos_indices, min_samples, replace=False)
            sampled_neg = np.random.choice(neg_indices, min_samples, replace=False)
            # Split by coverage
            pos_coverages = all_coverages[sampled_pos]
            neg_coverages = all_coverages[sampled_neg]
            # Define high/low coverage within each class
            pos_high_cov = sampled_pos[pos_coverages >= 30.0]
            pos_low_cov = sampled_pos[pos_coverages < 30.0]
            neg_high_cov = sampled_neg[neg_coverages >= 30.0]
            neg_low_cov = sampled_neg[neg_coverages < 30.0]
            # Create balanced sets (same number of pos/neg)
            min_class_size = min(len(pos_high_cov), len(neg_high_cov), len(pos_low_cov), len(neg_low_cov))
            if min_class_size >= 50:
                # Sample from each group
                sampled_pos_high = np.random.choice(pos_high_cov, min_class_size, replace=False)
                sampled_neg_high = np.random.choice(neg_high_cov, min_class_size, replace=False)
                sampled_pos_low = np.random.choice(pos_low_cov, min_class_size, replace=False)
                sampled_neg_low = np.random.choice(neg_low_cov, min_class_size, replace=False)
                # Create balanced high coverage set
                high_cov_indices = np.concatenate([sampled_pos_high, sampled_neg_high])
                high_cov_labels = np.concatenate([np.ones(min_class_size), np.zeros(min_class_size)])
                high_cov_preds = all_preds[high_cov_indices]
                high_cov_probs = all_probs[high_cov_indices]
                # Create balanced low coverage set
                low_cov_indices = np.concatenate([sampled_pos_low, sampled_neg_low])
                low_cov_labels = np.concatenate([np.ones(min_class_size), np.zeros(min_class_size)])
                low_cov_preds = all_preds[low_cov_indices]
                low_cov_probs = all_probs[low_cov_indices]
                # Calculate metrics for balanced sets
                for set_name, set_labels, set_preds, set_probs in [
                    ('balanced_high_cov', high_cov_labels, high_cov_preds, high_cov_probs),
                    ('balanced_low_cov', low_cov_labels, low_cov_preds, low_cov_probs)
                ]:
                    # Calculate metrics
                    set_tp = np.sum((set_preds == 1) & (set_labels == 1))
                    set_fp = np.sum((set_preds == 1) & (set_labels == 0))
                    set_tn = np.sum((set_preds == 0) & (set_labels == 0))
                    set_fn = np.sum((set_preds == 0) & (set_labels == 1))
                    if set_tp + set_fp > 0:
                        set_precision = set_tp / (set_tp + set_fp)
                    else:
                        set_precision = 0
                    if set_tp + set_fn > 0:
                        set_recall = set_tp / (set_tp + set_fn)
                    else:
                        set_recall = 0
                    if set_tn + set_fp > 0:
                        set_specificity = set_tn / (set_tn + set_fp)
                    else:
                        set_specificity = 0
                    if set_precision + set_recall > 0:
                        set_f1 = 2 * set_precision * set_recall / (set_precision + set_recall)
                    else:
                        set_f1 = 0
                    set_balanced_acc = (set_recall + set_specificity) / 2
                    # Calculate PR-AUC
                    set_p_curve, set_r_curve, _ = precision_recall_curve(set_labels, set_probs)
                    set_pr_auc = auc(set_r_curve, set_p_curve)
                    # Store results
                    dataset_results[set_name] = {
                        'precision': set_precision,
                        'recall': set_recall,
                        'specificity': set_specificity,
                        'f1': set_f1,
                        'balanced_acc': set_balanced_acc,
                        'pr_auc': set_pr_auc,
                        'prevalence': 0.5,  # By design
                        'sample_count': len(set_labels),
                        'tp': set_tp,
                        'fp': set_fp,
                        'tn': set_tn,
                        'fn': set_fn
                    }
        # Print summary
        print(f"\n=== {dataset_name} Summary ===")
        print(f"Overall: ")
        print(f"  Prevalence: {prevalence:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  Precision: {precision:.4f} (vs {prevalence:.4f} baseline)")
        print(f"  Recall: {recall:.4f}, Specificity: {specificity:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  Expected Error (balanced): {expected_balanced_error:.4f}")
        print(f"  Actual Error (weighted): {actual_weighted_error:.4f}")
        if 'balanced_high_cov' in dataset_results and 'balanced_low_cov' in dataset_results:
            high_bal_acc = dataset_results['balanced_high_cov']['balanced_acc']
            low_bal_acc = dataset_results['balanced_low_cov']['balanced_acc']
            print(f"\nBalanced Sets (50/50 class distribution):")
            print(f"  High Coverage ({dataset_results['balanced_high_cov']['sample_count']} samples): Balanced Acc = {high_bal_acc:.4f}")
            print(f"  Low Coverage ({dataset_results['balanced_low_cov']['sample_count']} samples): Balanced Acc = {low_bal_acc:.4f}")
            print(f"  Difference: {high_bal_acc - low_bal_acc:.4f}")
        results[dataset_name] = dataset_results
    return results

def run_high_coverage_diagnostic(model, dataset_loader, device):
    """
    Diagnostic test to evaluate model performance specifically on high-coverage samples.
    
    Args:
        model: Your trained presence model
        dataset_loader: DataLoader for a validation set
        device: Device to run inference on
    """
    model.eval()
    
    # Track metrics for high coverage samples
    high_cov_tp = 0
    high_cov_fp = 0
    high_cov_tn = 0
    high_cov_fn = 0
    high_cov_count = 0
    
    # Track metrics for all other samples for comparison
    other_tp = 0
    other_fp = 0
    other_tn = 0
    other_fn = 0
    other_count = 0
    
    # Define high coverage threshold
    high_cov_threshold = 30.0  # Adjust this based on your data
    
    with torch.no_grad():
        for batch in dataset_loader:
            X = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            labels = batch['label'].to(device).view(-1, 1)
            
            # Calculate mean coverage for each sample
            mean_coverage = coverage.mean(dim=1)
            
            # Identify high coverage samples
            high_cov_mask = mean_coverage >= high_cov_threshold
            
            # Forward pass
            logits, _ = model(X, coverage)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            # Calculate metrics for high coverage samples
            if high_cov_mask.any():
                high_cov_preds = preds[high_cov_mask]
                high_cov_labels = labels[high_cov_mask]
                
                high_cov_tp += torch.sum((high_cov_preds == 1) & (high_cov_labels == 1)).item()
                high_cov_fp += torch.sum((high_cov_preds == 1) & (high_cov_labels == 0)).item()
                high_cov_tn += torch.sum((high_cov_preds == 0) & (high_cov_labels == 0)).item()
                high_cov_fn += torch.sum((high_cov_preds == 0) & (high_cov_labels == 1)).item()
                high_cov_count += high_cov_mask.sum().item()
            
            # Calculate metrics for other samples
            other_mask = ~high_cov_mask
            if other_mask.any():
                other_preds = preds[other_mask]
                other_labels = labels[other_mask]
                
                other_tp += torch.sum((other_preds == 1) & (other_labels == 1)).item()
                other_fp += torch.sum((other_preds == 1) & (other_labels == 0)).item()
                other_tn += torch.sum((other_preds == 0) & (other_labels == 0)).item()
                other_fn += torch.sum((other_preds == 0) & (other_labels == 1)).item()
                other_count += other_mask.sum().item()
    
    # Calculate metrics for high coverage
    if high_cov_count > 0:
        high_cov_acc = (high_cov_tp + high_cov_tn) / high_cov_count
        high_cov_prec = high_cov_tp / (high_cov_tp + high_cov_fp) if (high_cov_tp + high_cov_fp) > 0 else 0
        high_cov_recall = high_cov_tp / (high_cov_tp + high_cov_fn) if (high_cov_tp + high_cov_fn) > 0 else 0
        high_cov_spec = high_cov_tn / (high_cov_tn + high_cov_fp) if (high_cov_tn + high_cov_fp) > 0 else 0
        high_cov_f1 = 2 * high_cov_prec * high_cov_recall / (high_cov_prec + high_cov_recall) if (high_cov_prec + high_cov_recall) > 0 else 0
        high_cov_balanced_acc = (high_cov_recall + high_cov_spec) / 2
    else:
        high_cov_acc, high_cov_prec, high_cov_recall, high_cov_spec, high_cov_f1, high_cov_balanced_acc = 0, 0, 0, 0, 0, 0
    
    # Calculate metrics for other samples
    if other_count > 0:
        other_acc = (other_tp + other_tn) / other_count
        other_prec = other_tp / (other_tp + other_fp) if (other_tp + other_fp) > 0 else 0
        other_recall = other_tp / (other_tp + other_fn) if (other_tp + other_fn) > 0 else 0
        other_spec = other_tn / (other_tn + other_fp) if (other_tn + other_fp) > 0 else 0
        other_f1 = 2 * other_prec * other_recall / (other_prec + other_recall) if (other_prec + other_recall) > 0 else 0
        other_balanced_acc = (other_recall + other_spec) / 2
    else:
        other_acc, other_prec, other_recall, other_spec, other_f1, other_balanced_acc = 0, 0, 0, 0, 0, 0
    
    # Print diagnostic results
    print(f"\n===== DIAGNOSTIC RESULTS =====")
    print(f"High Coverage Samples (>={high_cov_threshold}) Count: {high_cov_count}")
    print(f"  Balanced Acc: {high_cov_balanced_acc:.4f}")
    print(f"  Precision: {high_cov_prec:.4f}, Recall: {high_cov_recall:.4f}, Specificity: {high_cov_spec:.4f}, F1: {high_cov_f1:.4f}")
    print(f"  Confusion Matrix: TP={high_cov_tp}, FP={high_cov_fp}, TN={high_cov_tn}, FN={high_cov_fn}")
    
    print(f"\nOther Samples Count: {other_count}")
    print(f"  Balanced Acc: {other_balanced_acc:.4f}")
    print(f"  Precision: {other_prec:.4f}, Recall: {other_recall:.4f}, Specificity: {other_spec:.4f}, F1: {other_f1:.4f}")
    print(f"  Confusion Matrix: TP={other_tp}, FP={other_fp}, TN={other_tn}, FN={other_fn}")
    
    # Analyze distribution of positives and negatives
    high_cov_pos_ratio = ((high_cov_tp + high_cov_fn) / high_cov_count) if high_cov_count > 0 else 0
    other_pos_ratio = ((other_tp + other_fn) / other_count) if other_count > 0 else 0
    
    print(f"\nClass Distribution:")
    print(f"  High Coverage: {high_cov_pos_ratio:.4f} positive samples")
    print(f"  Other Samples: {other_pos_ratio:.4f} positive samples")
    
    return {
        'high_cov': {
            'balanced_acc': high_cov_balanced_acc,
            'precision': high_cov_prec,
            'recall': high_cov_recall,
            'specificity': high_cov_spec,
            'f1': high_cov_f1,
            'count': high_cov_count,
            'pos_ratio': high_cov_pos_ratio
        },
        'other': {
            'balanced_acc': other_balanced_acc,
            'precision': other_prec,
            'recall': other_recall,
            'specificity': other_spec,
            'f1': other_f1,
            'count': other_count,
            'pos_ratio': other_pos_ratio
        }
    }

def eval_model(presence_models_dir, target_cell_type_name):
    atlas_dir = "loyfer_atlas"
    atlas_path = f"/users/zetzioni/sharedscratch/{atlas_dir}/atlas/atlas_oac.blood+gi+tum.l4.bed"
    atlas = pd.read_csv(atlas_path, sep="\t")
    cell_types = list(atlas.columns[8:])
    names = set(atlas[atlas.target==target_cell_type_name].name.unique())
    cell_type_idx = cell_types.index(target_cell_type_name)
    model_path = Path(presence_models_dir) / f"presence_model_{cell_type_idx}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Presence model not found at {model_path}")
    
    # Load the model
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        from deep_conv.presence.model import SingleCellTypePresenceModel
        model = SingleCellTypePresenceModel()
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint

    model.eval()
    
    dataloaders = {}
    eval_pat_dir = f"/users/zetzioni/sharedscratch/{atlas_dir}/training/oac.blood+gi+tum.l4/eval"
    for cov in ['high','med','low']:
        tier1_dl, _ = get_validation_set(str(Path(eval_pat_dir+"_"+cov) / "tier1"), cell_type_idx, names)
        dataloaders['tier1_'+cov] = tier1_dl
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    diagnostic_results = run_high_coverage_diagnostic(model, tier1_dl, device)
    print(diagnostic_results)
    standardized_results = evaluate_with_standardized_metrics(model, dataloaders)
    max_balanced_high_cov = 0
    max_balanced_low_cov = 0
    best_dataset = None

    for dataset_name, metrics in standardized_results.items():
        if 'balanced_high_cov' in metrics and 'balanced_low_cov' in metrics:
            high_bal_acc = metrics['balanced_high_cov']['balanced_acc']
            low_bal_acc = metrics['balanced_low_cov']['balanced_acc']
            
            # We want both high and low coverage to perform well
            combined_score = 0.4 * high_bal_acc + 0.6 * low_bal_acc
            
            if combined_score > max_balanced_high_cov + max_balanced_low_cov:
                max_balanced_high_cov = high_bal_acc
                max_balanced_low_cov = low_bal_acc
                best_dataset = dataset_name

    print(f"Best validation set: {best_dataset}")
    print(f"  Balanced High Coverage Accuracy: {max_balanced_high_cov:.4f}")
    print(f"  Balanced Low Coverage Accuracy: {max_balanced_low_cov:.4f}")

