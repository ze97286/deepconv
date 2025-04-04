import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

def train_binary_classifier(
    model: nn.Module,
    dataloaders: dict,
    model_path: str,
    target_cell_type_index: int,
    num_epochs: int = 50,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-5,
    class_weight: float = None,  # Positive class weight (for imbalance)
    patience: int = 10,
    device: torch.device = None,
    fp16_training: bool = True,  # Use mixed precision
    gradient_accumulation: int = 1,  # Number of batches to accumulate
    eval_metric: str = 'balanced_accuracy',  # 'balanced_accuracy', 'f1', 'auroc'
    coverage_low_threshold: float = 6.0,  # Threshold for low coverage
    coverage_med_threshold: float = 12.0   # Threshold for medium coverage
):
    """
    Train a binary classifier for cell type detection with coverage-aware loss.
    
    Args:
        model: Binary classifier model
        dataloaders: Dictionary containing 'train' and 'val' dataloaders
        model_path: Path to save model checkpoints
        target_cell_type_index: Index of the target cell type
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        class_weight: Weight for positive class (None = auto-calculate)
        patience: Early stopping patience
        device: Training device (GPU/CPU)
        fp16_training: Whether to use mixed precision training
        gradient_accumulation: Number of batches to accumulate gradients
        eval_metric: Metric to use for model selection
        coverage_low_threshold: Threshold for defining low coverage
        coverage_med_threshold: Threshold for defining medium coverage
    
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Set up mixed precision training
    scaler = torch.cuda.amp.GradScaler() if fp16_training and torch.cuda.is_available() else None
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=patience//2, verbose=True
    # )
    steps_per_epoch = len(dataloaders['train'])

    # One complete cycle is 2 * step_size_up steps
    desired_cycles = 2.5
    total_steps = steps_per_epoch * 10  # 10 epochs
    step_size_up = int(total_steps / (2 * desired_cycles))
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
    cyclic_scheduler = CyclicLR(
        optimizer,
        base_lr=1e-4,
        max_lr=1e-3,
        step_size_up=step_size_up,  
        cycle_momentum=False
    )

    print(f"Cyclic LR config: steps_per_epoch={steps_per_epoch}, step_size_up={step_size_up}, "
      f"cycles in first 10 epochs={10*steps_per_epoch/(2*step_size_up):.2f}")

    # Create ReduceLROnPlateau for later epochs
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=patience//2, 
        verbose=True
    )

    
    # Create directory for saving models
    os.makedirs(model_path, exist_ok=True)
    
    # Automatically calculate class weight if not provided
    if class_weight is None:
        pos_count = 0
        total_count = 0
        
        for batch in dataloaders['train']:
            labels = batch['label']
            pos_count += torch.sum(labels).item()
            total_count += len(labels)
        
        pos_ratio = pos_count / total_count
        class_weight = (1 - pos_ratio) / pos_ratio
        print(f"Calculated positive class weight: {class_weight:.4f} (ratio: {pos_ratio:.4f})")
    
    # Create loss function with class weights and coverage awareness
    weights = torch.tensor([1.0, class_weight], device=device)
    
    def weighted_bce_loss(logits, targets, coverage, missing_rate=None):
        """
        Enhanced coverage-aware weighted BCE loss with:
        1. Better handling of missing markers
        2. Stronger regularization for very low coverage 
        3. Calibration penalty for low confidence regions
        
        Args:
            logits: [B, 1] Classification logits
            targets: [B, 1] Binary targets
            coverage: [B, M] Coverage values
            missing_rate: [B, 1] Missing marker rate (optional, will be calculated if not provided)
        """
        # Calculate mean coverage for each sample
        sample_coverage = coverage.mean(dim=1, keepdim=True)
        
        # Calculate missing rate if not provided
        if missing_rate is None:
            missing_rate = (coverage == 0).float().mean(dim=1, keepdim=True)
        
        # Calculate coverage weights (higher weight for lower coverage)
        # Enhanced to be more sensitive to very low coverage
        coverage_weights = torch.where(
            sample_coverage < 5.0,
            torch.clamp(1.5 + (15.0 / (sample_coverage + 3.0)), 1.0, 3.0),  # Higher weight for very low coverage
            torch.clamp(1.0 + (10.0 / (sample_coverage + 5.0)), 0.8, 2.0)   # Original scaling
        )
        
        # Class weights based on positive/negative imbalance
        per_sample_weights = torch.ones_like(targets)
        per_sample_weights[targets == 1] = weights[1]  # Assuming 'weights' is defined outside
        
        # Add concentration-based weighting for more balanced focus
        target_conc = targets.view(-1)
        conc_weights = torch.ones_like(target_conc)
        
        # Only apply concentration weights to positive samples (where conc > 0)
        pos_samples = (target_conc > 0)
        if torch.any(pos_samples):
            # Extract positive sample concentrations
            pos_conc = target_conc[pos_samples]
            
            # Initialize weights for different concentration ranges
            # Higher weights for very low and very high concentrations
            very_low_conc = (pos_conc > 0) & (pos_conc < 0.01)
            low_conc = (pos_conc >= 0.01) & (pos_conc < 0.05)
            med_conc = (pos_conc >= 0.05) & (pos_conc < 0.2)
            high_conc = (pos_conc >= 0.2) & (pos_conc < 0.5)
            very_high_conc = pos_conc >= 0.5
            
            # Assign weights to each concentration range
            # Enhanced weights for extreme cases
            pos_weights = torch.ones_like(pos_conc)
            pos_weights[very_low_conc] = 1.5  # Increased from 1.3
            pos_weights[low_conc] = 1.2      # Increased from 1.1
            pos_weights[med_conc] = 1.0
            pos_weights[high_conc] = 1.2
            pos_weights[very_high_conc] = 1.5
            
            # Update weights for positive samples
            conc_weights[pos_samples] = pos_weights
        
        # Apply coverage-dependent bias adjustment with missing marker awareness
        # This makes the model more conservative at low coverage and high missing rates
        coverage_bias = 0.2 * torch.clamp((sample_coverage - 20.0) / 30.0, -1.0, 1.0)
        missing_bias = -0.1 * torch.clamp(missing_rate * 2.0, 0.0, 1.0)  # Bias toward negative for many missing markers
        combined_bias = coverage_bias + missing_bias
        adjusted_logits = logits + combined_bias
        
        # Combine all weights: class balance × coverage × concentration
        # Add missing rate factor to give higher weight to samples with fewer missing markers
        missing_factor = torch.clamp(1.0 - missing_rate * 0.5, 0.5, 1.0)  # Reduce weight for high missing rate
        combined_weights = per_sample_weights * coverage_weights * conc_weights.view(-1, 1) * missing_factor
        
        # Calculate weighted loss with the adjusted logits
        bce_loss = F.binary_cross_entropy_with_logits(
            adjusted_logits, targets, weight=combined_weights, reduction='mean'
        )
        
        # Enhanced regularization for very low coverage and high missing rate
        very_low_cov_mask = (sample_coverage < 5.0).squeeze(-1)
        high_missing_mask = (missing_rate > 0.3).squeeze(-1)
        challenging_mask = very_low_cov_mask | high_missing_mask
        
        # Add focal loss component for challenging samples
        if torch.any(challenging_mask):
            # Get logits for challenging samples
            challenging_logits = logits[challenging_mask]
            challenging_targets = targets[challenging_mask]
            
            # Calculate probabilities
            probs = torch.sigmoid(challenging_logits)
            
            # Focal loss component (focus on hard examples)
            gamma = 2.0
            pt = torch.where(challenging_targets == 1, probs, 1 - probs)
            focal_loss = -((1 - pt) ** gamma) * torch.log(pt + 1e-7)
            
            # Penalize high confidence for challenging samples
            confidence_penalty = torch.abs(probs - 0.5).mean()
            
            # Scale regularization based on how challenging the samples are
            challenge_factor = missing_rate[challenging_mask].mean() + (5.0 / (sample_coverage[challenging_mask].mean() + 1e-5))
            reg_weight = torch.clamp(0.2 * challenge_factor, 0.2, 0.5)
            
            # Add to the loss
            total_loss = bce_loss + reg_weight * confidence_penalty + 0.1 * focal_loss.mean()
            return total_loss
        else:
            return bce_loss
        
    # Tracking variables
    best_metric = 0.0
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        # Add this at the start of each epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Learning rate: {current_lr:.6f}")
        
        # Tracking metrics by coverage level
        coverage_categories = ['all', 'low', 'medium', 'high']
        train_metrics = {cat: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'count': 0, 'loss': 0.0} 
                        for cat in coverage_categories}
        
        # Training
        for batch_idx, batch in enumerate(tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{num_epochs}")):
            marker_values = batch['X'].to(device)
            coverage = batch['coverage'].to(device)
            labels = batch['label'].to(device).view(-1, 1)
            
            # Calculate mean coverage for each sample for stratification
            mean_coverage = coverage.mean(dim=1)
            low_cov_mask = (mean_coverage < coverage_low_threshold)
            med_cov_mask = (mean_coverage >= coverage_low_threshold) & (mean_coverage < coverage_med_threshold)
            high_cov_mask = (mean_coverage >= coverage_med_threshold)
            
            # Forward pass with mixed precision if enabled
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Get logits and missing_rate from the model
                    logits, _, missing_rate = model(marker_values, coverage)
                    # Pass missing_rate to the loss function
                    loss = weighted_bce_loss(logits, labels, coverage, missing_rate)
                    loss = loss / gradient_accumulation  # Scale for gradient accumulation
            else:
                # Get logits and missing_rate from the model
                logits, _, missing_rate = model(marker_values, coverage)
                # Pass missing_rate to the loss function
                loss = weighted_bce_loss(logits, labels, coverage, missing_rate)
                loss = loss / gradient_accumulation  # Scale for gradient accumulation
            
            # Backward pass with mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation == 0 or batch_idx == len(dataloaders['train']) - 1:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation == 0 or batch_idx == len(dataloaders['train']) - 1:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Track metrics
            with torch.no_grad():
                # Record loss
                batch_loss = loss.item() * gradient_accumulation
                train_losses.append(batch_loss)
                
                # Calculate predictions
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).float()
                
                # Update metrics for all samples
                train_metrics['all']['tp'] += torch.sum((predictions == 1) & (labels == 1)).item()
                train_metrics['all']['fp'] += torch.sum((predictions == 1) & (labels == 0)).item()
                train_metrics['all']['tn'] += torch.sum((predictions == 0) & (labels == 0)).item()
                train_metrics['all']['fn'] += torch.sum((predictions == 0) & (labels == 1)).item()
                train_metrics['all']['count'] += len(labels)
                train_metrics['all']['loss'] += batch_loss * len(labels)
                
                # Update metrics for low coverage samples if present
                if low_cov_mask.any():
                    low_predictions = predictions[low_cov_mask]
                    low_labels = labels[low_cov_mask]
                    train_metrics['low']['tp'] += torch.sum((low_predictions == 1) & (low_labels == 1)).item()
                    train_metrics['low']['fp'] += torch.sum((low_predictions == 1) & (low_labels == 0)).item()
                    train_metrics['low']['tn'] += torch.sum((low_predictions == 0) & (low_labels == 0)).item()
                    train_metrics['low']['fn'] += torch.sum((low_predictions == 0) & (low_labels == 1)).item()
                    train_metrics['low']['count'] += len(low_labels)
                    train_metrics['low']['loss'] += batch_loss * len(low_labels)
                
                # Update metrics for medium coverage samples if present
                if med_cov_mask.any():
                    med_predictions = predictions[med_cov_mask]
                    med_labels = labels[med_cov_mask]
                    train_metrics['medium']['tp'] += torch.sum((med_predictions == 1) & (med_labels == 1)).item()
                    train_metrics['medium']['fp'] += torch.sum((med_predictions == 1) & (med_labels == 0)).item()
                    train_metrics['medium']['tn'] += torch.sum((med_predictions == 0) & (med_labels == 0)).item()
                    train_metrics['medium']['fn'] += torch.sum((med_predictions == 0) & (med_labels == 1)).item()
                    train_metrics['medium']['count'] += len(med_labels)
                    train_metrics['medium']['loss'] += batch_loss * len(med_labels)
                
                # Update metrics for high coverage samples if present
                if high_cov_mask.any():
                    high_predictions = predictions[high_cov_mask]
                    high_labels = labels[high_cov_mask]
                    train_metrics['high']['tp'] += torch.sum((high_predictions == 1) & (high_labels == 1)).item()
                    train_metrics['high']['fp'] += torch.sum((high_predictions == 1) & (high_labels == 0)).item()
                    train_metrics['high']['tn'] += torch.sum((high_predictions == 0) & (high_labels == 0)).item()
                    train_metrics['high']['fn'] += torch.sum((high_predictions == 0) & (high_labels == 1)).item()
                    train_metrics['high']['count'] += len(high_labels)
                    train_metrics['high']['loss'] += batch_loss * len(high_labels)
        
        # Calculate final training metrics for each coverage level
        for cat in coverage_categories:
            metrics = train_metrics[cat]
            if metrics['count'] > 0:
                metrics['loss'] = metrics['loss'] / metrics['count']
                
                tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                balanced_accuracy = (recall + specificity) / 2

                pos_weight = torch.sum(labels == 1).item() / len(labels) if len(labels) > 0 else 0.5
                neg_weight = 1.0 - pos_weight

                # Calculate weighted precision and recall
                weighted_precision = (precision * pos_weight) / (pos_weight + (1 - precision) * neg_weight)
                weighted_recall = (recall * pos_weight) / (pos_weight + (1 - recall) * neg_weight)

                # Calculate weighted F1
                weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0.0

                # Store in metrics
                metrics['weighted_f1'] = weighted_f1
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['specificity'] = specificity
                metrics['f1'] = f1
                metrics['balanced_accuracy'] = balanced_accuracy
                
                print(f"Epoch {epoch+1}/{num_epochs} - Train ({cat} coverage): loss={metrics['loss']:.4f}, " +
                    f"precision={precision:.4f}, recall={recall:.4f}, specificity={specificity:.4f}, " +
                    f"weighted_f1={weighted_f1:.4f}, " +
                    f"f1={f1:.4f}, balanced_acc={balanced_accuracy:.4f}, count={metrics['count']}"
                )
        
        # Validation
        model.eval()
        val_metrics = {}
        
        for val_name, val_loader in dataloaders['val'].items():
            # Setup metrics for each coverage category
            val_set_metrics = {cat: {
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'count': 0, 'loss': 0.0,
                'all_labels': [], 'all_probs': []
            } for cat in coverage_categories}
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating {val_name}"):
                    marker_values = batch['X'].to(device)
                    coverage = batch['coverage'].to(device)
                    labels = batch['label'].to(device).view(-1, 1)
                    
                    # Calculate mean coverage for each sample for stratification
                    mean_coverage = coverage.mean(dim=1)
                    low_cov_mask = (mean_coverage < coverage_low_threshold)
                    med_cov_mask = (mean_coverage >= coverage_low_threshold) & (mean_coverage < coverage_med_threshold)
                    high_cov_mask = (mean_coverage >= coverage_med_threshold)
                    
                    # Forward pass
                    logits, _, missing_rate = model(marker_values, coverage)
                    loss = weighted_bce_loss(logits, labels, coverage, missing_rate)
                    
                    # Calculate metrics
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities >= 0.5).float()
                    
                    # Store predictions and labels for all samples
                    val_set_metrics['all']['all_labels'].append(labels.cpu().numpy())
                    val_set_metrics['all']['all_probs'].append(probabilities.cpu().numpy())
                    val_set_metrics['all']['count'] += len(labels)
                    val_set_metrics['all']['loss'] += loss.item() * len(labels)
                    
                    # Update confusion matrix for all samples
                    val_set_metrics['all']['tp'] += torch.sum((predictions == 1) & (labels == 1)).item()
                    val_set_metrics['all']['fp'] += torch.sum((predictions == 1) & (labels == 0)).item()
                    val_set_metrics['all']['tn'] += torch.sum((predictions == 0) & (labels == 0)).item()
                    val_set_metrics['all']['fn'] += torch.sum((predictions == 0) & (labels == 1)).item()
                    
                    # Update metrics for low coverage samples if present
                    if low_cov_mask.any():
                        low_predictions = predictions[low_cov_mask]
                        low_labels = labels[low_cov_mask]
                        low_probs = probabilities[low_cov_mask]
                        
                        val_set_metrics['low']['all_labels'].append(low_labels.cpu().numpy())
                        val_set_metrics['low']['all_probs'].append(low_probs.cpu().numpy())
                        val_set_metrics['low']['count'] += len(low_labels)
                        val_set_metrics['low']['loss'] += loss.item() * len(low_labels)
                        
                        val_set_metrics['low']['tp'] += torch.sum((low_predictions == 1) & (low_labels == 1)).item()
                        val_set_metrics['low']['fp'] += torch.sum((low_predictions == 1) & (low_labels == 0)).item()
                        val_set_metrics['low']['tn'] += torch.sum((low_predictions == 0) & (low_labels == 0)).item()
                        val_set_metrics['low']['fn'] += torch.sum((low_predictions == 0) & (low_labels == 1)).item()
                    
                    # Update metrics for medium coverage samples if present
                    if med_cov_mask.any():
                        med_predictions = predictions[med_cov_mask]
                        med_labels = labels[med_cov_mask]
                        med_probs = probabilities[med_cov_mask]
                        
                        val_set_metrics['medium']['all_labels'].append(med_labels.cpu().numpy())
                        val_set_metrics['medium']['all_probs'].append(med_probs.cpu().numpy())
                        val_set_metrics['medium']['count'] += len(med_labels)
                        val_set_metrics['medium']['loss'] += loss.item() * len(med_labels)
                        
                        val_set_metrics['medium']['tp'] += torch.sum((med_predictions == 1) & (med_labels == 1)).item()
                        val_set_metrics['medium']['fp'] += torch.sum((med_predictions == 1) & (med_labels == 0)).item()
                        val_set_metrics['medium']['tn'] += torch.sum((med_predictions == 0) & (med_labels == 0)).item()
                        val_set_metrics['medium']['fn'] += torch.sum((med_predictions == 0) & (med_labels == 1)).item()
                    
                    # Update metrics for high coverage samples if present
                    if high_cov_mask.any():
                        high_predictions = predictions[high_cov_mask]
                        high_labels = labels[high_cov_mask]
                        high_probs = probabilities[high_cov_mask]
                        
                        val_set_metrics['high']['all_labels'].append(high_labels.cpu().numpy())
                        val_set_metrics['high']['all_probs'].append(high_probs.cpu().numpy())
                        val_set_metrics['high']['count'] += len(high_labels)
                        val_set_metrics['high']['loss'] += loss.item() * len(high_labels)
                        
                        val_set_metrics['high']['tp'] += torch.sum((high_predictions == 1) & (high_labels == 1)).item()
                        val_set_metrics['high']['fp'] += torch.sum((high_predictions == 1) & (high_labels == 0)).item()
                        val_set_metrics['high']['tn'] += torch.sum((high_predictions == 0) & (high_labels == 0)).item()
                        val_set_metrics['high']['fn'] += torch.sum((high_predictions == 0) & (high_labels == 1)).item()
            
            # Calculate validation metrics for each coverage level
            for cat in coverage_categories:
                metrics = val_set_metrics[cat]
                
                if metrics['count'] > 0:
                    metrics['loss'] = metrics['loss'] / metrics['count']
                    
                    tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    balanced_accuracy = (recall + specificity) / 2
                    pos_weight = torch.sum(labels == 1).item() / len(labels) if len(labels) > 0 else 0.5
                    neg_weight = 1.0 - pos_weight
                    weighted_precision = (precision * pos_weight) / (pos_weight + (1 - precision) * neg_weight)
                    weighted_recall = (recall * pos_weight) / (pos_weight + (1 - recall) * neg_weight)
                    weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0.0

                    
                    # Concat all labels and probabilities if available
                    if len(metrics['all_labels']) > 0 and len(metrics['all_probs']) > 0:
                        try:
                            all_labels = np.concatenate(metrics['all_labels']).flatten()
                            all_probs = np.concatenate(metrics['all_probs']).flatten()
                            
                            # Calculate AUROC and AUPRC (if there are positive and negative examples)
                            if len(np.unique(all_labels)) > 1:
                                auroc = roc_auc_score(all_labels, all_probs)
                                auprc = average_precision_score(all_labels, all_probs)
                            else:
                                auroc = 0.0
                                auprc = precision  # If only one class, AUPRC = precision
                                
                            metrics['auroc'] = auroc
                            metrics['auprc'] = auprc
                        except:
                            # Handle edge cases where concatenation fails
                            metrics['auroc'] = 0.0
                            metrics['auprc'] = 0.0
                    else:
                        metrics['auroc'] = 0.0
                        metrics['auprc'] = 0.0
                    
                    # Store metrics
                    metrics.update({
                        'precision': precision,
                        'recall': recall,
                        'specificity': specificity,
                        'f1': f1,
                        'balanced_accuracy': balanced_accuracy,
                        'weighted_f1': weighted_f1,
                    })
                    
                    # Print validation metrics
                    print(f"Validation ({val_name}, {cat} coverage): loss={metrics['loss']:.4f}, " +
                        f"precision={precision:.4f}, recall={recall:.4f}, specificity={specificity:.4f}, " +
                        f"f1={f1:.4f}, balanced_acc={balanced_accuracy:.4f}, " +
                        f"weighted_f1={weighted_f1:.4f}, "
                        f"AUROC={metrics.get('auroc', 0.0):.4f}, AUPRC={metrics.get('auprc', 0.0):.4f}, count={metrics['count']}")
                    
                    # Print confusion matrix
                    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            
            val_metrics[val_name] = val_set_metrics
        
        # Calculate average metric for validation sets, with emphasis on low coverage performance
        avg_metric_all = np.mean([m['all']['balanced_accuracy'] for m in val_metrics.values() if m['all']['count'] > 0])
        avg_metric_low = np.mean([m['low']['balanced_accuracy'] for m in val_metrics.values() if m['low']['count'] > 0])
        
        # Weight the overall metric to emphasize low coverage performance
        # 60% weight on low coverage, 40% weight on overall
        if np.isnan(avg_metric_low):
            avg_metric = avg_metric_all
        else:
            avg_metric = 0.4 * avg_metric_all + 0.6 * avg_metric_low
        
        print(f"Combined validation metric: {avg_metric:.4f} (All: {avg_metric_all:.4f}, Low: {avg_metric_low:.4f})")
        
        # Update learning rate scheduler
        if epoch < 10:
            cyclic_scheduler.step()
        else:
            plateau_scheduler.step(avg_metric)
        
        # Check for improvement
        if avg_metric > best_metric:
            best_metric = avg_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'metric_name': eval_metric,
                'low_coverage_metric': avg_metric_low,
                'all_coverage_metric': avg_metric_all
            }
            torch.save(checkpoint, os.path.join(model_path, f"presence_model_{target_cell_type_index}.pt"))
            
            print(f"New best model saved! Combined metric={best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_path, f"presence_model_{target_cell_type_index}.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with metric={checkpoint['best_metric']:.4f}")
    print(f"Low coverage: {checkpoint.get('low_coverage_metric', 'N/A')}, All coverage: {checkpoint.get('all_coverage_metric', 'N/A')}")
    
    return model
