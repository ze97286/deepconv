import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, precision_recall_curve, auc
torch.multiprocessing.set_sharing_strategy('file_system')

from deep_conv.detect.preprocess import prepare_data_for_evaluation
from deep_conv.detect.model import EnhancedCancerDetectionModel, CancerDetectionEnsemble


def setup_logging(output_dir=None):
    """Set up logging configuration"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('cancer_detection')
    logger.setLevel(logging.INFO)
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(console_handler)
    
    # Add file handler if output_dir is provided
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'evaluation.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_model(model_dir, device='cpu'):
    """
    Load a trained model from a directory
    
    Args:
        model_dir: Directory containing the model checkpoint
        device: Device to load the model onto ('cuda' or 'cpu')
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    logger = logging.getLogger('cancer_detection')
    logger.info(f"Loading model from {model_dir}")

    # Try to load the best model first
    model_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        # Fall back to final model if best model doesn't exist
        model_path = os.path.join(model_dir, 'final_model.pt')
        if not os.path.exists(model_path):
            # Check for ensemble model
            model_path = os.path.join(model_dir, 'ensemble_model.pt')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model checkpoint found in {model_dir}")

    logger.info(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Load args if available
    args_path = os.path.join(model_dir, 'args.json')
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            args = json.load(f)
    else:
        # Try to extract from checkpoint
        args = checkpoint.get('args', {})
        if not args:
            logger.warning(f"No args.json found in {model_dir}. Using default parameters.")
            args = {}
    
    # Check if it's an ensemble model
    if 'model_states' in checkpoint:
        logger.info("Detected ensemble model")
        # Extract model config
        config = checkpoint.get('model_config', {})
        
        # Create individual models
        models = []
        for model_state in checkpoint['model_states']:
            model = EnhancedCancerDetectionModel(
                num_markers=config.get('num_markers', 1000),
                feature_dim=config.get('feature_dim', 128),
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 3),
                dropout_rate=config.get('dropout_rate', 0.2),
                focal_weight_factor=config.get('focal_weight_factor', 100),
                low_concentration_threshold=config.get('low_concentration_threshold', 0.01),
                detection_thresholds=config.get('detection_thresholds', [0.001, 0.01, 0.05])
            )
            model.load_state_dict(model_state)
            models.append(model)
        
        # Create ensemble
        model = CancerDetectionEnsemble(models)
        
    else:
        # Standard single model
        # Get model parameters
        model_state = checkpoint.get('model', None)
        if model_state is None:
            # Some checkpoints store the model state directly
            model_state = checkpoint
        
        # Get num_markers from the first layer weights if not in args
        if 'num_markers' not in args and isinstance(model_state, dict):
            # Try to infer from marker_embedding.weight
            marker_weights = model_state.get('marker_embedding.weight', None)
            if marker_weights is not None:
                args['num_markers'] = marker_weights.shape[0]
            
        # Create and load model
        model = EnhancedCancerDetectionModel(
            num_markers=args.get('num_markers', 1000),
            feature_dim=args.get('feature_dim', 128),
            num_heads=args.get('num_heads', 8),
            num_layers=args.get('num_layers', 3),
            dropout_rate=args.get('dropout_rate', 0.2),
            detection_thresholds=args.get('detection_thresholds', [0.001, 0.01, 0.05])
        )
        
        # Load model weights
        model.load_state_dict(model_state)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    return model, args


def evaluate_model(model, data_loader, output_dir=None, thresholds=None, device='cuda'):
    """
    Evaluate a trained model on a dataset with enhanced regression metrics
    
    Args:
        model: Trained model
        data_loader: DataLoader with evaluation data
        output_dir: Directory to save results
        thresholds: Detection thresholds
        device: Device to run evaluation on
        
    Returns:
        results: Dictionary of evaluation results
    """
    logger = logging.getLogger('cancer_detection')
    logger.info("Starting model evaluation with enhanced metrics...")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    if thresholds is None:
        # Get thresholds from model if available
        if hasattr(model, 'detection_thresholds'):
            thresholds = model.detection_thresholds
        else:
            thresholds = [0.001, 0.01, 0.05]
    
    all_preds = []
    all_targets = []
    all_lower_ci = []
    all_upper_ci = []
    all_sample_ids = []
    all_detection_probs = {t: [] for t in thresholds}
    
    # Create progress bar for evaluation
    eval_bar = tqdm(data_loader, desc="Evaluating", position=0)
    
    with torch.no_grad():
        for batch in eval_bar:
            # Handle different batch formats
            if len(batch) == 3:
                marker_values, coverage, y_true = batch
                sample_ids = None
            elif len(batch) == 4:
                marker_values, coverage, y_true, sample_ids = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            marker_values = marker_values.to(device)
            coverage = coverage.to(device)
            y_true = y_true.to(device)
            
            # Handle different model types
            if isinstance(model, CancerDetectionEnsemble):
                # Ensemble model
                mu, phi, det_probs = model(marker_values, coverage)
                estimate, ci, uncertainty = model.get_estimate_and_ci(mu, phi)
            else:
                # Single model
                if hasattr(model, 'forward_with_detection'):
                    # Enhanced model with detection
                    mu, phi, det_probs, _ = model.forward_with_detection(marker_values, coverage)
                else:
                    # Standard model
                    mu, phi, det_probs, _ = model(marker_values, coverage)
                
                estimate, ci, uncertainty = model.get_estimate_and_ci(mu, phi)
            
            # Store predictions
            all_preds.append(estimate.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
            all_lower_ci.append(ci[:, 0:1].cpu().numpy())
            all_upper_ci.append(ci[:, 1:2].cpu().numpy())
            
            # Store detection probabilities
            for i, threshold in enumerate(thresholds):
                if i < len(det_probs):
                    all_detection_probs[threshold].append(det_probs[i].cpu().numpy())
            
            # Store sample IDs if available
            if sample_ids is not None:
                all_sample_ids.extend(sample_ids)
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_lower_ci = np.concatenate(all_lower_ci)
    all_upper_ci = np.concatenate(all_upper_ci)
    
    for threshold in thresholds:
        if threshold in all_detection_probs and all_detection_probs[threshold]:
            all_detection_probs[threshold] = np.concatenate(all_detection_probs[threshold])
    
    # Create sample ID list if not available
    if not all_sample_ids:
        all_sample_ids = [f"sample_{i}" for i in range(len(all_preds))]
    
    # Calculate regression metrics
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    # Calculate percentage of targets within CI
    in_ci = ((all_targets >= all_lower_ci) & (all_targets <= all_upper_ci)).mean()
    
    # Calculate average CI width
    ci_width = (all_upper_ci - all_lower_ci).mean()
    
    # Calculate detection metrics
    detection_metrics = {}
    for threshold in thresholds:
        if threshold in all_detection_probs and len(all_detection_probs[threshold]) > 0:
            binary_targets = (all_targets >= threshold).astype(float)
            probs = all_detection_probs[threshold].flatten()
            
            # AUC
            try:
                auc_score = roc_auc_score(binary_targets, probs)
            except:
                auc_score = float('nan')
            
            # Sensitivity at 95% specificity
            try:
                # Sort by probability
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                sorted_targets = binary_targets[sorted_indices]
                
                # Get number of negatives
                n_negatives = np.sum(1 - sorted_targets)
                
                # Calculate number of false positives allowed for 95% specificity
                n_false_positives_allowed = int(0.05 * n_negatives)
                
                # Count false positives until threshold
                n_false_positives = 0
                threshold_idx = 0
                
                for i, (prob, target) in enumerate(zip(sorted_probs, sorted_targets)):
                    if target == 0:  # Negative sample
                        n_false_positives += 1
                    
                    if n_false_positives > n_false_positives_allowed:
                        threshold_idx = i - 1
                        break
                
                if threshold_idx == 0:
                    # Not enough false positives, use the last index
                    threshold_idx = len(sorted_probs) - 1
                
                # Calculate sensitivity at this threshold
                threshold_prob = sorted_probs[threshold_idx]
                predictions = (probs >= threshold_prob).astype(float)
                true_positives = np.sum(predictions * binary_targets)
                sensitivity = true_positives / np.sum(binary_targets) if np.sum(binary_targets) > 0 else 0
            except:
                sensitivity = float('nan')
            
            # Average precision
            try:
                precision, recall, _ = precision_recall_curve(binary_targets, probs)
                avg_precision = auc(recall, precision)
            except:
                avg_precision = float('nan')
            
            detection_metrics[threshold] = {
                'auc': float(auc_score),
                'sensitivity_at_95spec': float(sensitivity),
                'average_precision': float(avg_precision)
            }
    
    # Calculate enhanced regression metrics
    enhanced_metrics = calculate_enhanced_metrics(all_targets, all_preds, all_lower_ci, all_upper_ci)
    
    # Log results
    logger.info("\nRESULTS:")
    logger.info(f"Number of samples: {len(all_targets)}")
    logger.info(f"R² Score: {r2:.6f}")
    logger.info(f"Mean Absolute Error: {mae:.6f}")
    logger.info(f"Targets within CI: {in_ci * 100:.2f}%")
    logger.info(f"Average CI Width: {ci_width:.6f}")
    
    # Log enhanced metrics
    logger.info("\nENHANCED METRICS:")
    logger.info(f"Weighted R² Score: {enhanced_metrics['weighted_r2']:.6f}")
    logger.info(f"Weighted MAE: {enhanced_metrics['weighted_mae']:.6f}")
    logger.info(f"Concordance Correlation Coefficient: {enhanced_metrics['concordance_correlation']:.6f}")
    
    # Log detection metrics
    logger.info("\nDETECTION METRICS:")
    for threshold, metrics in detection_metrics.items():
        logger.info(f"At {threshold:.3%} threshold:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Sensitivity at 95% specificity: {metrics['sensitivity_at_95spec']:.4f}")
        logger.info(f"  Average precision: {metrics['average_precision']:.4f}")
    
    # Create results dictionary
    results = {
        'predictions': all_preds.flatten().tolist(),
        'targets': all_targets.flatten().tolist(),
        'lower_ci': all_lower_ci.flatten().tolist(),
        'upper_ci': all_upper_ci.flatten().tolist(),
        'sample_ids': all_sample_ids,
        'metrics': {
            'r2': float(r2),
            'mae': float(mae),
            'in_ci_percentage': float(in_ci * 100),
            'ci_width': float(ci_width),
            **enhanced_metrics
        },
        'detection_metrics': {
            str(float(k)): {
                'auc': float(v['auc']), 
                'sensitivity_at_95spec': float(v['sensitivity_at_95spec']),
                'average_precision': float(v['average_precision'])
            } for k, v in detection_metrics.items()
        }
    }
    
    # Save results
    if output_dir:
        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'sample_id': all_sample_ids,
            'true_concentration': all_targets.flatten(),
            'predicted_concentration': all_preds.flatten(),
            'lower_ci': all_lower_ci.flatten(),
            'upper_ci': all_upper_ci.flatten(),
            'in_ci': ((all_targets.flatten() >= all_lower_ci.flatten()) & 
                     (all_targets.flatten() <= all_upper_ci.flatten())),
            'absolute_error': np.abs(all_targets.flatten() - all_preds.flatten()),
            'relative_error': np.abs((all_targets.flatten() - all_preds.flatten()) / 
                                    np.maximum(all_targets.flatten(), 1e-10)) * 100,
            'error_bands': get_error_bands(all_targets.flatten(), all_preds.flatten())
        })
        
        # Add detection probabilities
        for threshold in thresholds:
            if threshold in all_detection_probs and len(all_detection_probs[threshold]) > 0:
                predictions_df[f'detection_prob_{threshold:.3f}'] = all_detection_probs[threshold].flatten()
                predictions_df[f'detection_{threshold:.3f}'] = all_targets.flatten() >= threshold
        
        predictions_file = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Detailed predictions saved to {predictions_file}")
        
        # Save summary results
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {results_file}")
        
        # Create enhanced visualisations
        viz_dir = os.path.join(output_dir, 'visualisations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create standard visualisations if available
        try:
            from deep_conv.detect.visualise import create_visualisations
            
            logger.info("Creating standard visualisations...")
            viz_metrics = create_visualisations(
                predictions=all_preds.flatten(), 
                ground_truth=all_targets.flatten(),
                output_dir=viz_dir
            )
            
            logger.info(f"Standard visualisations saved to {viz_dir}")
        except ImportError:
            logger.warning("Standard visualisation module not found. Skipping standard visualisations.")
        except Exception as e:
            logger.error(f"Error creating standard visualisations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Create enhanced visualisations
        try:
            logger.info("Creating enhanced visualisations...")
            
            # Create prediction plots with CI
            plot_predictions(all_preds, all_targets, all_lower_ci, all_upper_ci, viz_dir)
            
            # Create error band visualisation
            plot_error_bands(all_targets, all_preds, viz_dir)
            
            # Create concentration-dependent performance plot
            plot_concentration_dependent_performance(all_targets, all_preds, viz_dir)
            
            # Create error distribution plot
            plot_error_distribution(all_targets, all_preds, viz_dir)
            
            # Create CI reliability plot
            plot_ci_reliability(all_targets, all_lower_ci, all_upper_ci, viz_dir)
            
            logger.info(f"Enhanced visualisations saved to {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error creating enhanced visualisations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return results


def calculate_enhanced_metrics(targets, predictions, lower_ci=None, upper_ci=None):
    """
    Calculate enhanced regression metrics
    
    Args:
        targets: Ground truth values
        predictions: Predicted values
        lower_ci: Lower confidence interval (optional)
        upper_ci: Upper confidence interval (optional)
        
    Returns:
        metrics: Dictionary of enhanced metrics
    """
    # Flatten arrays
    targets = targets.flatten()
    predictions = predictions.flatten()
    
    # Basic metrics
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    # Weighted R² - more weight to higher concentrations
    weights = np.sqrt(targets + 0.0001)  # Avoid sqrt(0)
    weights = weights / np.mean(weights)  # Normalise weights
    
    # Calculate weighted R²
    y_weighted_mean = np.average(targets, weights=weights)
    total_variance = np.sum(weights * (targets - y_weighted_mean) ** 2)
    unexplained_variance = np.sum(weights * (targets - predictions) ** 2)
    weighted_r2 = 1 - (unexplained_variance / total_variance)
    
    # Weighted MAE
    weighted_mae = np.average(np.abs(targets - predictions), weights=weights)
    
    # Concordance Correlation Coefficient (CCC)
    mean_true = np.mean(targets)
    mean_pred = np.mean(predictions)
    var_true = np.var(targets)
    var_pred = np.var(predictions)
    
    # Covariance between true and predicted
    covariance = np.mean((targets - mean_true) * (predictions - mean_pred))
    
    # CCC calculation
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    
    # Calculate relative error metrics
    non_zero_mask = targets > 0
    relative_errors = np.zeros_like(targets)
    relative_errors[non_zero_mask] = np.abs(targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask]
    
    mean_relative_error = np.mean(relative_errors[non_zero_mask]) * 100  # as percentage
    median_relative_error = np.median(relative_errors[non_zero_mask]) * 100  # as percentage
    
    # Calculate percentage within error bands
    within_10pct = np.mean((np.abs(targets - predictions) <= 0.1 * np.maximum(targets, 1e-10)) | 
                           (np.abs(targets - predictions) < 0.001)) * 100
    within_20pct = np.mean((np.abs(targets - predictions) <= 0.2 * np.maximum(targets, 1e-10)) | 
                           (np.abs(targets - predictions) < 0.001)) * 100
    
    # If confidence intervals are provided, calculate calibration metrics
    if lower_ci is not None and upper_ci is not None:
        lower_ci = lower_ci.flatten()
        upper_ci = upper_ci.flatten()
        
        # Percentage of samples within CI
        in_ci = np.mean((targets >= lower_ci) & (targets <= upper_ci)) * 100
        
        # Average CI width
        ci_width = np.mean(upper_ci - lower_ci)
        
        # Normalised CI width (by target value)
        non_zero_mask = targets > 0
        norm_ci_width = np.zeros_like(targets)
        norm_ci_width[non_zero_mask] = (upper_ci[non_zero_mask] - lower_ci[non_zero_mask]) / targets[non_zero_mask]
        mean_norm_ci_width = np.mean(norm_ci_width[non_zero_mask])
        
        # Create CI calibration curve
        alphas = np.linspace(0, 1, 20)
        expected_coverage = alphas * 100
        actual_coverage = []
        
        for alpha in alphas:
            half_interval = alpha / 2
            lower = predictions - (upper_ci - predictions) * half_interval / 0.975
            upper = predictions + (upper_ci - predictions) * half_interval / 0.975
            coverage = np.mean((targets >= lower) & (targets <= upper)) * 100
            actual_coverage.append(coverage)
        
        # Calculate CI calibration error (RMSE between expected and actual coverage)
        ci_calibration_error = np.sqrt(np.mean((np.array(expected_coverage) - np.array(actual_coverage)) ** 2))
        
    else:
        in_ci = None
        ci_width = None
        mean_norm_ci_width = None
        ci_calibration_error = None
    
    # Return all metrics
    metrics = {
        'weighted_r2': float(weighted_r2),
        'weighted_mae': float(weighted_mae),
        'concordance_correlation': float(ccc),
        'mean_relative_error': float(mean_relative_error),
        'median_relative_error': float(median_relative_error),
        'within_10pct': float(within_10pct),
        'within_20pct': float(within_20pct)
    }
    
    # Add CI metrics if available
    if in_ci is not None:
        metrics.update({
            'in_ci_percentage': float(in_ci),
            'ci_width': float(ci_width),
            'mean_Normalised_ci_width': float(mean_norm_ci_width),
            'ci_calibration_error': float(ci_calibration_error)
        })
    
    return metrics


def get_error_bands(targets, predictions):
    """
    Categorise errors into bands based on relative error
    
    Args:
        targets: Ground truth values
        predictions: Predicted values
        
    Returns:
        error_bands: Array of error band categories
    """
    # Calculate absolute errors
    abs_errors = np.abs(targets - predictions)
    
    # Calculate relative errors for non-zero targets
    relative_errors = np.zeros_like(targets)
    non_zero_mask = targets > 0
    relative_errors[non_zero_mask] = abs_errors[non_zero_mask] / targets[non_zero_mask]
    
    # For zero targets, use absolute error
    zero_mask = ~non_zero_mask
    
    # Create error bands
    error_bands = np.full_like(targets, fill_value='', dtype=object)
    
    # Non-zero targets use relative error bands
    error_bands[non_zero_mask & (relative_errors <= 0.1)] = '<10%'
    error_bands[non_zero_mask & (relative_errors > 0.1) & (relative_errors <= 0.2)] = '10-20%'
    error_bands[non_zero_mask & (relative_errors > 0.2) & (relative_errors <= 0.5)] = '20-50%'
    error_bands[non_zero_mask & (relative_errors > 0.5)] = '>50%'
    
    # Zero targets use absolute error bands
    error_bands[zero_mask & (abs_errors <= 0.001)] = '<0.1%'
    error_bands[zero_mask & (abs_errors > 0.001) & (abs_errors <= 0.01)] = '0.1-1%'
    error_bands[zero_mask & (abs_errors > 0.01)] = '>1%'
    
    return error_bands


def plot_error_bands(targets, predictions, output_dir):
    """
    Create visualisation of error bands at different concentration ranges
    
    Args:
        targets: Ground truth values
        predictions: Predicted values
        output_dir: Directory to save visualisation
    """
    import os
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    import pandas as pd
    
    # Flatten arrays
    targets = targets.flatten()
    predictions = predictions.flatten()
    
    # Calculate error bands
    error_bands = get_error_bands(targets, predictions)
    
    # Create DataFrame
    df = pd.DataFrame({
        'True Concentration': targets,
        'Predicted Concentration': predictions,
        'Error Band': error_bands
    })
    
    # Create log-scaled concentration bins
    bin_edges = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    bin_labels = ['0', '0.001-0.01', '0.01-0.05', '0.05-0.1', '0.1-0.5', '>0.5']
    
    df['Concentration Range'] = pd.cut(df['True Concentration'], bins=bin_edges, labels=bin_labels, right=False)
    
    # Count samples in each bin and error band
    pivot_table = pd.pivot_table(
        df, 
        values='True Concentration',
        index='Concentration Range',
        columns='Error Band',
        aggfunc='count',
        fill_value=0
    )
    
    # Convert to percentages
    row_sums = pivot_table.sum(axis=1)
    pivot_pct = pivot_table.div(row_sums, axis=0) * 100
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Desired order of error bands
    error_band_order = ['<10%', '10-20%', '20-50%', '>50%', '<0.1%', '0.1-1%', '>1%']
    
    # Add each error band as a bar
    for error_band in error_band_order:
        if error_band in pivot_pct.columns:
            fig.add_trace(go.Bar(
                x=pivot_pct.index,
                y=pivot_pct[error_band],
                name=error_band,
                text=pivot_pct[error_band].apply(lambda x: f'{x:.1f}%'),
                textposition='inside'
            ))
    
    # Update layout
    fig.update_layout(
        title='Prediction Error Bands by Concentration Range',
        xaxis_title='True Concentration Range',
        yaxis_title='Percentage',
        barmode='stack',
        template='plotly_white',
        legend_title='Error Band',
        width=900,
        height=600
    )
    
    # Save figure
    fig.write_html(os.path.join(output_dir, 'error_bands_by_concentration.html'))
    fig.write_image(os.path.join(output_dir, 'error_bands_by_concentration.png'), scale=2)
    
    # Create table with actual counts
    fig_counts = go.Figure(data=[go.Table(
        header=dict(
            values=['Concentration Range'] + list(pivot_table.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[pivot_table.index] + [pivot_table[col] for col in pivot_table.columns],
            fill_color='lavender',
            align='left'
        )
    )])
    
    fig_counts.update_layout(
        title='Sample Counts by Concentration Range and Error Band',
        width=900,
        height=400
    )
    
    # Save table figure
    fig_counts.write_html(os.path.join(output_dir, 'error_bands_counts.html'))
    fig_counts.write_image(os.path.join(output_dir, 'error_bands_counts.png'), scale=2)


def plot_concentration_dependent_performance(targets, predictions, output_dir):
    """
    Create visualisation showing how performance metrics vary with concentration
    
    Args:
        targets: Ground truth values
        predictions: Predicted values
        output_dir: Directory to save visualisation
    """
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Flatten arrays
    targets = targets.flatten()
    predictions = predictions.flatten()
    
    # Create log-scaled concentration bins
    bin_edges = np.logspace(-4, 0, 11)  # 10 bins from 0.0001 to 1.0
    bin_midpoints = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    
    # Initialise arrays to store metrics by bin
    mae_by_bin = []
    mre_by_bin = []  # Mean relative error
    r2_by_bin = []
    sample_counts = []
    
    # Calculate metrics for each bin
    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i+1]
        
        # Get samples in this bin
        bin_mask = (targets >= low) & (targets < high)
        bin_targets = targets[bin_mask]
        bin_predictions = predictions[bin_mask]
        
        # Store sample count
        sample_counts.append(np.sum(bin_mask))
        
        # Skip bins with too few samples
        if len(bin_targets) < 5:
            mae_by_bin.append(np.nan)
            mre_by_bin.append(np.nan)
            r2_by_bin.append(np.nan)
            continue
        
        # Calculate metrics
        mae = np.mean(np.abs(bin_targets - bin_predictions))
        
        # Relative error (avoid division by zero)
        rel_errors = np.abs(bin_targets - bin_predictions) / np.maximum(bin_targets, 1e-10)
        mre = np.mean(rel_errors) * 100  # as percentage
        
        # R² (handle case where all targets in bin are identical)
        if np.std(bin_targets) > 0:
            r2 = r2_score(bin_targets, bin_predictions)
        else:
            r2 = np.nan
        
        # Store metrics
        mae_by_bin.append(mae)
        mre_by_bin.append(mre)
        r2_by_bin.append(r2)
    
    # Create figure with two subplots (MAE and MRE)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Mean Absolute Error by Concentration', 'Mean Relative Error by Concentration'),
        vertical_spacing=0.1
    )
    
    # Add MAE trace
    fig.add_trace(
        go.Scatter(
            x=bin_midpoints,
            y=mae_by_bin,
            mode='lines+markers',
            name='MAE',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Add MRE trace
    fig.add_trace(
        go.Scatter(
            x=bin_midpoints,
            y=mre_by_bin,
            mode='lines+markers',
            name='MRE (%)',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Add sample count as a secondary axis on the first subplot
    fig.add_trace(
        go.Bar(
            x=bin_midpoints,
            y=sample_counts,
            name='Sample Count',
            marker_color='rgba(200, 200, 200, 0.5)',
            opacity=0.5,
            yaxis='y2'
        ),
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        xaxis_type='log',
        xaxis_title='True Concentration (log scale)',
        yaxis_title='Mean Absolute Error',
        yaxis2=dict(
            title='Sample Count',
            overlaying='y',
            side='right'
        ),
        yaxis3_title='Mean Relative Error (%)',
        template='plotly_white',
        width=900,
        height=800,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Save figure
    fig.write_html(os.path.join(output_dir, 'concentration_dependent_performance.html'))
    fig.write_image(os.path.join(output_dir, 'concentration_dependent_performance.png'), scale=2)
    
    # Create R² by concentration plot
    fig_r2 = go.Figure()
    
    # Add R² trace
    fig_r2.add_trace(
        go.Scatter(
            x=bin_midpoints,
            y=r2_by_bin,
            mode='lines+markers',
            name='R²',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        )
    )
    
    # Add sample count as a secondary axis
    fig_r2.add_trace(
        go.Bar(
            x=bin_midpoints,
            y=sample_counts,
            name='Sample Count',
            marker_color='rgba(200, 200, 200, 0.5)',
            opacity=0.5,
            yaxis='y2'
        )
    )
    
    # Update layout
    fig_r2.update_layout(
        title='R² Score by Concentration',
        xaxis_type='log',
        xaxis_title='True Concentration (log scale)',
        yaxis_title='R² Score',
        yaxis2=dict(
            title='Sample Count',
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Save figure
    fig_r2.write_html(os.path.join(output_dir, 'r2_by_concentration.html'))
    fig_r2.write_image(os.path.join(output_dir, 'r2_by_concentration.png'), scale=2)


def plot_error_distribution(targets, predictions, output_dir):
    """
    Create visualisation of error distribution at different concentration ranges
    
    Args:
        targets: Ground truth values
        predictions: Predicted values
        output_dir: Directory to save visualisation
    """
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Flatten arrays
    targets = targets.flatten()
    predictions = predictions.flatten()
    
    # Calculate absolute and relative errors
    absolute_errors = predictions - targets
    
    # Create log-scaled concentration bins
    bins = [0, 0.001, 0.01, 0.05, 0.1, 1.0]
    bin_labels = ['0', '0.001-0.01', '0.01-0.05', '0.05-0.1', '>0.1']
    
    # Create subplots
    fig = make_subplots(
        rows=len(bins) - 1, 
        cols=1,
        subplot_titles=[f'Error Distribution for Concentration Range: {label}' for label in bin_labels],
        vertical_spacing=0.05
    )
    
    # For each bin, create error histogram
    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i+1]
        bin_mask = (targets >= low) & (targets < high)
        
        bin_errors = absolute_errors[bin_mask]
        
        if len(bin_errors) < 5:
            continue
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=bin_errors,
                name=bin_labels[i],
                opacity=0.7,
                nbinsx=30,
                marker_color='blue'
            ),
            row=i+1, col=1
        )
        
        # Add vertical line at zero
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=0, y1=1,
            yref="paper",
            xref=f"x{i+1}",
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Calculate and display stats
        mean_error = np.mean(bin_errors)
        std_error = np.std(bin_errors)
        median_error = np.median(bin_errors)
        
        text = f"Mean: {mean_error:.6f}<br>Std Dev: {std_error:.6f}<br>Median: {median_error:.6f}"
        
        fig.add_annotation(
            x=0.05, y=0.9,
            xref=f"x{i+1} domain",
            yref=f"y{i+1} domain",
            text=text,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title='Error Distribution by Concentration Range',
        template='plotly_white',
        width=900,
        height=150 * (len(bins) - 1),
        showlegend=False
    )
    
    # Save figure
    fig.write_html(os.path.join(output_dir, 'error_distribution_by_concentration.html'))
    fig.write_image(os.path.join(output_dir, 'error_distribution_by_concentration.png'), scale=2)
    
    # Create relative error distribution for non-zero targets
    non_zero_mask = targets > 0
    non_zero_targets = targets[non_zero_mask]
    non_zero_predictions = predictions[non_zero_mask]
    
    relative_errors = (non_zero_predictions - non_zero_targets) / non_zero_targets * 100  # as percentage
    
    # Create figure for relative errors
    fig_relative = go.Figure()
    
    # Add histogram
    fig_relative.add_trace(
        go.Histogram(
            x=relative_errors,
            opacity=0.7,
            nbinsx=50,
            marker_color='green',
            name='Relative Error'
        )
    )
    
    # Add vertical line at zero
    fig_relative.add_shape(
        type="line",
        x0=0, y0=0,
        x1=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Calculate and display stats
    mean_rel_error = np.mean(relative_errors)
    std_rel_error = np.std(relative_errors)
    median_rel_error = np.median(relative_errors)
    
    within_10pct = np.mean(np.abs(relative_errors) <= 10) * 100
    within_20pct = np.mean(np.abs(relative_errors) <= 20) * 100
    
    text = (f"Mean: {mean_rel_error:.2f}%<br>"
           f"Std Dev: {std_rel_error:.2f}%<br>"
           f"Median: {median_rel_error:.2f}%<br>"
           f"Within ±10%: {within_10pct:.1f}%<br>"
           f"Within ±20%: {within_20pct:.1f}%")
    
    fig_relative.add_annotation(
        x=0.05, y=0.9,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig_relative.update_layout(
        title='Relative Error Distribution (Non-zero Targets Only)',
        xaxis_title='Relative Error (%)',
        yaxis_title='Count',
        template='plotly_white',
        width=900,
        height=500
    )
    
    # Set range to focus on main distribution
    p95 = np.percentile(np.abs(relative_errors), 95)
    fig_relative.update_xaxes(range=[-p95, p95])
    
    # Save figure
    fig_relative.write_html(os.path.join(output_dir, 'relative_error_distribution.html'))
    fig_relative.write_image(os.path.join(output_dir, 'relative_error_distribution.png'), scale=2)


def plot_ci_reliability(targets, lower_ci, upper_ci, output_dir):
    """
    Create visualisation of confidence interval reliability
    
    Args:
        targets: Ground truth values
        lower_ci: Lower confidence interval values
        upper_ci: Upper confidence interval values
        output_dir: Directory to save visualisation
    """
    import os
    import plotly.graph_objects as go
    import numpy as np
    
    # Flatten arrays
    targets = targets.flatten()
    lower_ci = lower_ci.flatten()
    upper_ci = upper_ci.flatten()
    
    # Calculate percentage of targets within CI
    in_ci = (targets >= lower_ci) & (targets <= upper_ci)
    in_ci_percentage = np.mean(in_ci) * 100
    
    # Calculate expected vs actual coverage for different confidence levels
    alphas = np.linspace(0, 1, 20)
    expected_coverage = alphas * 100
    actual_coverage = []
    
    # Calculate midpoint of predictions (for scaling CIs)
    predictions = (lower_ci + upper_ci) / 2
    half_width = (upper_ci - lower_ci) / 2
    
    for alpha in alphas:
        # Scale CI according to alpha (from original 95% CI)
        half_interval = alpha / 2
        lower = predictions - half_width * half_interval / 0.475  # 0.95/2 = 0.475
        upper = predictions + half_width * half_interval / 0.475
        
        # Calculate actual coverage
        coverage = np.mean((targets >= lower) & (targets <= upper)) * 100
        actual_coverage.append(coverage)
    
    # Create figure
    fig = go.Figure()
    
    # Add perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=expected_coverage,
            y=expected_coverage,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Perfect Calibration'
        )
    )
    
    # Add actual calibration curve
    fig.add_trace(
        go.Scatter(
            x=expected_coverage,
            y=actual_coverage,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            name='Actual Coverage'
        )
    )
    
    # Calculate calibration error (RMSE between expected and actual coverage)
    ci_calibration_error = np.sqrt(np.mean((np.array(expected_coverage) - np.array(actual_coverage)) ** 2))
    
    # Add annotation with CI statistics
    text = (f"95% CI Coverage: {in_ci_percentage:.1f}%<br>"
           f"Calibration Error: {ci_calibration_error:.2f}%<br>"
           f"Mean CI Width: {np.mean(upper_ci - lower_ci):.6f}")
    
    fig.add_annotation(
        x=0.05, y=0.95,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title='Confidence Interval Reliability',
        xaxis_title='Expected Coverage (%)',
        yaxis_title='Actual Coverage (%)',
        template='plotly_white',
        width=800,
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Make square plot
    fig.update_layout(yaxis=dict(scaleanchor='x', scaleratio=1))
    
    # Save figure
    fig.write_html(os.path.join(output_dir, 'ci_reliability.html'))
    fig.write_image(os.path.join(output_dir, 'ci_reliability.png'), scale=2)
    
    # Create concentration-dependent CI width plot
    fig_width = go.Figure()
    
    # Create log-scaled concentration bins
    bin_edges = np.logspace(-4, 0, 11)  # 10 bins from 0.0001 to 1.0
    bin_midpoints = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    
    # Initialise arrays to store metrics by bin
    width_by_bin = []
    norm_width_by_bin = []  # Normalised by concentration
    in_ci_by_bin = []
    sample_counts = []
    
    # Calculate metrics for each bin
    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i+1]
        
        # Get samples in this bin
        bin_mask = (targets >= low) & (targets < high)
        bin_targets = targets[bin_mask]
        bin_lower = lower_ci[bin_mask]
        bin_upper = upper_ci[bin_mask]
        
        # Store sample count
        sample_counts.append(np.sum(bin_mask))
        
        # Skip bins with too few samples
        if len(bin_targets) < 5:
            width_by_bin.append(np.nan)
            norm_width_by_bin.append(np.nan)
            in_ci_by_bin.append(np.nan)
            continue
        
        # Calculate metrics
        width = np.mean(bin_upper - bin_lower)
        
        # Normalised width (avoid division by zero)
        norm_width = np.mean((bin_upper - bin_lower) / np.maximum(bin_targets, 1e-10))
        
        # In-CI percentage
        in_ci_pct = np.mean((bin_targets >= bin_lower) & (bin_targets <= bin_upper)) * 100
        
        # Store metrics
        width_by_bin.append(width)
        norm_width_by_bin.append(norm_width)
        in_ci_by_bin.append(in_ci_pct)
    
    # Add absolute width trace
    fig_width.add_trace(
        go.Scatter(
            x=bin_midpoints,
            y=width_by_bin,
            mode='lines+markers',
            name='Absolute CI Width',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        )
    )
    
    # Update layout
    fig_width.update_layout(
        title='CI Width by Concentration',
        xaxis_type='log',
        xaxis_title='True Concentration (log scale)',
        yaxis_title='Mean CI Width',
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Save figure
    fig_width.write_html(os.path.join(output_dir, 'ci_width_by_concentration.html'))
    fig_width.write_image(os.path.join(output_dir, 'ci_width_by_concentration.png'), scale=2)
    
    # Create Normalised CI width plot
    fig_norm = go.Figure()
    
    # Add Normalised width trace
    fig_norm.add_trace(
        go.Scatter(
            x=bin_midpoints,
            y=norm_width_by_bin,
            mode='lines+markers',
            name='Normalised CI Width',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        )
    )
    
    # Update layout
    fig_norm.update_layout(
        title='Normalised CI Width by Concentration',
        xaxis_type='log',
        xaxis_title='True Concentration (log scale)',
        yaxis_title='Mean CI Width / Concentration',
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Save figure
    fig_norm.write_html(os.path.join(output_dir, 'Normalised_ci_width_by_concentration.html'))
    fig_norm.write_image(os.path.join(output_dir, 'Normalised_ci_width_by_concentration.png'), scale=2)
    
    # Create CI coverage by concentration plot
    fig_coverage = go.Figure()
    
    # Add in-CI percentage trace
    fig_coverage.add_trace(
        go.Scatter(
            x=bin_midpoints,
            y=in_ci_by_bin,
            mode='lines+markers',
            name='In-CI Percentage',
            line=dict(color='purple', width=2),
            marker=dict(size=8)
        )
    )
    
    # Add ideal coverage line (95%)
    fig_coverage.add_shape(
        type="line",
        x0=min(bin_midpoints),
        y0=95,
        x1=max(bin_midpoints),
        y1=95,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Update layout
    fig_coverage.update_layout(
        title='CI Coverage by Concentration',
        xaxis_type='log',
        xaxis_title='True Concentration (log scale)',
        yaxis_title='In-CI Percentage',
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Save figure
    fig_coverage.write_html(os.path.join(output_dir, 'ci_coverage_by_concentration.html'))
    fig_coverage.write_image(os.path.join(output_dir, 'ci_coverage_by_concentration.png'), scale=2)


def plot_predictions(predictions, targets, lower_ci, upper_ci, output_dir):
    """Plot predictions vs targets with confidence intervals"""
    import os
    import numpy as np
    import plotly.graph_objects as go
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by targets for clearer visualisation
    sorted_indices = np.argsort(targets.flatten())
    sorted_targets = targets.flatten()[sorted_indices]
    sorted_preds = predictions.flatten()[sorted_indices]
    sorted_lower = lower_ci.flatten()[sorted_indices]
    sorted_upper = upper_ci.flatten()[sorted_indices]
    
    # Plot predictions with CI (sorted by true value)
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sorted_targets)),
            y=sorted_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sorted_targets)),
            y=sorted_lower,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0, 176, 246, 0.2)',
            fill='tonexty',
            showlegend=True,
            name='95% CI'
        )
    )
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sorted_targets)),
            y=sorted_preds,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Predictions'
        )
    )
    
    # Add targets
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sorted_targets)),
            y=sorted_targets,
            mode='lines',
            line=dict(color='red', width=2),
            name='True Values'
        )
    )
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    # Update layout
    fig.update_layout(
        title=f'Predictions vs Targets (sorted) (R² = {r2:.4f}, MAE = {mae:.4f})',
        xaxis_title='Sample Index (sorted by true value)',
        yaxis_title='Cell type Concentration',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        width=900,
        height=600
    )
    
    # Save figure
    fig.write_html(os.path.join(output_dir, 'predictions_vs_targets.html'))
    fig.write_image(os.path.join(output_dir, 'predictions_vs_targets.png'), scale=2)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add points
    fig.add_trace(
        go.Scatter(
            x=targets.flatten(),
            y=predictions.flatten(),
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6
            ),
            error_y=dict(
                type='data',
                symmetric=False,
                array=upper_ci.flatten() - predictions.flatten(),
                arrayminus=predictions.flatten() - lower_ci.flatten(),
                thickness=1.5,
                width=3
            ),
            name='Predictions with 95% CI'
        )
    )
    
    # Add identity line
    min_val = min(min(targets.min(), predictions.min()), 0)
    max_val = max(targets.max(), predictions.max()) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'True vs Predicted (R² = {r2:.4f}, MAE = {mae:.4f})',
        xaxis_title='True Cell type Concentration',
        yaxis_title='Predicted Cell type Concentration',
        template='plotly_white',
        width=800,
        height=800,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Make square plot
    fig.update_layout(yaxis=dict(scaleanchor='x', scaleratio=1))
    
    # Save figure
    fig.write_html(os.path.join(output_dir, 'true_vs_predicted.html'))
    fig.write_image(os.path.join(output_dir, 'true_vs_predicted.png'), scale=2)


def run_evaluation(model_dir, input_dir, output_dir=None, device=None):
    """
    Load a model and evaluate it on data from the input directory
    
    Args:
        model_dir: Directory containing the trained model
        input_dir: Directory containing the data to evaluate
        output_dir: Directory to save evaluation results (defaults to model_dir/evaluation)
        device: Device to run evaluation on (defaults to CUDA if available)
        
    Returns:
        results: Dictionary of evaluation results
    """
    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(model_dir, 'evaluation', os.path.basename(input_dir))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting evaluation")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        model, args = load_model(model_dir, device)
        
        # Get thresholds from model or args
        if hasattr(model, 'detection_thresholds'):
            thresholds = model.detection_thresholds
        elif 'detection_thresholds' in args:
            thresholds = args['detection_thresholds']
        else:
            thresholds = [0.001, 0.01, 0.05]
            
        # Load dataset
        logger.info(f"Loading data from {input_dir}")
        if 'atlas_path' in args:
            atlas_path = args['atlas_path']
            logger.info(f"Using atlas from training: {atlas_path}")
        else:
            # Try to find atlas in the model directory
            atlas_files = [f for f in os.listdir(model_dir) if f.endswith('.bed')]
            if atlas_files:
                atlas_path = os.path.join(model_dir, atlas_files[0])
                logger.info(f"Found atlas in model directory: {atlas_path}")
            else:
                atlas_path = None
                logger.warning("No atlas file specified or found in model directory")
        
        target_cell_type = args.get('target_cell_type', None)
        target_cell_idx = args.get('target_cell_idx', None)
        
        # Load dataset
        data_loader = prepare_data_for_evaluation(
            data_dir=input_dir,
            atlas_path=atlas_path,
            target_cell_type=target_cell_type,
            target_cell_idx=target_cell_idx,
        )
        
        # Evaluate model
        results = evaluate_model(model, data_loader, output_dir, thresholds, device)
        
        logger.info("Evaluation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained cancer detection model')

    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the trained model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the data to evaluate')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None, help='Device to run evaluation on (cuda or cpu)')

    return parser.parse_args()


# python -m deep_conv.detect.evaluate \
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/CpGenie_OAC/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval_single_cell_clinical/OAC/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval_single_cell_clinical/OAC/CpGenie_OAC

# python -m deep_conv.detect.evaluate \
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/CpGenie_T-cells/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval_single_cell_clinical/T-cells/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/training/oac.blood+gi+tum.l4/eval_single_cell_clinical/T-cells/CpGenie_T-cells

if __name__ == '__main__':
    args = parse_args()
    results = run_evaluation(args.model_dir, args.input_dir, args.output_dir, args.device)
    if results is None:
        sys.exit(1)
