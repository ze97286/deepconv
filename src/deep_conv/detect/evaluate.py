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

from deep_conv.detect.preprocess import load_dataset_from_directory
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


def load_model(model_dir, device='cuda'):
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
                use_pos_encoding=config.get('use_pos_encoding', True),
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
            use_pos_encoding=args.get('use_pos_encoding', True),
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
    Evaluate a trained model on a dataset
    
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
    logger.info("Starting model evaluation...")
    
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
    
    # Log results
    logger.info("\nRESULTS:")
    logger.info(f"Number of samples: {len(all_targets)}")
    logger.info(f"R² Score: {r2:.6f}")
    logger.info(f"Mean Absolute Error: {mae:.6f}")
    logger.info(f"Targets within CI: {in_ci * 100:.2f}%")
    logger.info(f"Average CI Width: {ci_width:.6f}")
    
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
            'ci_width': float(ci_width)
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
                     (all_targets.flatten() <= all_upper_ci.flatten()))
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
        
        # Create visualizations if possible
        try:
            from deep_conv.detect.visualise import create_visualizations
            
            viz_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            logger.info("Creating visualizations...")
            viz_metrics = create_visualizations(
                predictions=all_preds.flatten(), 
                ground_truth=all_targets.flatten(),
                output_dir=viz_dir
            )
            
            logger.info(f"Visualizations saved to {viz_dir}")
            
            # Also create prediction plots with CI
            plot_predictions(all_preds, all_targets, all_lower_ci, all_upper_ci, viz_dir)
            
        except ImportError:
            logger.warning("Visualization module not found. Skipping visualizations.")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return results


def plot_predictions(predictions, targets, lower_ci, upper_ci, output_dir):
    """Plot predictions vs targets with confidence intervals"""
    import os
    import numpy as np
    import plotly.graph_objects as go
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by targets for clearer visualization
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
        yaxis_title='Cancer Concentration',
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
        xaxis_title='True Cancer Concentration',
        yaxis_title='Predicted Cancer Concentration',
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
        data_loader = load_dataset_from_directory(
            input_dir,
            atlas_path=atlas_path,
            target_cell_type=target_cell_type,
            target_cell_idx=target_cell_idx,
            batch_size=args.get('batch_size', 32),
            shuffle=False
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


if __name__ == '__main__':
    args = parse_args()
    results = run_evaluation(args.model_dir, args.input_dir, args.output_dir, args.device)
    if results is None:
        sys.exit(1)