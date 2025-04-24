import os
import torch
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm

from deep_conv.detect.preprocess import prepare_data_for_evaluation
from deep_conv.detect.model import EnhancedCancerDetectionModel

def parse_args():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained cancer detection model')

    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the trained model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the data to evaluate')
    parser.add_argument('--output_dir',type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None, help='Device to run evaluation on (cuda or cpu)')

    return parser.parse_args()

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

def predict(model, data_loader, output_dir=None, thresholds=None, device='cpu'):
    """
    Predict using a trained model on a dataset and save the results to the output dir
    
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
    all_lower_ci = []
    all_upper_ci = []
    all_sample_ids = []
    all_uncertainties = []

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
           
            if hasattr(model, 'forward_with_detection'):
                # Enhanced model with detection
                mu, phi, det_probs, _ = model.forward_with_detection(marker_values, coverage)
            else:
                # Standard model
                mu, phi, det_probs, _ = model(marker_values, coverage)

            estimate, ci, uncertainty = model.get_estimate_and_ci(mu, phi)

            # Store predictions
            all_preds.append(estimate.cpu().numpy())
            all_lower_ci.append(ci[:, 0:1].cpu().numpy())
            all_upper_ci.append(ci[:, 1:2].cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())

            # Store sample IDs if available
            if sample_ids is not None:
                all_sample_ids.extend(sample_ids)

    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_lower_ci = np.concatenate(all_lower_ci)
    all_upper_ci = np.concatenate(all_upper_ci)
    all_uncertainties = np.concatenate(all_uncertainties)

    # Create sample ID list if not available
    if not all_sample_ids:
        all_sample_ids = [f"sample_{i}" for i in range(len(all_preds))]

    # Save results
    predictions_df = pd.DataFrame(
        {
            "sample_id": all_sample_ids,
            "estimated": all_preds.flatten(),
            "lower_ci": all_lower_ci.flatten(),
            "upper_ci": all_upper_ci.flatten(),
            "uncertainty": all_uncertainties.flatten(),
        }
    )

    predictions_file = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Detailed predictions saved to {predictions_file}")

def run_predict(model_dir, input_dir, output_dir=None, device=None):
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
        predict(model, data_loader, output_dir, thresholds, device)        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == '__main__':
    args = parse_args()
    run_predict(args.model_dir, args.input_dir, args.output_dir, args.device)
