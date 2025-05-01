import os
import torch
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm

from deep_conv.detect.preprocess import prepare_data_for_predict
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

    # Standard single model
    # Get model parameters
    model_state = checkpoint.get('model', None)
    if model_state is None:
        # Some checkpoints store the model state directly
        model_state = checkpoint

    # Get num_markers from the first layer weights if not in args
    if isinstance(model_state, dict):
        # Check for background_level dimension
        if 'background_level' in model_state:
            bg_shape = model_state['background_level'].shape
            if len(bg_shape) > 1 and bg_shape[1] > 1:
                # This is a marker-specific background, extract the number of markers
                args['num_markers'] = bg_shape[1]
                args['marker_specific_bg'] = True

        # If still not found, try to infer from value_embedding or other layers
        if 'num_markers' not in args:
            for key in model_state:
                if 'value_embedding.weight' in key:
                    feature_dim = model_state[key].shape[1]
                    args['feature_dim'] = feature_dim * 2  # Assuming feature_dim//2 in the embedding
                    break
                elif 'embedding' in key and 'weight' in key:
                    # Try to infer from embedding dimensions
                    shape = model_state[key].shape
                    if len(shape) > 1:
                        for dim in shape:
                            if dim > 50:  # Likely the marker dimension
                                args['num_markers'] = dim
                                break

    # Set defaults with fallbacks
    detection_thresholds = args.get('detection_thresholds', [0.001, 0.01, 0.05])
    # Convert from string if needed
    if isinstance(detection_thresholds, str):
        try:
            detection_thresholds = json.loads(detection_thresholds)
        except:
            detection_thresholds = [0.001, 0.01, 0.05]

    # Create and load model
    try:
        model = EnhancedCancerDetectionModel(
            num_markers=args.get("num_markers", 136),
            feature_dim=args.get("feature_dim", 128),
            num_heads=args.get("num_heads", 8),
            num_layers=args.get("num_layers", 3),
            dropout_rate=args.get("dropout_rate", 0.2),
            detection_thresholds=detection_thresholds,
            critical_ranges = args.get("critical_ranges", [(0.001,0.005,1.0),(0.005,0.01,1.0)]),
            marker_specific_bg=args.get("marker_specific_bg", True),
            min_reliable_coverage=args.get("min_reliable_coverage", 5.0),
            enable_adaptive_thresholds=args.get("enable_adaptive_thresholds", False)
        )

        # First try strict loading
        try:
            model.load_state_dict(model_state, strict=True)
        except Exception as e:
            logger.warning(f"Strict loading failed: {e}")
            # Try non-strict loading
            model.load_state_dict(model_state, strict=False)
            logger.info("Used non-strict loading instead")

        # Apply calibration values if available
        if 'calibration' in checkpoint:
            with torch.no_grad():
                if hasattr(model, 'calibration'):
                    model.calibration.fill_(checkpoint['calibration'].get('calibration_factor', 1.0))

                # Handle background level properly
                if hasattr(model, 'background_level'):
                    if 'global_bg_level' in checkpoint['calibration']:
                        # Single background level
                        bg_level = checkpoint['calibration'].get('global_bg_level', 0.05)
                        model.background_level.fill_(bg_level)
                    elif model.marker_specific_bg and 'marker_bg_levels' in checkpoint['calibration']:
                        # Marker-specific background levels (if shape matches)
                        bg_levels = checkpoint['calibration']['marker_bg_levels']
                        if isinstance(bg_levels, torch.Tensor) and bg_levels.shape == model.background_level.shape:
                            model.background_level.copy_(bg_levels)
                        else:
                            # Fall back to global stats if available
                            if 'marker_bg_stats' in checkpoint['calibration']:
                                stats = checkpoint['calibration']['marker_bg_stats']
                                median_val = stats.get('median', 0.05)
                                model.background_level.fill_(median_val)
                            else:
                                # Default fallback
                                model.background_level.fill_(0.05)

                if hasattr(model, 'low_calibration'):
                    model.low_calibration.fill_(checkpoint['calibration'].get('low_calibration_factor', 1.0))

    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

    # Move model to device
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    return model, args

def predict(model, marker_values, coverage, sample_ids, output_dir=None, thresholds=None, device='cpu'):
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
    with torch.no_grad():
        marker_values = marker_values.to(device)
        coverage = coverage.to(device)
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


    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_lower_ci = np.concatenate(all_lower_ci)
    all_upper_ci = np.concatenate(all_upper_ci)
    all_uncertainties = np.concatenate(all_uncertainties)
    all_sample_ids = sample_ids
    
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
        
        # Load dataset
        marker_values, coverage, sample_ids = prepare_data_for_predict(
            data_dir=input_dir,
            atlas_path=atlas_path,
            target_cell_type=target_cell_type,
        )
        
        # Evaluate model
        predict(model, marker_values, coverage, sample_ids, output_dir, thresholds, device)        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# python -m deep_conv.detect.predict \
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/oac_conc_focused/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/AB/cfDNA/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/oac_conc_focused

# python -m deep_conv.detect.predict \
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/CpGenie_T-cells/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/AB/cfDNA/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/CpGenie_T-cells

if __name__ == '__main__':
    args = parse_args()
    run_predict(args.model_dir, args.input_dir, args.output_dir, args.device)
