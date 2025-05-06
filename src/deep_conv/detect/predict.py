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
    Load a trained model from a directory with enhanced compatibility
    
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
            # Try to find any .pt file
            pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if pt_files:
                model_path = os.path.join(model_dir, pt_files[0])
                logger.warning(f"Standard model files not found, using {pt_files[0]} instead")
            else:
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

    # Get model parameters
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        model_state = checkpoint['model']
    else:
        # Try other common keys
        for key in ['state_dict', 'model_state_dict', 'model_state']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                model_state = checkpoint[key]
                break
        else:
            # If no recognized key is found, assume the checkpoint itself is the state dict
            model_state = checkpoint

    # Get num_markers from the model state if not in args
    if 'num_markers' not in args:
        # Try to infer from marker_pos_embedding parameter size
        for key in model_state:
            if 'marker_pos_embedding' in key:
                args['num_markers'] = model_state[key].shape[1]
                logger.info(f"Inferred num_markers={args['num_markers']} from model state")
                break
            elif 'value_embedding.weight' in key:
                # Can also try to infer from other layer dimensions if needed
                pass

    # Set defaults with fallbacks
    defaults = {
        'num_markers': 136,
        'feature_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'dropout_rate': 0.2,
        'min_reliable_coverage': 3.0
    }
    
    for key, default_value in defaults.items():
        if key not in args:
            args[key] = default_value

    # Create and load model
    try:
        logger.info(f"Creating model with parameters: {args}")

        model = EnhancedCancerDetectionModel(
            num_markers=args['num_markers'],
            feature_dim=args['feature_dim'],
            num_heads=args['num_heads'],
            num_layers=args['num_layers'],
            dropout_rate=args['dropout_rate'],
            min_reliable_coverage=args.get('min_reliable_coverage', 3.0)
        )

        # Try strict loading first
        try:
            model.load_state_dict(model_state, strict=True)
            logger.info("Model loaded with strict=True")
        except Exception as e:
            # If strict loading fails, try non-strict loading
            logger.warning(f"Strict loading failed: {e}")
            logger.warning("Attempting non-strict loading...")
            model.load_state_dict(model_state, strict=False)
            logger.info("Model loaded with strict=False")

        # Apply calibration if available
        if 'calibration' in checkpoint and isinstance(checkpoint['calibration'], dict):
            with torch.no_grad():
                if hasattr(model, 'calibration'):
                    calibration_factor = checkpoint['calibration'].get('calibration_factor', 1.0)
                    model.calibration.fill_(calibration_factor)
                    logger.info(f"Applied calibration factor: {calibration_factor}")
                else:
                    logger.warning("Model has no calibration attribute, skipping calibration")

        if 'clinical_threshold' in checkpoint and isinstance(checkpoint['clinical_threshold'], (float, int)):
            print(f"clinical_threshold => {checkpoint['clinical_threshold']}")
            with torch.no_grad():
                if hasattr(model, 'clinical_threshold'):
                    threshold_value = checkpoint['clinical_threshold']
                    model.clinical_threshold.fill_(threshold_value)
                    logger.info(f"Applied clinical threshold: {threshold_value}")
                else:
                    logger.warning("Model has no clinical_threshold attribute, skipping threshold")


    except Exception as e:
        logger.error(f"Error creating/loading model: {e}")
        raise

    # Move model to device
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    return model, args


def predict(model, marker_values, coverage, sample_ids, output_dir=None, device='cpu'):
    """
    Predict using a trained model on a dataset and save the results to the output dir
    
    Args:
        model: Trained model
        marker_values: Marker values tensor
        coverage: Coverage tensor
        sample_ids: Sample IDs list
        output_dir: Directory to save results
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

    all_preds = []
    all_lower_ci = []
    all_upper_ci = []
    all_uncertainties = []

    # Process predictions
    with torch.no_grad():
        marker_values = marker_values.to(device)
        coverage = coverage.to(device)
        
       
        # Standard model - handle different return formats
        model_output = model(marker_values, coverage)
        
        
        # Standard model returns (concentration, uncertainty, attention_weights)
        mu = model_output[0]  # concentration
        phi = model_output[1]  # uncertainty
        
        estimate, ci, scaled_uncertainty = model.get_estimate_and_ci(mu, phi)
        
        # Store predictions
        all_preds.append(estimate.cpu().numpy())
        all_lower_ci.append(ci[:, 0:1].cpu().numpy())
        all_upper_ci.append(ci[:, 1:2].cpu().numpy())
        all_uncertainties.append(scaled_uncertainty.cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_lower_ci = np.concatenate(all_lower_ci)
    all_upper_ci = np.concatenate(all_upper_ci)
    all_uncertainties = np.concatenate(all_uncertainties)
    
    # Save results
    predictions_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "estimated": all_preds.flatten(),
            "lower_ci": all_lower_ci.flatten(),
            "upper_ci": all_upper_ci.flatten(),
            "uncertainty": all_uncertainties.flatten(),
        }
    )

    predictions_file = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to {predictions_file}")
    
    # Save summary statistics
    summary = {
        'num_samples': len(sample_ids),
        'mean_prediction': float(all_preds.mean()),
        'std_prediction': float(all_preds.std()),
        'mean_uncertainty': float(all_uncertainties.mean()),
        'predictions_above_0.001': int((all_preds > 0.001).sum()),
        'predictions_above_0.01': int((all_preds > 0.01).sum()),
        'predictions_above_0.05': int((all_preds > 0.05).sum()),
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_file}")
    
    return predictions_df

def run_predict(model_dir, input_dir, output_dir=None, device=None):
    """
    Load a model and evaluate it on data from the input directory
    
    Args:
        model_dir: Directory containing the trained model
        input_dir: Directory containing the data to evaluate
        output_dir: Directory to save evaluation results
        device: Device to run evaluation on (defaults to CUDA if available)
        
    Returns:
        results: DataFrame of prediction results
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
        
        # Load dataset
        logger.info(f"Loading data from {input_dir}")
        atlas_path = args.get('atlas_path', None)
        if atlas_path:
            logger.info(f"Using atlas from training: {atlas_path}")
        else:
            logger.warning("No atlas file specified in model args")
        
        target_cell_type = args.get('target_cell_type', None)
        target_cell_idx = args.get('target_cell_idx', None)
        excluded_markers = args.get('excluded_markers', [])
        
        # Parse excluded markers if it's a string
        if isinstance(excluded_markers, str) and excluded_markers:
            excluded_markers = [int(x.strip()) for x in excluded_markers.split(',')]
        
        # Load dataset
        marker_values, coverage, sample_ids = prepare_data_for_predict(
            data_dir=input_dir,
            atlas_path=atlas_path,
            target_cell_type=target_cell_type,
            excluded_markers=excluded_markers
        )
        
        # Make predictions
        results_df = predict(model, marker_values, coverage, sample_ids, output_dir, device)        
        logger.info("Evaluation completed successfully")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# python -m deep_conv.detect.predict \
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/oac_unbias/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/AB/cfDNA/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/oac_unbias/

# python -m deep_conv.detect.predict \
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/oac_unbias/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/CD/cfDNA/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/CD/cfDNA/oac_unbias/

if __name__ == '__main__':
    args = parse_args()
    run_predict(args.model_dir, args.input_dir, args.output_dir, args.device)