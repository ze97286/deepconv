import os
import torch
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px

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
        'num_markers': 349,
        'feature_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'dropout_rate': 0.2,
        'min_reliable_coverage': 3.0
    }
    
    for key, default_value in defaults.items():
        if key not in args:
            args[key] = default_value

    print(args)

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

        # Try loading with strict mode first
        try:
            model.load_state_dict(model_state, strict=True)
            logger.info("Model loaded with strict=True")
        except Exception as e:
            # If strict loading fails, try non-strict loading
            logger.warning(f"Strict loading failed: {e}")
            logger.warning("Attempting non-strict loading...")
            model.load_state_dict(model_state, strict=False)
            logger.info("Model loaded with strict=False")

        # Manually check for and apply clinical_threshold
        # First, check if the buffer exists in the model state
        clinical_threshold_key = 'clinical_threshold'
        if clinical_threshold_key in model_state:
            with torch.no_grad():
                # Load the threshold as a tensor
                threshold_value = model_state[clinical_threshold_key].item()
                model.clinical_threshold.fill_(threshold_value)
                logger.info(f"Loaded clinical threshold from state dict: {threshold_value}")
        # If not in the state dict, check if it's elsewhere in the checkpoint
        elif 'clinical_threshold' in checkpoint:
            threshold_value = checkpoint['clinical_threshold']
            # Handle different types of threshold values
            if isinstance(threshold_value, (torch.Tensor, np.ndarray)):
                threshold_value = float(threshold_value.item() if hasattr(threshold_value, 'item') else threshold_value)
            elif isinstance(threshold_value, (float, int)):
                threshold_value = float(threshold_value)
            else:
                logger.warning(f"Unexpected type for clinical_threshold: {type(threshold_value)}")
                threshold_value = 0.001  # Default value
            
            with torch.no_grad():
                model.clinical_threshold.fill_(threshold_value)
                logger.info(f"Loaded clinical threshold from checkpoint: {threshold_value}")
        else:
            logger.warning("Clinical threshold not found in checkpoint. Using default value of 0.001.")

        # Similarly check for calibration
        calibration_key = 'calibration'
        if calibration_key in model_state:
            with torch.no_grad():
                calibration_value = model_state[calibration_key].item()
                model.calibration.fill_(calibration_value)
                logger.info(f"Loaded calibration factor from state dict: {calibration_value}")
        elif 'calibration' in checkpoint:
            calibration_data = checkpoint['calibration']
            if isinstance(calibration_data, dict) and 'calibration_factor' in calibration_data:
                calibration_value = calibration_data['calibration_factor']
            elif isinstance(calibration_data, (torch.Tensor, np.ndarray)):
                calibration_value = float(calibration_data.item() if hasattr(calibration_data, 'item') else calibration_data)
            elif isinstance(calibration_data, (float, int)):
                calibration_value = float(calibration_data)
            else:
                logger.warning(f"Unexpected type for calibration: {type(calibration_data)}")
                calibration_value = 1.0  # Default value
            
            with torch.no_grad():
                model.calibration.fill_(calibration_value)
                logger.info(f"Loaded calibration factor from checkpoint: {calibration_value}")
        else:
            logger.warning("Calibration factor not found in checkpoint. Using default value of 1.0.")

    except Exception as e:
        logger.error(f"Error creating/loading model: {e}")
        raise

    # Move model to device
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    logger.info(f"Clinical threshold: {model.clinical_threshold.item()}")
    logger.info(f"Calibration factor: {model.calibration.item()}")
    
    return model, args

def predict(model, marker_values, coverage, sample_ids, output_dir=None, device='cpu', ichor_cna_path=None):
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
        mu, phi, _, _ = model(marker_values, coverage)  
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
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/oac_detector_without_cna_headless/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/AB/cfDNA/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/oac_detector_without_cna_headless/

# python -m deep_conv.detect.predict \
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/oac_detector_cna_corrected_headless/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/AB/cfDNA/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/oac_detector_cna_corrected_headless/

# python -m deep_conv.detect.predict \
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/oac_detector_without_cna_headless/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/CD/cfDNA/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/CD/cfDNA/oac_detector_without_cna_headless/

# python -m deep_conv.detect.predict \
# --model_dir /users/zetzioni/sharedscratch/loyfer_atlas/saved_models/single_cell/oac_detector_cna_corrected_headless/ \
# --input_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/atlas_oac.blood+gi+tum.l4/CD/cfDNA/ \
# --output_dir /users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/CD/cfDNA/oac_detector_cna_corrected_headless/


def plot_timepoint(df, output_path, name, title):
    fig = go.Figure()
    # Add estimated values as scatter plot with error bars
    fig.add_trace(go.Scatter(
        x=df['sample_id'],
        y=df['estimated'],
        mode='markers',
        name='Estimated',
        error_y=dict(
            type='data',
            symmetric=False,
            array=df['upper_ci'] - df['estimated'],
            arrayminus=df['estimated'] - df['lower_ci']
        ),
        marker=dict(size=10, color='blue'),
        text=[f"Uncertainty: {u}" for u in df['uncertainty']],
        hovertemplate=
            "Sample ID: %{x}<br>" +
            "Estimated: %{y:.3f}<br>" +
            "%{text}<br>" +
            "CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
        customdata=df[['lower_ci', 'upper_ci']].values
    ))
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Sample ID",
        yaxis_title="Estimated Value",
        template="plotly_white"
    )
    fig.write_html(f"{output_path}/{name}_model_estimates.html")
    fig.write_image(f"{output_path}/{name}_model_estimates.png", scale=2)


def scatter_plot_vs_ichor_cna(df_merged, output_path, col_name, name, title):
    # Create scatter plot
    fig = px.scatter(
        df_merged,
        x='estimate_method1',
        y='estimate_method2',
        text='sample',
        labels={
            'estimate_method1': 'ichorCNA',
            'estimate_method2': col_name
        },
        title=title
    )
    # Optional: Add a y = x reference line
    fig.add_shape(
        type='line',
        x0=min(df_merged['estimate_method1'].min(), df_merged['estimate_method2'].min()),
        y0=min(df_merged['estimate_method1'].min(), df_merged['estimate_method2'].min()),
        x1=max(df_merged['estimate_method1'].max(), df_merged['estimate_method2'].max()),
        y1=max(df_merged['estimate_method1'].max(), df_merged['estimate_method2'].max()),
        line=dict(color='gray', dash='dash')
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title='ichorCNA',
        yaxis_title=col_name,
        template='plotly_white'
    )
    fig.write_html(f"{output_path}/{name}_scatter_plot.html")
    fig.write_image(f"{output_path}/{name}_scatter_plot.png", scale=2)


def plot_vs_nnls_vs_ichorcna(merged, output_path, name, title):
    fig = px.bar(
        merged,
        x='sample',
        y='estimated',
        color='method',
        barmode='group',
        text='estimated',
        title=title
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        yaxis_title='Estimate',
        xaxis_title='Sample',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        template='plotly_white'
    )
    fig.write_html(f"{output_path}/{name}_dc_vs_nnls_vs_ichorcna.html")
    fig.write_image(f"{output_path}/{name}_dc_vs_nnls_vs_ichorcna.png", scale=2)

def plot_results(model_name="oac_detector"):
    import pandas as pd
    import os

    # A/B model estimates with CI and uncertainty
    out_base_dir = f"/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/{model_name}/"
    ab_df =  pd.read_csv(out_base_dir+"/predictions.csv")
    os.makedirs(out_base_dir+"model_estimates", exist_ok=True)
    plot_timepoint(ab_df, out_base_dir+"model_estimates","ab", "Estimated OAC Content with Confidence Intervals for A/B cohort")

    # A/B model estimates with CI and uncertainty at timepoint
    ab_df['timepoint'] = ab_df['sample_id'].map(lambda x: x.split("_")[1])
    for tp in ab_df['timepoint'].unique():
        os.makedirs(out_base_dir+f"model_estimates/{tp}", exist_ok=True)
        df = ab_df[ab_df.timepoint == tp]
        plot_timepoint(df, out_base_dir+f"model_estimates/{tp}",f"ab_{tp}", f"Estimated OAC Content with Confidence Intervals for A/B cohort at timepoint {tp}")

    ab_df['method'] = "deepconv"
    ab_df['sample'] = ab_df['sample_id'].map(lambda x: x.split("_plasma")[0])
    ab_dc = ab_df[['sample', 'timepoint', 'estimated','method']]
    nnls_df = pd.read_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/nnls/nnls_ab_cfDNA_deconvolution.csv", sep="\t")
    nnls_df['estimated'] = nnls_df['OAC']
    nnls_df['method'] = 'nnls'
    nnls_df['timepoint'] = nnls_df['sample'].map(lambda x: x.split("_")[1])
    ab_nnls = nnls_df[['sample', 'timepoint', 'estimated','method']]
    ichorcna_ab = pd.read_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/ab_ichorcna_cfdna.csv", sep="\t")
    ichorcna_ab.columns=['sample', 'tf','ploidy']
    ichorcna_ab['timepoint'] = ichorcna_ab['sample'].map(lambda x: x.split("_")[1])  
    ichorcna_ab['method'] = 'ichorcna'
    ichorcna_ab['estimated'] = ichorcna_ab['tf']
    ichorcna_ab = ichorcna_ab[['sample', 'timepoint', 'estimated','method']]
    df_all = pd.concat([ab_dc,ab_nnls, ichorcna_ab])

    # A/B deepconv vs nnls vs ichorCNA
    os.makedirs(out_base_dir+"benchmarks", exist_ok=True)
    plot_vs_nnls_vs_ichorcna(df_all,out_base_dir+"benchmarks", "ab","Estimate Comparison Across Methods for A/B cohort")

    # A/B deepconv vs nnls vs ichorCNA at timepoint
    for tp in df_all['timepoint'].unique():
        os.makedirs(out_base_dir+f"benchmarks/{tp}", exist_ok=True)
        df = df_all[df_all.timepoint == tp]
        plot_vs_nnls_vs_ichorcna(
            df,
            out_base_dir+f"benchmarks/{tp}",
            f"ab_{tp}",
            f"Estimate Comparison Across Methods for A/B cohort at timepoint {tp}",
        )

    # A/B deepconv vs ichorCNA scatter plot
    df1 = ichorcna_ab.rename(columns={'estimated': 'estimate_method1'})
    df2 = ab_dc.rename(columns={'estimated': 'estimate_method2'})
    df_merged = pd.merge(df1, df2, on='sample')
    scatter_plot_vs_ichor_cna(
        df_merged,
        out_base_dir+"benchmarks",
        "deepconv",
        "ab_deepconv_vs_ichorcna",
        "Deep conv vs ichorCNA in A/B cohort",
    )
    for tp in df_merged['timepoint_x'].unique():
        df = df_merged[df_merged.timepoint_x == tp]
        scatter_plot_vs_ichor_cna(
            df,
            out_base_dir+f"benchmarks/{tp}",
            "deepconv",
            f"ab_deepconv_vs_ichorcna_{tp}",
            f"Deep conv vs ichorCNA in A/B cohort at timepoint {tp}",
        )

    # A/B nnls vs ichorCNA scatter plot
    df2 = ab_nnls.rename(columns={'estimated': 'estimate_method2'})
    df_merged = pd.merge(df1, df2, on='sample')
    scatter_plot_vs_ichor_cna(
        df_merged,
        out_base_dir+"benchmarks",
        "nnls",
        "ab_nnls_vs_ichorcna",
        "NNLS vs ichorCNA in A/B cohort",
    )
    for tp in df_merged['timepoint_x'].unique():
        df = df_merged[df_merged.timepoint_x == tp]
        scatter_plot_vs_ichor_cna(
            df,
            out_base_dir+f"benchmarks/{tp}",
            "nnls",
            f"ab_nnls_vs_ichorcna_{tp}",
            f"NNLS vs ichorCNA in A/B cohort at timepoint {tp}",
        )

    # C/D
    out_base_dir = f"/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/CD/cfDNA/{model_name}/"
    cd_df =  pd.read_csv(out_base_dir+"/predictions.csv")
    # C/D model estimates with CI and uncertainty
    os.makedirs(out_base_dir+"model_estimates", exist_ok=True)
    plot_timepoint(cd_df, out_base_dir+"model_estimates","cd", "Estimated OAC Content with Confidence Intervals for C/D cohort")

    cd_df['timepoint'] = cd_df['sample_id'].map(lambda x: x.split("-")[-1] if "SCAN" not in x and "GI" not in x else "Ctrl")
    cd_df["timepoint"].replace("ScrBsI", "ScrBsl", inplace=True)
    for tp in cd_df['timepoint'].unique():
        os.makedirs(out_base_dir+f"model_estimates/{tp}", exist_ok=True)
        df = cd_df[cd_df.timepoint == tp]
        plot_timepoint(df, out_base_dir+f"model_estimates/{tp}",f"cd_{tp}", "Estimated OAC Content with Confidence Intervals for C/D cohort")

    cd_df['method'] = "deepconv"
    cd_df['sample'] = cd_df['sample_id'].map(lambda x: x.split("_plasma")[0])
    cd_dc = cd_df[['sample', 'timepoint', 'estimated','method']]
    nnls_df = pd.read_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/CD/cfDNA/nnls/nnls_cd_cfDNA_deconvolution.csv", sep="\t")
    nnls_df['estimated'] = nnls_df['OAC']
    nnls_df['method'] = 'nnls'
    nnls_df['timepoint'] = nnls_df['sample'].map(lambda x: x.split("-")[-1] if "SCAN" not in x and "GI" not in x else "Ctrl")
    nnls_df["timepoint"].replace("ScrBsI", "ScrBsl", inplace=True)
    cd_nnls = nnls_df[['sample', 'timepoint', 'estimated','method']]

    ichorcna_cd = pd.read_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/CD/cfDNA/cd_ichorcna_cfdna.csv", sep="\t")
    ichorcna_cd.columns=['sample', 'tf','ploidy']
    ichorcna_cd['timepoint'] = ichorcna_cd['sample'].map(lambda x: x.split("-")[-1] if "SCAN" not in x and "GI" not in x else "Ctrl")
    ichorcna_cd['timepoint'].replace("ScrBsI", "ScrBsl", inplace=True)
    ichorcna_cd['method'] = 'ichorcna'
    ichorcna_cd['estimated'] = ichorcna_cd['tf']
    ichorcna_cd = ichorcna_cd[['sample', 'timepoint', 'estimated','method']]
    df_all = pd.concat([cd_dc,cd_nnls, ichorcna_cd])

    # C/D deepconv vs nnls vs ichorCNA
    os.makedirs(out_base_dir+"benchmarks", exist_ok=True)
    plot_vs_nnls_vs_ichorcna(df_all,out_base_dir+"benchmarks", "cd","Estimate Comparison Across Methods for C/D cohort")

    # C/D deepconv vs nnls vs ichorCNA at timepoint
    for tp in df_all['timepoint'].unique():
        os.makedirs(out_base_dir+f"benchmarks/{tp}", exist_ok=True)
        df = df_all[df_all.timepoint == tp]
        plot_vs_nnls_vs_ichorcna(
            df,
            out_base_dir+f"benchmarks/{tp}",
            f"cd_{tp}",
            f"Estimate Comparison Across Methods for C/D cohort at timepoint {tp}",
        )

    # C/D deepconv vs ichorCNA scatter plot
    df1 = ichorcna_cd.rename(columns={'estimated': 'estimate_method1'})
    df2 = cd_dc.rename(columns={'estimated': 'estimate_method2'})
    df_merged = pd.merge(df1, df2, on='sample')
    scatter_plot_vs_ichor_cna(
        df_merged,
        out_base_dir + "benchmarks",
        "deepconv",
        "cd_deepconv_vs_ichorcna",
        "DeepConv vs ichorCNA in C/D cohort",
    )
    for tp in df_merged['timepoint_x'].unique():
        df = df_merged[df_merged.timepoint_x == tp]
        scatter_plot_vs_ichor_cna(
            df,
            out_base_dir+f"benchmarks/{tp}",
            "deepconv",
            f"cd_deepconv_vs_ichorcna_{tp}",
            f"DeepConv vs ichorCNA in C/D cohort at timepoint {tp}",
        )

    # C/D nnls vs ichorCNA scatter plot
    df2 = cd_nnls.rename(columns={'estimated': 'estimate_method2'})
    df_merged = pd.merge(df1, df2, on='sample')
    scatter_plot_vs_ichor_cna(
        df_merged,
        out_base_dir + "benchmarks",
        "nnls",
        "cd_nnls_vs_ichorcna",
        "NNLS vs ichorCNA in C/D cohort",
    )
    for tp in df_merged['timepoint_x'].unique():
        df = df_merged[df_merged.timepoint_x == tp]
        scatter_plot_vs_ichor_cna(
            df,
            out_base_dir+f"benchmarks/{tp}",
            "nnls",
            f"cd_nnls_vs_ichorcna_{tp}",
            f"NNLS vs ichorCNA in C/D cohort at timepoint {tp}",
        )


if __name__ == '__main__':
    args = parse_args()
    run_predict(args.model_dir, args.input_dir, args.output_dir, args.device)
