import argparse
import logging
import subprocess
from datetime import datetime
import json
from typing import Dict
from pathlib import Path
import random
import torch 
import numpy as np
from deep_conv.deconvolution.deconv import train


def get_git_info() -> Dict[str, str]:
    """Get git repository information"""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).strip().decode('utf-8')
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).strip().decode('utf-8')
        
        status = subprocess.check_output(
            ['git', 'status', '--porcelain']
        ).strip().decode('utf-8')
        
        return {
            'commit': commit_hash,
            'branch': branch,
            'clean': len(status) == 0
        }
    except subprocess.CalledProcessError:
        return {'commit': 'unknown', 'branch': 'unknown', 'clean': False}


def setup_experiment_dir(base_dir: str, model_name: str) -> Path:
    """Setup experiment directory with timestamp and git info"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    git_info = get_git_info()
    
    exp_name = f"{model_name}_{timestamp}_{git_info['commit']}"
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save git info
    with open(exp_dir / 'git_info.json', 'w') as f:
        json.dump(git_info, f, indent=2)
    
    return exp_dir


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# python -m deep_conv.deconvolution.train --batch_size 32 --n_train 100000 --n_val 20000 --atlas_path /mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed --model_save_path /users/zetzioni/sharedscratch/deepconv/src/deep_conv/saved_models/
def main():
    parser = argparse.ArgumentParser(description="DeepConv")
    parser.add_argument("--batch_size",type=int, help="training batch size",required=False, default=32)
    parser.add_argument("--n_train",type=int, help="training set size",required=False, default=100000)
    parser.add_argument("--n_val",type=int, help="validation set size",required=False, default=20000)
    parser.add_argument("--atlas_path",help="path to atlas",required=True)
    parser.add_argument("--model_save_path",help="path to save the trained model",required=True)
    parser.add_argument("--model_name",help="model name",required=False, default="dual_encoder")
    inputs = parser.parse_args()   

    set_seed()
    exp_dir = setup_experiment_dir(inputs.model_save_path, inputs.model_name)
    logging.info(f"Experiment directory: {exp_dir}")

    # Save experiment config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(inputs), f, indent=2)

    train(
        atlas_path=inputs.atlas_path,
        batch_size=inputs.batch_size,
        model_save_path=str(exp_dir),
        n_train=inputs.n_train,
        n_val=inputs.n_val,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        raise