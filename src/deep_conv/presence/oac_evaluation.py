from pathlib import Path

from deep_conv.presence.presence import *
from deep_conv.benchmark.benchmark_utils import *

def train_and_evaluate(model_name, target_cell_type_name, use_loyfer=True):
    atlas_dir = "atlas"
    if use_loyfer:
        atlas_dir = "loyfer_atlas"
    atlas_path = f"/users/zetzioni/sharedscratch/{atlas_dir}/atlas/atlas_cov_oac.blood+gi+tum.l4.bed"
    train_pat_dir = f"/users/zetzioni/sharedscratch/{atlas_dir}/training/oac.blood+gi+tum.l4/train"
    eval_pat_dir = f"/users/zetzioni/sharedscratch/{atlas_dir}/training/oac.blood+gi+tum.l4/eval"

    threads = 32
    output_path = Path(f"/users/zetzioni/sharedscratch/{atlas_dir}/saved_models/{model_name}/")
    train_and_eval(atlas_path=atlas_path, 
                           train_pat_dir=train_pat_dir, 
                           eval_pat_dir=eval_pat_dir, 
                           threads=threads,
                           output_path=output_path, 
                           target_cell_type_name=target_cell_type_name, 
                           use_loyfer=use_loyfer)
