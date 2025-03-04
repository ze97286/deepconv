from pathlib import Path

from deep_conv.presence.presence import *
from deep_conv.benchmark.benchmark_utils import *

def train_and_evaluate(model_name, use_low_coverage:bool=False, use_high_coverage:bool=True):
    atlas_path = "/users/zetzioni/sharedscratch/atlas/atlas/atlas_oac.blood+gi+tum.l4.bed"
    train_pat_dir = "/users/zetzioni/sharedscratch/atlas/training/oac.blood+gi+tum.l4/train"
    eval_pat_dir = "/users/zetzioni/sharedscratch/atlas/training/oac.blood+gi+tum.l4/eval"
    if use_low_coverage:
        train_pat_dir+="_low/"
        eval_pat_dir+="_low/"
    elif use_high_coverage:
        train_pat_dir+="_high/"
        eval_pat_dir+="_high/"
    else:
        train_pat_dir+="/"
        eval_pat_dir+="/"

    threads = 32
    output_path = Path("/users/zetzioni/sharedscratch/atlas/saved_models/"+model_name+"/")
    train_and_eval(atlas_path=atlas_path, 
                           train_pat_dir=train_pat_dir, 
                           eval_pat_dir=eval_pat_dir, 
                           threads=threads,
                           output_path=output_path)
