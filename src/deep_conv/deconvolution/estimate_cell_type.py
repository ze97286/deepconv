import logging 
import argparse

from deep_conv.deconvolution.predict import predict 
from deep_conv.deconvolution.preprocess_pats import pats_to_homog

# python -m deep_conv.deconvolution.estimate_cell_type \
# --model_path /users/zetzioni/sharedscratch/deconvolution_model.pt \
# --cell_type CD4-T-cells \
# --atlas_path /mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/TAPS_Atlas/Atlas_dmr_by_read.blood+gi+tum.U100.l4.bed \
# --wgbs_tools_exec_path /users/zetzioni/sharedscratch/wgbs_tools/wgbstools \
# --pats_path /mnt/lustre/users/bschuster/OAC_Trial_TAPS_Tissue/Data/Benchmark/pat/dmr_by_read.blood+gi+tum.U100/Song/mixed/CD4 \
# --output_path /users/zetzioni/sharedscratch/cd4 
def main():
    parser = argparse.ArgumentParser(description="DeepConv")
    parser.add_argument("--model_path",help="path to the saved deep learning model",required=True)
    parser.add_argument("--cell_type",help="cell type to estimate",required=True)
    parser.add_argument("--pats_path",help="path to pat zip files",required=True)
    parser.add_argument("--atlas_path",help="path to atlas",required=True)
    parser.add_argument("--output_path",help="path to save output plots",required=True)
    parser.add_argument("--wgbs_tools_exec_path",help="path to wgbs_tools executable",required=True)
    inputs = parser.parse_args()    

    marker_read_proportions, counts = pats_to_homog(
            atlas_path=inputs.atlas_path,
            pats_path=inputs.pats_path,
            wgbs_tools_exec_path=inputs.wgbs_tools_exec_path,
    )

    predict(
        model_path=inputs.model_path,
        marker_read_proportions=marker_read_proportions,
        marker_read_coverage=counts,
        cell_type=inputs.cell_type,
        atlas_path=inputs.atlas_path,
        output_path=inputs.output_path,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        raise