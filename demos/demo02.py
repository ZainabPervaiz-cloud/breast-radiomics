# Demo on ST-mapping
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc.getDicomFiles import getDicomFiles
from misc.getinfo import getinfo
from misc.ffdmRead import ffdmRead
from segmentation.segBreast import segBreast
from mapping.refpoints import refpoints
from mapping.stmap import stmap
from mapping.mapping import mapping
from mapping.imapping import imapping
from mapping.showmap import showmap

def main(patient_folder, output_dir=None):
    print(f"Reading DICOMs from: {patient_folder}")
    dicom_paths = getDicomFiles(patient_folder)
    
    if not dicom_paths:
        print(f"No DICOM files found in {patient_folder}")
        return

    out_folder = output_dir if output_dir else os.getcwd()
    os.makedirs(out_folder, exist_ok=True)
    
    for i, impath in enumerate(dicom_paths):
        print(f"Processing Demo02 Mapping for file {i+1}/{len(dicom_paths)}: {impath}")
        
        info = getinfo(impath)
        im_norm, _ = ffdmRead(impath, info)
        
        from skimage.transform import resize
        scale = info.get('psize', 0.1) / 0.4
        im_scaled = resize(im_norm, (int(im_norm.shape[0]*scale), int(im_norm.shape[1]*scale)), preserve_range=True)
        
        mask, contour, cwall = segBreast(im_scaled, info.get('ismlo', False))
        rpts = refpoints(contour, cwall, info.get('ismlo', False))
        mapp = stmap(contour, rpts)
        
        # Start visualization (1x3 subplots as in MATLAB)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Subplot 1: Mapping Grid
        plt.sca(axes[0])
        showmap(im_scaled, mapp)
        axes[0].set_title(f"Mapping - View {i+1}")

        # Subplot 2: Forward ST mapping
        st_im, _, _ = mapping(im_scaled, mapp)
        axes[1].imshow(st_im, cmap='gray', aspect='auto')
        axes[1].set_title('ST map')

        # Subplot 3: Inverse ST mapping
        im_rec, is_valid = imapping(st_im, mapp)
        from misc.showseg import showseg
        axes[2].imshow(showseg(im_rec, is_valid))
        axes[2].set_title('Inverse ST map')

        save_path = os.path.join(out_folder, f"demo02_view_{i+1}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    print("Finished Phase 3 verification.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Demo02: ST Mapping")
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    parser.add_argument("--folder", type=str, default=default_path, help="Path to patient folder")
    args = parser.parse_args()
    main(args.folder)
