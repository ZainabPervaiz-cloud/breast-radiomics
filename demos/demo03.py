"""
Python port of demo03.m
ROI-detection: Squared ROI, Multi-ROI, and Retroareolar region detection.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc.getDicomFiles import getDicomFiles
from misc.getinfo import getinfo
from misc.ffdmRead import ffdmRead
from segmentation.segBreast import segBreast
from misc.sqmax import sqmax
from misc.rcenters import rcenters
from misc.xySampling import xySampling
from mapping.refpoints import refpoints
from mapping.stmap import stmap
from mapping.st2mask import st2mask
from misc.showseg import showseg

def main(patient_folder, output_dir=None):
    print(f"Reading DICOMs from: {patient_folder}")
    dicom_paths = getDicomFiles(patient_folder)
    
    if not dicom_paths:
        print(f"No DICOM files found in {patient_folder}")
        return

    out_folder = output_dir if output_dir else os.getcwd()
    os.makedirs(out_folder, exist_ok=True)

    for i, impath in enumerate(dicom_paths):
        print(f"\nProcessing ROI-detection for patient view {i+1}/{len(dicom_paths)}: {impath}")
        
        info = getinfo(impath)
        im_norm, _ = ffdmRead(impath, info)
        
        # Scale to 0.4mm for analysis
        scale = info['psize'] / 0.4
        im_scaled = resize(im_norm, (int(round(im_norm.shape[0]*scale)), int(round(im_norm.shape[1]*scale))), preserve_range=True)
        
        # 1. Breast segmentation
        mask, contour, cwall = segBreast(im_scaled, info['ismlo'])
        
        # 2. Maximum squared ROI
        mask_SQ, _ = sqmax(mask)
        
        # 3. Multiple-ROIs of size 32x32 pixels
        x, y = rcenters(64, 32, mask)
        _, _, mask_multi = xySampling(im_scaled, x, y, 32)
        
        # 4. Retroareolar region
        rpts = refpoints(contour, cwall, info['ismlo'])
        mapp = stmap(contour, rpts)
        mask_RA = st2mask(mapp, [0.1, 0.5], [0.1, 0.9])
        
        # Visualization (1x4 subplots as in demo03.m)
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        # Plot 1: Full breast contour
        axes[0].imshow(showseg(im_scaled, mask, alpha=0, color=[0, 255, 0]))
        axes[0].set_title(f'Full breast - View {i+1}')
        
        # Plot 2: Squared ROI
        axes[1].imshow(showseg(im_scaled, mask_SQ, alpha=0, color=[0, 255, 0]))
        axes[1].set_title('Squared ROI')
        
        # Plot 3: Multi-ROI
        axes[2].imshow(showseg(im_scaled, mask_multi > 0, alpha=0, color=[0, 255, 0]))
        axes[2].set_title('Multi-ROI')
        
        # Plot 4: Retroareolar region (ST)
        axes[3].imshow(showseg(im_scaled, mask_RA, alpha=0, color=[255, 0, 0]))
        axes[3].set_title('RA-region (ST)')
        
        save_path = os.path.join(out_folder, f"demo03_view_{i+1}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved result to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Demo03: ROI Detection")
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    parser.add_argument("--folder", type=str, default=default_path, help="Path to patient folder")
    args = parser.parse_args()
    main(args.folder)
