"""
Python port of demo05.m
Breast density segmentation demo (Morphology and LIBRA fallback).
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
from density.mseg import mseg
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
        print(f"\nDensity segmentation for patient view {i+1}/{len(dicom_paths)}: {impath}")
        
        info = getinfo(impath)
        im_norm, _ = ffdmRead(impath, info)
        
        # Scale to 0.4mm
        scale = info['psize'] / 0.4
        im_scaled = resize(im_norm, (int(round(im_norm.shape[0]*scale)), int(round(im_norm.shape[1]*scale))), preserve_range=True)
        
        # 1. Breast Mask
        mask, _, _ = segBreast(im_scaled, info['ismlo'])
        
        # 2. Morphology-based density estimation
        seg_morph, pd_morph = mseg(im_scaled, mask, 0.4)
        
        # Visualization (MATLAB Parity)
        fig = plt.figure(figsize=(10, 10))
        # First overlay breast boundary in green (alpha=0 for contour)
        from misc.showseg import showseg
        im_border = showseg(im_scaled, mask, alpha=0, color='g')
        # Overlay dense tissue in yellow
        im_final = showseg(im_border, seg_morph, alpha=0.25, color=[255, 255, 0])
        
        plt.imshow(im_final)
        plt.title(f"Density Morph - View {i+1}")
        
        save_path = os.path.join(out_folder, f"demo05_view_{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved result to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Demo05: Density Segmentation")
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    parser.add_argument("--folder", type=str, default=default_path, help="Path to patient folder")
    args = parser.parse_args()
    main(args.folder)
