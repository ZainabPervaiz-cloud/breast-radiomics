# Demo on Breast Segmentation
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

def main(patient_folder, output_dir=None):
    print(f"Reading DICOMs from: {patient_folder}")
    dicom_paths = getDicomFiles(patient_folder)
    
    if not dicom_paths:
        print(f"No DICOM files found in {patient_folder}")
        return

    out_folder = output_dir if output_dir else os.getcwd()
    os.makedirs(out_folder, exist_ok=True)
    
    fig_idx = 0
    for i, impath in enumerate(dicom_paths):
        info = getinfo(impath)
        im_norm, _ = ffdmRead(impath, info)
        
        side = info.get('side', 'U')
        view = info.get('view', 'U')
        print(f"Processing Demo01 for {side}-{view} ({info.get('psize', 0.1)} mm/pixel)...")
        
        from skimage.transform import resize
        scale = info.get('psize', 0.1) / 0.4
        im_scaled = resize(im_norm, (int(im_norm.shape[0]*scale), int(im_norm.shape[1]*scale)), preserve_range=True)
        
        mask, contour, cwall = segBreast(im_scaled, info.get('ismlo', False))
        rpts = refpoints(contour, cwall, info.get('ismlo', False))
        
        # MATLAB Parity: showseg(fliplr(im), fliplr(mask))
        im_disp = np.fliplr(im_scaled)
        mask_disp = np.fliplr(mask)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        ax.imshow(im_disp, cmap='gray')
        
        # Draw contour on flipped mask
        ax.contour(mask_disp, [0.5], colors='g', linewidths=1)
        ax.set_title(f'Demo 01: Breast Segmentation ({side}-{view})')
        
        # Plot nipple (detect point from contour/rpts and flip x if needed)
        if rpts.get('p0'):
            # Calculate flipped x coordinate
            nx = im_scaled.shape[1] - rpts['p0']['x']
            ny = rpts['p0']['y']
            ax.plot(nx, ny, 's', markersize=10, color='r', label='Nipple')
            ax.legend()
        
        save_path = os.path.join(out_folder, f"demo01_{side}_{view}.png")
        plt.savefig(save_path)
        plt.close(fig)
        fig_idx += 1
    
    print(f"Finished Demo 01. Total figures saved to {out_folder}: {fig_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Demo01: Breast Segmentation")
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    parser.add_argument("--folder", type=str, default=default_path, help="Path to patient folder")
    args = parser.parse_args()
    main(args.folder)
