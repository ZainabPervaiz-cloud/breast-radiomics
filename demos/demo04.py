"""
Python port of demo04.m
Feature extraction for FFDM images.
"""
import os
import sys
import argparse
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc.getDicomFiles import getDicomFiles
from misc.getinfo import getinfo
from misc.ffdmRead import ffdmRead
from segmentation.segBreast import segBreast
from features.xfeatures import xfeatures
from features.ffdmFeatures import ffdmFeatures
from skimage.transform import resize

def main(patient_folder, output_dir=None):
    print(f"Reading DICOMs from: {patient_folder}")
    dicom_paths = getDicomFiles(patient_folder)
    
    if not dicom_paths:
        print(f"No DICOM files found in {patient_folder}")
        return

    out_folder = output_dir if output_dir else os.getcwd()
    os.makedirs(out_folder, exist_ok=True)

    all_features = []
    patient_id = "UnknownPatient"

    for i, impath in enumerate(dicom_paths):
        print(f"\nExtracting features for patient view {i+1}/{len(dicom_paths)}: {impath}")
        
        info = getinfo(impath)
        if i == 0:
            patient_id = info.get('patient_id', 'UnknownPatient').replace('^', '_')
            
        im_norm, _ = ffdmRead(impath, info)
        
        from skimage.transform import resize
        scale = info['psize'] / 0.4
        im_scaled = resize(im_norm, (int(round(im_norm.shape[0]*scale)), int(round(im_norm.shape[1]*scale))), preserve_range=True)
        
        # 1. Breast segmentation
        mask, _, _ = segBreast(im_scaled, info['ismlo'])
        
        # 2. High-level feature extraction (Clinical Grade)
        norm = ['none', 'zscore']
        samp = ['full', 'multi']
        scal = ['1.0']
        res = 0.1
        
        start_time = time.time()
        xf, f_info = ffdmFeatures(impath, res, scal, norm, samp)
        duration = time.time() - start_time
        
        side = info.get('side', 'U')
        view = info.get('view', 'U')
        v_label = f"{side}-{view}"
        
        print(f"  -> {len(f_info['fnames'])} advanced features in {duration:.2f}s for {v_label}")
        
        # Store for consolidated CSV
        for name, value in zip(f_info['fnames'], xf):
            all_features.append([v_label, name, value])

    # Save unique consolidated CSV for the patient
    import csv
    feat_path = os.path.join(out_folder, f"{patient_id}_features.csv")
    with open(feat_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['View', 'Feature', 'Value'])
        writer.writerows(all_features)
    
    print(f"\n[DONE] Consolidated features saved to: {feat_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Demo04: Feature Extraction")
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    parser.add_argument("--folder", type=str, default=default_path, help="Path to patient folder")
    args = parser.parse_args()
    main(args.folder)
