"""
Python port of demo06.m
Area-Based Percent Density (PD) Calculation and BI-RADS Categorization.
Includes hierarchical aggregation logic (View -> Breast -> Patient) as requested.
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

def categorize_bi_rads(pd_val):
    """Categorize BI-RADS density based on automated PD percentage."""
    if pd_val < 25:
        return 'A', 'Predominantly Fatty'
    elif pd_val < 50:
        return 'B', 'Scattered Fibroglandular'
    elif pd_val < 75:
        return 'C', 'Heterogeneously Dense'
    else:
        return 'D', 'Extremely Dense'

def main(patient_folder, output_dir=None):
    print("--- Phase 6: Percent Density (PD) and Patient-Level Categorization ---")
    dicom_paths = getDicomFiles(patient_folder)
    
    if not dicom_paths:
        print(f"No DICOM files found in {patient_folder}")
        return

    out_folder = output_dir if output_dir else os.getcwd()
    os.makedirs(out_folder, exist_ok=True)

    # Data structure for aggregation: pds[side][view] = pd_value
    pd_storage = {
        'L': {'CC': [], 'MLO': []},
        'R': {'CC': [], 'MLO': []}
    }
    
    patient_id = "UnknownID"
    patient_name = "UnknownName"
    patient_age = "UnknownAge"

    for i, impath in enumerate(dicom_paths):
        print(f"\nProcessing view {i+1}/{len(dicom_paths)}: {impath}")
        
        info = getinfo(impath)
        if i == 0:
            patient_id = info.get('patient_id', 'UnknownID').replace('^', '_')
            patient_name = info.get('name', 'UnknownName').replace('^', ' ')
            patient_age = info.get('age', 'UnknownAge')

        im_norm, _ = ffdmRead(impath, info)
        
        # Scale for analysis
        scale = info['psize'] / 0.4
        im_scaled = resize(im_norm, (int(round(im_norm.shape[0]*scale)), int(round(im_norm.shape[1]*scale))), preserve_range=True)
        
        # 1. Full breast segmentation
        mask, _, _ = segBreast(im_scaled, info['ismlo'])
        full_breast_area = np.sum(mask > 0)
        
        # 2. Dense tissue segmentation
        seg, _ = mseg(im_scaled, mask, 0.4)
        dense_area = np.sum(seg > 0)
        
        # 3. Calculate View-level PD
        pd_val = (dense_area / full_breast_area) * 100 if full_breast_area > 0 else 0
        cat_letter, _ = categorize_bi_rads(pd_val)
        
        print(f"  -> View:             {info.get('side', 'U')}-{info.get('view', 'U')}")
        print(f"  -> Percent Density:  {pd_val:.2f}% (Category {cat_letter})")
        
        # Store for aggregation
        side = info.get('side', 'U').upper()
        view = info.get('view', 'U').upper()
        if side in pd_storage and view in pd_storage[side]:
            pd_storage[side][view].append(pd_val)
        
        # Visualization omitted as per user request (report only)

    # --- CLINICAL AGGREGATION (Refined) ---
    # Breast PD is the max of its views. Patient PD uses both Global Avg and Max Breast.
    breast_pds = {'L': None, 'R': None}
    for side in ['L', 'R']:
        view_vals = []
        for v_type in ['CC', 'MLO']:
            if pd_storage[side][v_type]:
                view_vals.append(np.mean(pd_storage[side][v_type]))
        if view_vals:
            # Breast PD is the max of its available views
            breast_pds[side] = np.max(view_vals)

    valid_breast_pds = [v for v in breast_pds.values() if v is not None]
    if not valid_breast_pds:
        print("Error: No valid PD values found.")
        return

    # Metrics for Report
    global_avg_pd = np.mean(valid_breast_pds)
    max_breast_pd = np.max(valid_breast_pds)
    
    # BI-RADS Category (based on Max Breast PD for clinical risk)
    final_cat, final_desc = categorize_bi_rads(max_breast_pd)

    # Generate Clinical Report ([PatientID].txt)
    report_path = os.path.join(out_folder, f"{patient_id}.txt")
    with open(report_path, 'w') as f:
        f.write("============================================================\n")
        f.write("XyCAD_Breast - CLINICAL DENSITY REPORT\n")
        f.write("============================================================\n")
        f.write(f"Patient Name: {patient_name}\n")
        f.write(f"Patient ID:   {patient_id}\n")
        f.write(f"Patient Age:  {patient_age}\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"Views found: {len(dicom_paths)}\n")
        f.write("Processed demos: demo01, demo02, demo03, demo04, demo05, demo06\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"Left breast PD:  {breast_pds['L'] if breast_pds['L'] is not None else 0.0:.2f}%\n")
        f.write(f"Right breast PD: {breast_pds['R'] if breast_pds['R'] is not None else 0.0:.2f}%\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"Global Average PD:  {global_avg_pd:.2f}%\n")
        f.write(f"Max Breast PD:      {max_breast_pd:.2f}%\n")
        f.write(f"Final BI-RADS Category: {final_cat}\n")
        f.write(f"Classification:         {final_desc}\n")
        f.write("============================================================\n")
    
    print(f"\n[DONE] Clinical Report saved to: {report_path}")
    print(f"       Final Category: {final_cat}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Demo06: Patient-Level Density")
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    parser.add_argument("--folder", type=str, default=default_path, help="Path to patient folder")
    args = parser.parse_args()
    main(args.folder)
