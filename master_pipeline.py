from pathlib import Path
import os
import sys
import argparse
import traceback

# Add root directory to sys.path for internal imports
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Import modular demos
try:
    from demos import demo01, demo02, demo03, demo04, demo05, demo06
except ImportError:
    import demo01, demo02, demo03, demo04, demo05, demo06

def run_pipeline(input_root, output_root):
    """
    Recursively finds folders with DICOMS and mirrors structure in output.
    Uses pathlib for robust clinical structuring: PatientID/demoXX/
    """
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    
    print(f"\n" + "="*70)
    print(f"XyCAD_Breast Master Clinical Pipeline (Production Multi-Level)")
    print(f"Input Root:  {input_root}")
    print(f"Output Root: {output_root}")
    print("="*70 + "\n")

    # 1. Discover leaf folders and group by patient directory
    print("Scanning for patients and views...")
    patient_groups = {} # patient_src_root -> list of dicoms (or just use patient_src_root)
    patient_folders = set()
    
    for root, dirs, files in os.walk(input_root):
        dcm_files = [f for f in files if f.lower().endswith(('.dcm', '.ima'))]
        if dcm_files:
            root_path = Path(root)
            name = root_path.name.upper().replace('_', ' ').replace('-', ' ')
            # Heuristic to detect if this folder is just a view folder (L MLO, R CC, etc.)
            is_view = any(v in name.split() for v in ['MLO', 'CC', 'L', 'R', 'LEFT', 'RIGHT', 'VIEW']) or any(v in name for v in ['L MLO', 'R MLO', 'L CC', 'R CC'])
            
            if is_view:
                patient_folders.add(root_path.parent)
            else:
                patient_folders.add(root_path)

    if not patient_folders:
        print(f"Critical: No valid DICOM patients found in {input_root}")
        return

    # Filter out any folders that are ancestors of other patient folders to prevent nested processing
    # (e.g. if 'samples' has a stray .dcm, it shouldn't process everything inside it again).
    final_patients = set()
    for p in patient_folders:
        is_ancestor = any((p != other and p in other.parents) for other in patient_folders)
        if not is_ancestor:
            final_patients.add(p)
            
    # 2. Process each patient group
    patient_list = sorted(list(final_patients))
    print(f"Discovered {len(patient_list)} patients to process.\n")

    for i, patient_src_root in enumerate(patient_list):
        pid = patient_src_root.name
            
        # Determine relative path for output mirroring
        rel_path = patient_src_root.relative_to(input_root)
        if str(rel_path) == ".":
            patient_out_root = output_root / pid
        else:
            patient_out_root = output_root / rel_path
            
        print(f"[{i+1}/{len(patient_list)}] Processing Patient: {pid}")
        print(f"             Source: {patient_src_root}")
        print(f"             Output: {patient_out_root}")
        print("-" * 50)
        
        try:
            # Pre-create demo subfolders
            for d in range(1, 7):
                (patient_out_root / f"demo0{d}").mkdir(parents=True, exist_ok=True)

            # Phase 1: Breast Segmentation
            demo01.main(str(patient_src_root), output_dir=str(patient_out_root / "demo01"))
            
            # Phase 2: Spatial Transformation
            demo02.main(str(patient_src_root), output_dir=str(patient_out_root / "demo02"))
            
            # Phase 3: ROI Selection
            demo03.main(str(patient_src_root), output_dir=str(patient_out_root / "demo03"))
            
            # Phase 4: Feature Extraction (Mandatory)
            demo04.main(str(patient_src_root), output_dir=str(patient_out_root / "demo04"))
            
            # Phase 5: Density Estimation
            demo05.main(str(patient_src_root), output_dir=str(patient_out_root / "demo05"))
            
            # Phase 6: Clinical Aggregation & Report
            # This will save [PatientID].txt in the patient root folder
            demo06.main(str(patient_src_root), output_dir=str(patient_out_root))
            
            # Move demo06 images to their own subfolder
            d6_img_dir = patient_out_root / "demo06"
            for f in os.listdir(patient_out_root):
                if f.startswith("demo06_view_") and f.endswith(".png"):
                    os.rename(patient_out_root / f, d6_img_dir / f)

            print(f"\n[OK] Completed {pid}\n")
            
        except Exception:
            print(f"\n[!] ERROR processing {pid}:")
            traceback.print_exc()
            print("-" * 50)

    print("\n" + "="*70)
    print(f"All clinical processing completed.")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser("XyCAD_Breast Orchestrator")
    default_input = root_dir / "samples"
    default_output = root_dir / "prediction"
    
    parser.add_argument("--input", type=str, default=str(default_input), help="Input folder")
    parser.add_argument("--output", type=str, default=str(default_output), help="Output folder")
    
    args = parser.parse_args()
    run_pipeline(args.input, args.output)

if __name__ == "__main__":
    main()
