import os
import argparse
from misc.io_utils import read_patient_folder

def main():
    parser = argparse.ArgumentParser(description="XyCAD_Breast: Python port of OpenBreast")
    parser.add_argument("--patient_folder", type=str, required=True, help="Path to the patient folder containing DICOM images")
    args = parser.parse_args()

    # Step 1: Read the patient folder (all views)
    patient_data = read_patient_folder(args.patient_folder)
    print(f"Loaded {len(patient_data)} views for patient.")

if __name__ == "__main__":
    main()
