import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from misc.io_utils import read_patient_folder
from segmentation.segBreast import segBreast

def demo_patient(patient_folder):
    """
    Demo: Process a full patient folder with multiple mammograms (CC, MLO sides).
    """
    print(f"Reading DICOMs from: {patient_folder}")
    patient_data = read_patient_folder(patient_folder)
    
    total_views = 0
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    view_idx = 0
    for side in ['L', 'R']:
        for view in ['CC', 'MLO']:
            if view in patient_data[side]:
                total_views += 1
                data = patient_data[side][view]
                info = data['info']
                im_norm = data['im_norm']
                
                print(f"Processing {side}-{view} ({info['psize']} mm/pixel)")
                
                try:
                    # Segment breast, mapping, etc.
                    mask, contour, cwall = segBreast(im_norm, info['ismlo'])
                    
                    if view_idx < 4:
                        ax = axes[view_idx]
                        ax.imshow(im_norm, cmap='gray')
                        ax.contour(mask, [0.5], colors='g', linewidths=1)
                        ax.set_title(f"{side}-{view}")
                        ax.axis('off')
                        view_idx += 1
                except Exception as e:
                    print(f"Error processing {side}-{view}: {e}")
                    
    print(f"Finished processing {total_views} views.")
    plt.tight_layout()
    # Save the figure to be reviewed
    plt.savefig('demo_patient_output.png')
    print("Saved visualization to demo_patient_output.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Run demo on patient folder")
    parser.add_argument("--folder", type=str, required=True, help="Path to patient folder")
    args = parser.parse_args()
    
    demo_patient(args.folder)
