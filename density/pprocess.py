"""
Python port of pprocess.m
Post-process density segmentation by applying morphological and area filters.
"""
import numpy as np
from skimage.morphology import erosion, remove_small_objects


def pprocess(seg, mask, skin_gap_mm, area_th_mm2, pixel_size):
    """
    Post-process density segmentation.

    Parameters
    ----------
    seg : ndarray
        Binary segmentation mask of dense tissue.
    mask : ndarray
        Breast mask.
    skin_gap_mm : float
        Skin gap in mm.
    area_th_mm2 : float
        Area threshold in mm^2.
    pixel_size : float
        Pixel size in mm.

    Returns
    -------
    seg : ndarray
        Cleaned density mask.
    """
    # Convert parameters to pixels (MATLAB line 5-6)
    # skin_gap = round(2*skin_gap/pixel_size)
    skin_gap_px = int(round(2 * skin_gap_mm / pixel_size))
    # area_threshold = round(area_th/(pixel_size^2))
    area_th_px = int(round(area_th_mm2 / (pixel_size**2)))

    # Erode breast mask (MATLAB line 9)
    if skin_gap_px > 0:
        maske = erosion(mask, np.ones((skin_gap_px, skin_gap_px)))
    else:
        maske = mask

    # Apply area filter (MATLAB line 12)
    # bwareaopen(seg&maske, area_threshold)
    combined = seg & maske
    if area_th_px > 0:
        cleaned = remove_small_objects(combined.astype(bool), min_size=area_th_px)
    else:
        cleaned = combined
        
    return cleaned
