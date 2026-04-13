"""
Python port of mseg.m
Morphological percent density (PD) estimation using area-gradient minimization.
"""
import numpy as np
from scipy.signal import windows, convolve
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from density.pprocess import pprocess


def mseg(im, mask, pixel_size):
    """
    Morphological PD estimation.

    Parameters
    ----------
    im : ndarray
        Input image.
    mask : ndarray
        Breast mask.
    pixel_size : float
        Pixel size in mm.

    Returns
    -------
    seg : ndarray
        Segmented dense tissue mask.
    pd : float
        Percent density (0.0 to 1.0).
    """
    # Segmentation parameters (MATLAB line 18-23)
    sigma = 0.8
    n_levels = 25
    skin_gap = 8.0 # mm
    area_th = 16.0 # mm^2

    im = np.asarray(im, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    # Smoothing filter (MATLAB line 27)
    # gausswin(5*sigma+1)
    win_len = int(5 * sigma + 1)
    if win_len < 1: win_len = 1
    h = windows.gaussian(win_len, std=sigma)
    h = h / np.sum(h) # normalize

    # Intensity levels
    masked_im = im[mask]
    if len(masked_im) == 0:
        return np.zeros_like(mask), 0.0

    imin = np.min(masked_im)
    imax = np.max(masked_im)
    ivalues = np.linspace(imin, imax, n_levels)

    # Compute morphological area curve (MATLAB line 35-39)
    areas = np.zeros(n_levels)
    for k in range(n_levels):
        seg_k = (im >= ivalues[k]) & mask
        areas[k] = np.sum(seg_k)

    # Compute first morphological area gradient (MAG) (MATLAB line 42)
    mag = -np.diff(areas) # MATLAB diff is area(k+1)-area(k), which is negative.
    # Wait, MATLAB: mag = diff(area); min(mag) will find where it drops fastest?
    # Actually, area is decreasing. diff(area) is [area2-area1, area3-area2...].
    # These are negative. Min(mag) finds the most negative drop.
    mag_raw = np.diff(areas)

    # Smooth MAG (MATLAB line 45)
    mag_smoothed = convolve(mag_raw, h, mode='same')

    # Minimize MAG (MATLAB line 48)
    idx = np.argmin(mag_smoothed)

    # Segment image (MATLAB line 51)
    # MATLAB uses i+1 because diff results in n-1 elements and i is index in diff.
    # ivalues(i+1) in 1-based MATLAB is the intensity at the end of the interval with max drop.
    # In Python, idx is index in mag_smoothed (len n-1).
    # ivalues[idx+1] matches.
    threshold = ivalues[idx + 1]
    seg = (im >= threshold) & mask

    # Post-process (MATLAB line 54)
    seg_final = pprocess(seg, mask, skin_gap, area_th, pixel_size)

    # Percent density
    total_area = np.sum(mask)
    if total_area > 0:
        pd = np.sum(seg_final) / total_area
    else:
        pd = 0.0

    return seg_final, pd
