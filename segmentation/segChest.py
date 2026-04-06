"""
Python port of segChest.m
Detect chest wall (pectoral line) in FFDM image using Hough transform.
"""
import numpy as np
from skimage.feature import canny
from skimage.morphology import dilation, disk
from scipy.ndimage import gaussian_filter
import sys
import os


def segChest(im, contour):
    """
    Chest wall detection in FFDM image.

    Parameters
    ----------
    im : ndarray
        MxN grayscale mammography image.
    contour : dict
        Breast contour as returned by getcontour().

    Returns
    -------
    mask : ndarray
        MxN binary mask with breast region (excludes chest wall).
    cwall : dict
        Slope 'm' and intercept 'b' of the pectoral line.
    """
    im = np.asarray(im, dtype=float)
    m_img, n_img = im.shape

    # Pre-process image (MATLAB lines 22-23)
    # dilate with disk 8 then gaussian 5x5 sigma 1
    imc = dilation(im, disk(8))
    imc = gaussian_filter(imc, sigma=1.0)

    # Edge detection (MATLAB line 27)
    # MATLAB edge(im, 'canny', [], 2.0)
    edge_map = canny(imc, sigma=2.0)

    # Crop edge map according to contour (MATLAB lines 36-38)
    ymax = int(round(0.6 * np.max(contour['y'])))
    # find min x where y < ymax
    mask_y = contour['y'] < ymax
    if np.any(mask_y):
        xmax = int(round(np.min(contour['x'][mask_y])))
    else:
        xmax = n_img // 2

    # Clip indices to image bounds
    ymax = min(ymax, m_img)
    xmax = min(xmax, n_img)
    
    cropped_edge = edge_map[:ymax, :xmax].copy()

    # remove lower diagonal (MATLAB lines 41-44)
    m, n = cropped_edge.shape
    y_coords, x_coords = np.indices((m, n))
    # yref = m - (m - 1)*(x-1)/(n-1);  (Note: MATLAB 1-based x, we use x_coords 0-based)
    yref = m - (m - 1) * (x_coords) / (n - 1 if n > 1 else 1)
    cropped_edge[y_coords + 1 > yref] = False

    # Detect pectoral line using custom Hough logic (MATLAB lines 57-81)
    i_coords, j_coords = np.where(cropped_edge)
    if len(i_coords) == 0:
        # Fallback if no edges found
        return np.ones((m_img, n_img), dtype=bool), {'m': 0, 'b': 0}

    # x*cos(theta) + y*sin(theta) = rho
    # Quantize parameter space
    num_bins = 128
    rho_max = min(m, n)
    rho_min = 1.0
    theta_min = 20 * np.pi / 180
    theta_max = 45 * np.pi / 180

    thetas = np.linspace(theta_min, theta_max, num_bins)
    rhos = np.linspace(rho_min, rho_max, num_bins)

    # Accumulation
    # rho_k = x*cos(theta) + y*sin(theta)
    # Broadcast to (len(points), num_bins)
    rho_calculated = j_coords[:, None] * np.cos(thetas) + i_coords[:, None] * np.sin(thetas)
    
    # Accumulate (similar to histc in MATLAB line 70)
    accumulator = np.zeros((num_bins, num_bins))
    for t_idx in range(num_bins):
        hist, _ = np.histogram(rho_calculated[:, t_idx], bins=num_bins, range=(rho_min, rho_max))
        accumulator[:, t_idx] = hist

    # Find maximum (MATLAB line 73)
    max_idx = np.argmax(accumulator)
    rho_idx, theta_idx = np.unravel_index(max_idx, accumulator.shape)

    T = thetas[theta_idx]
    R = (rho_max - rho_min) * rho_idx / (num_bins - 1) + rho_min
    
    # R = x*cos(T) + y*sin(T) -> y = (R - x*cos(T))/sin(T)
    # b = R/sin(T), m = -cos(T)/sin(T)
    b = R / np.sin(T)
    m_slope = -np.cos(T) / np.sin(T)

    # Generate full-size mask (MATLAB lines 83-85)
    y_full, x_full = np.indices((m_img, n_img))
    mask = np.ones((m_img, n_img), dtype=bool)
    mask[y_full < (b + m_slope * x_full)] = False

    return mask, {'m': m_slope, 'b': b}
