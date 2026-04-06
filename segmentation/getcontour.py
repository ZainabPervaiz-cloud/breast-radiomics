"""
Python port of getcontour.m
Retrieve breast contour from a binary mask.
"""
import numpy as np
from skimage.measure import find_contours
from scipy.interpolate import pchip_interpolate
from scipy.ndimage import gaussian_filter1d


def getcontour(mask, cflag=False):
    """
    Retrieve breast contour from a binary mask.

    Parameters
    ----------
    mask : ndarray
        MxN binary segmentation mask.
    cflag : bool
        True for contour clipping based on curvature. Default False.

    Returns
    -------
    contour : dict
        Structure with 'x', 'y', 'ycut', and 'size'.
    """
    mask = np.asarray(mask, dtype=bool)
    m, n = mask.shape
    elim = 8
    npts = 100
    k_th = 0.05

    # Find contour points using skimage.measure.find_contours
    # matches MATLAB bwboundaries(mask)
    contours = find_contours(mask, 0.5)
    if not contours:
        return {'x': np.array([]), 'y': np.array([]), 'ycut': m, 'size': mask.shape}

    # Use the longest contour (primary breast boundary)
    main_contour = sorted(contours, key=len, reverse=True)[0]
    ys = main_contour[:, 0]
    xs = main_contour[:, 1]

    # Filter out edge points (MATLAB lines 25-28)
    # Note: MATLAB is 1-based, we are 0-based.
    # elim=8 means points within [0, 8] or [m-8, m] are removed.
    remov = (xs <= elim) | (ys <= elim) | (ys >= (m - elim))
    xs = xs[~remov]
    ys = ys[~remov]

    if len(xs) < 2:
        return {'x': np.array([]), 'y': np.array([]), 'ycut': m, 'size': mask.shape}

    # sub-sample and smooth contour (MATLAB lines 31-33)
    # Interp using PCHIP and then smooth
    t_old = np.linspace(0, 1, len(xs))
    t_new = np.linspace(0, 1, npts)
    
    xs_interp = pchip_interpolate(t_old, xs, t_new)
    ys_interp = pchip_interpolate(t_old, ys, t_new)
    
    # MATLAB smooth(x) defaults to moving average 5. gaussian_filter1d(1.5) is similar.
    xs_smooth = gaussian_filter1d(xs_interp, sigma=1.5)
    ys_smooth = gaussian_filter1d(ys_interp, sigma=1.5)

    # Crop contour by curvature analysis
    xc, yc, ycut = cropContour(xs_smooth, ys_smooth, k_th)

    if not cflag:
        res_x = xs_smooth
        res_y = ys_smooth
    else:
        res_x = xc
        res_y = yc

    return {
        'x': res_x.ravel(),
        'y': res_y.ravel(),
        'ycut': float(ycut),
        'size': mask.shape
    }


def cropContour(xs, ys, k_th):
    """
    Crop breast contour using curvature thresholding.
    """
    # compute curvature k (MATLAB lines 69-77)
    dx1 = np.gradient(xs)
    dx2 = np.gradient(dx1)
    dy1 = np.gradient(ys)
    dy2 = np.gradient(dy1)

    # Note: MATLAB indices shift slightly with diff; gradient handles it differently but results are comparable.
    denom = (dx1**2 + dy1**2)**1.5
    k = (dx1 * dy2 - dy1 * dx2) / (denom + 1e-12)

    # Cut contour points with curvature above threshold (MATLAB lines 82-90)
    kmin_idx = np.argmin(k)
    kmin = k[kmin_idx]

    # condition: abs(kmin) > K_th and peak location constraints
    if (abs(kmin) > k_th) and (xs[kmin_idx] < 0.4 * np.max(xs)) and (ys[kmin_idx] > 0.5 * np.max(ys)):
        ycut = int(np.floor(ys[kmin_idx]))
        xc = xs[:kmin_idx + 1]
        yc = ys[:kmin_idx + 1]
    else:
        ycut = int(np.floor(np.max(ys)))
        xc = xs
        yc = ys

    return xc, yc, ycut
