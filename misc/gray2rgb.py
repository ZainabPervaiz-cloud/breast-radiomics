"""
Python port of gray2rgb.m
Convert grayscale image to RGB using a colormap.
"""
import numpy as np
import matplotlib.cm as cm


def gray2rgb(im, cmap=None, clims=None):
    """
    Convert grayscale image to RGB.

    Parameters
    ----------
    im : ndarray
        MxN intensity matrix.
    cmap : ndarray or None
        Nx3 RGB colormap array (values in [0,1]). Default: viridis (≈parula).
    clims : array-like or None
        [Imin, Imax] intensity range to map. Default: [min(im), max(im)].

    Returns
    -------
    im_rgb : ndarray
        MxNx3 uint8 RGB image.
    """
    im = np.asarray(im, dtype=float)

    if clims is None:
        clims = [im.min(), im.max()]

    # Build colormap
    if cmap is None:
        # viridis is the closest matplotlib equivalent to MATLAB parula
        cmap_obj = cm.get_cmap('viridis', 64)
        cmap = cmap_obj(np.linspace(0, 1, 64))[:, :3]  # Nx3, [0,1]

    cmap = np.asarray(cmap, dtype=float)
    if cmap.max() > 1.0:
        cmap = cmap / 255.0

    # Normalize image to [0, 1]
    lo, hi = float(clims[0]), float(clims[1])
    if hi == lo:
        im_n = np.zeros_like(im)
    else:
        im_n = (im - lo) / (hi - lo)
    im_n = np.clip(im_n, 0.0, 1.0)

    ncolors = cmap.shape[0] - 1
    idx = (ncolors * im_n).astype(np.uint8)  # matches MATLAB uint8(ncolors*im)+1 (0-based here)
    idx = np.clip(idx, 0, ncolors)

    im_rgb = (cmap[idx] * 255).astype(np.uint8)
    return im_rgb
