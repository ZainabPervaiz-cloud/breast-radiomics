"""
Python port of rcenters.m
Get region centers in XY coordinates on a grid within a binary mask.
"""
import numpy as np
from skimage.measure import regionprops, label as skimage_label


def rcenters(spacing, wsize, mask):
    """
    Get region centers in XY coordinates.

    Parameters
    ----------
    spacing : int
        Step between region centers in pixels.
    wsize : int
        Window size of each region.
    mask : ndarray
        MxN binary mask of valid breast area.

    Returns
    -------
    x : ndarray
        1-based x-coordinates of valid ROI centers.
    y : ndarray
        1-based y-coordinates of valid ROI centers.
    """
    mask = np.asarray(mask, dtype=bool)
    m, n = mask.shape
    delta = int(np.floor(0.5 * wsize))

    # Build grid (1-based to match MATLAB)
    xs_1d = np.arange(1, n + 1, spacing)
    ys_1d = np.arange(1, m + 1, spacing)
    xs, ys = np.meshgrid(xs_1d, ys_1d)
    xs = xs.ravel().astype(int)
    ys = ys.ravel().astype(int)

    # Remove regions that go outside image bounds (1-based)
    xmin = xs - delta
    xmax = xs + delta
    ymin = ys - delta
    ymax = ys + delta
    keep = (xmin >= 1) & (xmax <= n) & (ymin >= 1) & (ymax <= m)
    xs = xs[keep]
    ys = ys[keep]
    xmin = xmin[keep]
    xmax = xmax[keep]
    ymin = ymin[keep]
    ymax = ymax[keep]

    # Remove regions whose corners fall outside the mask (0-based indexing)
    i1 = mask[ymin - 1, xmin - 1]  # upper-left
    i2 = mask[ymin - 1, xmax - 1]  # upper-right
    i3 = mask[ymax - 1, xmin - 1]  # lower-left
    i4 = mask[ymax - 1, xmax - 1]  # lower-right
    valid = i1 & i2 & i3 & i4
    x = xs[valid]
    y = ys[valid]

    if len(x) > 0 and len(y) > 0:
        return x, y

    # Fallback: single window at centroid (matches MATLAB else case)
    lbl = skimage_label(mask)
    props = regionprops(lbl)
    if props:
        cy, cx = props[0].centroid
        xc = min(int(round(cx)) + 1, n - delta)  # 1-based
        yc = min(int(round(cy)) + 1, m - delta)
        xc = max(xc, 1 + delta)
        yc = max(yc, 1 + delta)
        return np.array([xc]), np.array([yc])

    return np.array([]), np.array([])
