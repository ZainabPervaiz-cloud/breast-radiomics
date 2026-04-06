"""
Python port of roi2mask.m
Convert ROI polygon points to a binary mask.
"""
import numpy as np
from skimage.draw import polygon2mask
from skimage.morphology import dilation


def roi2mask(xpoints, ypoints, imsize):
    """
    Convert ROI points to binary mask.

    Parameters
    ----------
    xpoints : array-like
        X-coordinates of ROI edges.
    ypoints : array-like
        Y-coordinates of ROI edges.
    imsize : tuple
        (height, width) of the output mask.

    Returns
    -------
    mask : ndarray
        MxN binary mask.
    """
    # polygon2mask takes (M, N) and a list of (y, x) points
    # MATLAB poly2mask(x, y, M, N)
    m, n = imsize
    # polygon2mask expects points as (row, col) i.e. (y, x)
    points = np.column_stack([ypoints, xpoints])
    mask = polygon2mask((m, n), points)
    
    # MATLAB: imdilate(mask, ones(9))
    mask = dilation(mask, np.ones((9, 9)))
    
    return mask
