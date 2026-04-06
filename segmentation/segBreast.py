"""
Python port of segBreast.m
Main entry point for breast and chest wall segmentation.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from segmentation.ffdmForeground import ffdmForeground
from segmentation.segChest import segChest
from misc.isright import isright


def segBreast(im, ismlo=True, isffdm=True):
    """
    Segment breast and chest wall.

    Parameters
    ----------
    im : ndarray
        Original mammogram image.
    ismlo : bool
        True if view is MLO, False otherwise.
    isffdm : bool
        True if image is FFDM. Default True.

    Returns
    -------
    mask : ndarray
        Binary mask of segmented breast.
    contour : dict
        Breast contour data.
    cwall : dict
        Chest wall parameters.
    """
    im = np.asarray(im, dtype=float)

    # Check laterality (MATLAB parity via isright)
    isflipped = isright(im)
    if isflipped:
        im = np.fliplr(im)

    if isffdm:
        mask, contour = ffdmForeground(im, ismlo)
    else:
        # Placeholder for sfmForeground if ever needed
        mask, contour = ffdmForeground(im, ismlo)

    contour['flip'] = isflipped

    if ismlo:
        cmask, cwall = segChest(im, contour)
    else:
        cmask = np.ones_like(mask, dtype=bool)
        cwall = {'m': 0.0, 'b': 0.0}

    mask = mask & cmask

    if isflipped:
        mask = np.fliplr(mask)

    return mask, contour, cwall
