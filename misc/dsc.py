"""
Python port of dsc.m
Dice's Similarity Coefficient between two binary masks.
"""
import numpy as np


def dsc(mask1, mask2):
    """
    Compute Dice's Similarity Coefficient.

    Parameters
    ----------
    mask1, mask2 : ndarray
        Binary (bool or 0/1) arrays of the same shape.

    Returns
    -------
    d : float
        DSC value in [0, 1].
    """
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    num = 2.0 * np.sum(mask1 & mask2)
    den = np.sum(mask1) + np.sum(mask2)
    if den == 0:
        return 0.0
    return num / den
