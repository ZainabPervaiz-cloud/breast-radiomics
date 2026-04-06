"""
Python port of sqmax.m
Find largest inscribed square ROI within a binary mask.
Delegates to FindLargestSquares from the support module.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from support.Inscribed_Rectangle.FindLargestSquares import FindLargestSquares


def sqmax(mask):
    """
    Find largest circumscribed squared ROI within a binary mask.

    Parameters
    ----------
    mask : ndarray
        MxN binary mask.

    Returns
    -------
    sq : ndarray
        MxN binary mask with the largest square inscribed in mask.
    rect : list
        [x, y, width, height] bounding box (0-based).
    """
    mask = np.asarray(mask, dtype=bool)
    sq = np.zeros_like(mask, dtype=bool)
    s = FindLargestSquares(mask)
    # Find the location of the maximum value in s
    idx = np.unravel_index(np.argmax(s), s.shape)
    y, x = idx
    size = int(s[y, x])
    if size > 0:
        sq[y:y + size + 1, x:x + size + 1] = True
    rect = [x, y, size - 1, size - 1]
    return sq, rect
