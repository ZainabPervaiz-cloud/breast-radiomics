"""
Python port of isright.m
Determine whether mammogram breast is on the right side of the image.
"""
import numpy as np


def isright(im):
    """
    Determine whether mammogram is left or right.

    Returns True if the breast is on the RIGHT side of the image.

    MATLAB logic:
        im(im > 0.95*max(im(:))) = 0;   % suppress very bright pixels
        s = sum(im);                     % column-wise sum
        n = round(0.5*size(im, 2));      % midpoint column
        flag = sum(s(1:n)) < sum(s(n+1:end));  % more energy in right half
    """
    im = im.astype(float)
    im[im > 0.95 * im.max()] = 0.0   # suppress saturation / high-intensity artefacts
    s = im.sum(axis=0)                # column-wise sum (matches MATLAB sum(im))
    n = round(0.5 * im.shape[1])     # mid-column index (0-based: use n)
    return s[:n].sum() < s[n:].sum()  # True → more energy on right → right breast
