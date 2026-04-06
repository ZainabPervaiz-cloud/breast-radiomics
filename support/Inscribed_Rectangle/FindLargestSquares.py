"""
Python port of FindLargestSquares.m
Finds the size of the largest square region of ones starting at each pixel (r, c)
as the upper-left corner.
"""
import numpy as np


def FindLargestSquares(I):
    """
    Find largest square regions with all points set to 1.

    Parameters
    ----------
    I : ndarray
        B/W boolean matrix.

    Returns
    -------
    S : ndarray
        For each pixel I[r,c], returns the size of the largest all-white 
        square with its upper-left corner at I[r,c].
    """
    I = np.asarray(I, dtype=bool)
    nr, nc = I.shape
    S = I.astype(float)

    # MATLAB logic: for r=(nr-1):-1:1, for c=(nc-1):-1:1
    # This is a DP from bottom-right to top-left.
    for r in range(nr - 2, -1, -1):
        for c in range(nc - 2, -1, -1):
            if S[r, c]:
                # S(r,c) = min([a b d]) + 1
                a = S[r, c + 1]
                b = S[r + 1, c]
                d = S[r + 1, c + 1]
                S[r, c] = min(a, b, d) + 1.0
                
    return S
