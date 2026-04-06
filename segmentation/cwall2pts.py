"""
Python port of cwall2pts.m
Convert chest wall (pectoral line) parameters to points.
"""
import numpy as np


def cwall2pts(cwall):
    """
    Convert chest wall structure to points.

    Parameters
    ----------
    cwall : dict
        Structure with 'm' (slope) and 'b' (intercept).

    Returns
    -------
    pts : ndarray
        100x2 array with [x, y] coordinates of points on the chest wall.
    """
    m = cwall['m']
    b = cwall['b']

    # Find the x where y=0: 0 = b + m*x -> x = -b/m
    if abs(m) < 1e-12:
        # horizontal line (shouldn't happen for chest wall but for safety)
        x_end = 1000
    else:
        x_end = -b / m

    x = np.linspace(1.0, float(np.floor(x_end)), 100)
    y = b + m * x

    # MATLAB lines 18-19:
    x[0] = 0
    y[-1] = 0

    return np.column_stack([x, y])
