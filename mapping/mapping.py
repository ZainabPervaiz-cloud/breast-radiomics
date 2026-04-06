"""
Python port of mapping.m
Perform ST mapping on an image using calculated parameters.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def mapping(im, mapp, s=None, t=None):
    """
    Perform ST coordinate mapping on image.

    Parameters
    ----------
    im : ndarray
        MxN grayscale image.
    mapp : dict
        Mapping parameters from stmap().
    s : array-like, optional
        Sampling points for s coordinate [0,1].
    t : array-like, optional
        Sampling points for t coordinate [0,1].

    Returns
    -------
    st_im : ndarray
        Mapped image.
    S : ndarray
        Grid of s-coordinates.
    T : ndarray
        Grid of t-coordinates.
    """
    im = np.asarray(im, dtype=float)
    m, n = im.shape

    if mapp.get('flip', False):
        im = np.fliplr(im)

    if s is None or t is None:
        # Default sampling (at image resolution)
        s = np.linspace(0.0, 1.0, n)
        t = np.linspace(0.0, 1.0, m)

    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)

    # Create sampling grid
    S, T = np.meshgrid(s, t)

    # Calculate angles (MATLAB line 40)
    theta = (mapp['theta'][1] - mapp['theta'][0]) * T + mapp['theta'][0]

    # Calculate P-A and P-B components
    PA = T * np.polyval(mapp['P1'], S) + np.polyval(mapp['P2'], S)
    PB = T * np.polyval(mapp['P3'], S) + np.polyval(mapp['P4'], S)

    # Convert to image coordinates (Xi, Yi)
    # Xi is column (x), Yi is row (y)
    Xi = PA * np.cos(theta) - PB * np.sin(theta) + mapp['x0']
    Yi = PA * np.sin(theta) + PB * np.cos(theta) + mapp['y0']

    # Sample with map_coordinates
    # map_coordinates expects (row, col) coordinates
    # Note: MATLAB is 1-based, Python 0-based. Subtraction of 1 is needed if mapp.x0/y0 are 1-based.
    # In my port, getcontour returns 0-based coordinates from find_contours.
    coords = np.array([Yi.ravel(), Xi.ravel()])
    
    st_im = map_coordinates(im, coords, order=1, mode='constant', cval=0.0)
    st_im = st_im.reshape(Xi.shape)

    return st_im, S, T
