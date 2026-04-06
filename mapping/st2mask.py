"""
Python port of st2mask.m
Convert ST-map parameters and limits to a binary mask in Cartesian coordinates.
"""
import numpy as np
from skimage.morphology import dilation


def st2mask(mapp, slims, tlims):
    """
    Convert ST-map region to Cartesian binary mask.

    Parameters
    ----------
    mapp : dict
        Mapping parameters from stmap().
    slims : list/tuple
        [s_min, s_max] limits.
    tlims : list/tuple
        [t_min, t_max] limits.

    Returns
    -------
    mask : ndarray
        Binary mask of the ROI.
    """
    m, n = mapp['size']
    # Use image size for grid density (MATLAB lines 22-23)
    s_vals = np.linspace(slims[0], slims[1], n)
    t_vals = np.linspace(tlims[0], tlims[1], m)
    S, T = np.meshgrid(s_vals, t_vals)

    # Calculate Cartesian coordinates (MATLAB lines 26-35)
    theta = (mapp['theta'][1] - mapp['theta'][0]) * T + mapp['theta'][0]
    PA = T * np.polyval(mapp['P1'], S) + np.polyval(mapp['P2'], S)
    PB = T * np.polyval(mapp['P3'], S) + np.polyval(mapp['P4'], S)

    Xi = np.round(PA * np.cos(theta) - PB * np.sin(theta) + mapp['x0']).astype(int)
    Yi = np.round(PA * np.sin(theta) + PB * np.cos(theta) + mapp['y0']).astype(int)

    mask = np.zeros((m, n), dtype=bool)
    
    # Filter valid points (MATLAB lines 37-39)
    # Note: MATLAB is 1-based, we use 0-based.
    valid = (Xi >= 0) & (Xi < n) & (Yi >= 0) & (Yi < m)
    
    mask[Yi[valid], Xi[valid]] = True
    
    # MATLAB: imdilate(mask, [1 1; 1 1])
    mask = dilation(mask, np.ones((2, 2)))

    if mapp.get('flip', False):
        mask = np.fliplr(mask)

    return mask
