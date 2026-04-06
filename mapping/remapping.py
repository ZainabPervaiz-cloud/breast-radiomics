"""
Python port of remapping.m
Remap mammography image using a reference ST map (implicit registration).
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mapping.st2mask import st2mask


def remapping(im, mapp, rmapp):
    """
    Remap image from one ST-space to another (reference) ST-space.

    Parameters
    ----------
    im : ndarray
        Original image.
    mapp : dict
        ST mapping of input image.
    rmapp : dict
        Reference ST mapping.

    Returns
    -------
    Ir : ndarray
        Remapped image.
    isvalid : ndarray
        Validity mask.
    """
    im = np.asarray(im, dtype=float)
    nt, nr = im.shape
    m_img, n_img = im.shape

    s = np.linspace(0.0, 1.0, nr)
    t = np.linspace(0.0, 1.0, nt)
    S, T = np.meshgrid(s, t)

    # Coordinates in source ST-space
    theta1 = (mapp['theta'][1] - mapp['theta'][0]) * T + mapp['theta'][0]
    PA1 = T * np.polyval(mapp['P1'], S) + np.polyval(mapp['P2'], S)
    PB1 = T * np.polyval(mapp['P3'], S) + np.polyval(mapp['P4'], S)
    x1 = np.round(PA1 * np.cos(theta1) - PB1 * np.sin(theta1) + mapp['x0']).astype(int)
    y1 = np.round(PA1 * np.sin(theta1) + PB1 * np.cos(theta1) + mapp['y0']).astype(int)

    # Coordinates in reference ST-space
    theta2 = (rmapp['theta'][1] - rmapp['theta'][0]) * T + rmapp['theta'][1] # Re-checking MATLAB... wait
    # Re-reading remapping.m line 36: theta = (RMAPP.theta(2) - RMAPP.theta(1))*T + RMAPP.theta(1);
    theta2 = (rmapp['theta'][1] - rmapp['theta'][0]) * T + rmapp['theta'][0]
    PA2 = T * np.polyval(rmapp['P1'], S) + np.polyval(rmapp['P2'], S)
    PB2 = T * np.polyval(rmapp['P3'], S) + np.polyval(rmapp['P4'], S)
    x2 = np.round(PA2 * np.cos(theta2) - PB2 * np.sin(theta2) + rmapp['x0']).astype(int)
    y2 = np.round(PA2 * np.sin(theta2) + PB2 * np.cos(theta2) + rmapp['y0']).astype(int)

    # Validity checks
    invalid1 = (x1 >= n_img) | (x1 < 0) | (y1 >= m_img) | (y1 < 0)
    invalid2 = (x2 >= n_img) | (x2 < 0) | (y2 >= m_img) | (y2 < 0)
    invalid = invalid1 | invalid2
    
    x1_v = x1[~invalid]
    y1_v = y1[~invalid]
    x2_v = x2[~invalid]
    y2_v = y2[~invalid]

    if mapp.get('flip', False):
        im = np.fliplr(im)

    Ir = np.zeros_like(im)
    Ir[y2_v, x2_v] = im[y1_v, x1_v]

    isvalid = st2mask(rmapp, [0.0, 1.0], [0.0, 1.0])

    if rmapp.get('flip', False):
        Ir = np.fliplr(Ir)

    return Ir, isvalid
