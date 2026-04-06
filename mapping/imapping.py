"""
Python port of imapping.m
Inverse ST-mapping: reconstruct Cartesian image from ST representation.
"""
import numpy as np
from scipy.interpolate import griddata
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mapping.st2mask import st2mask


def imapping(st_im, mapp, s=None, t=None):
    """
    Inverse ST-mapping.

    Parameters
    ----------
    st_im : ndarray
        NTxNS matrix with ST representation.
    mapp : dict
        Mapping parameters.
    s, t : array-like, optional
        Sampling points.

    Returns
    -------
    im : ndarray
        MxN Cartesian image.
    isvalid : ndarray
        Binary validity mask.
    """
    m_out, n_out = mapp['size']

    if s is None or t is None:
        s = np.linspace(0.0, 1.0, st_im.shape[1])
        t = np.linspace(0.0, 1.0, st_im.shape[0])

    S, T = np.meshgrid(s, t)
    theta = (mapp['theta'][1] - mapp['theta'][0]) * T + mapp['theta'][0]
    PA = T * np.polyval(mapp['P1'], S) + np.polyval(mapp['P2'], S)
    PB = T * np.polyval(mapp['P3'], S) + np.polyval(mapp['P4'], S)
    x = PA * np.cos(theta) - PB * np.sin(theta) + mapp['x0']
    y = PA * np.sin(theta) + PB * np.cos(theta) + mapp['y0']

    # Cartesian grid
    xi = np.arange(n_out)
    yi = np.arange(m_out)
    XI, YI = np.meshgrid(xi, yi)

    # Use griddata for inverse mapping - 'nearest' is much faster for large images
    points = np.column_stack([x.ravel(), y.ravel()])
    im_out = griddata(points, st_im.ravel(), (XI, YI), method='nearest', fill_value=0.0)

    isvalid = st2mask(mapp, [0.0, 1.0], [0.0, 1.0])

    if mapp.get('flip', False):
        im_out = np.fliplr(im_out)

    im_out[~isvalid | np.isnan(im_out)] = 0.0

    return im_out, isvalid
