"""
Python port of xy2st.m
Convert Cartesian XY coordinates to ST coordinates.
"""
import numpy as np
from scipy.interpolate import griddata


def xy2st(x, y, mapp, sc=1.0):
    """
    Convert Cartesian XY coordinates to ST coordinates.

    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates.
    mapp : dict
        Mapping parameters.
    sc : float
        Scaling factor.

    Returns
    -------
    si, ti : ndarray
        ST-coordinates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    m, n = mapp['size']
    s_bins = np.linspace(0, 1, n)
    t_bins = np.linspace(0, 1, m)
    
    # Create reference grid (MATLAB lines 26-31)
    S, T = np.meshgrid(s_bins, t_bins)
    theta = (mapp['theta'][1] - mapp['theta'][0]) * T + mapp['theta'][0]
    PA = T * np.polyval(mapp['P1'], S) + np.polyval(mapp['P2'], S)
    PB = T * np.polyval(mapp['P3'], S) + np.polyval(mapp['P4'], S)
    X = PA * np.cos(theta) - PB * np.sin(theta) + mapp['x0']
    Y = PA * np.sin(theta) + PB * np.cos(theta) + mapp['y0']

    if mapp.get('flip', False):
        x = sc * n - x

    # Interpolate (MATLAB: griddata)
    # Note: griddata in Python takes points as (N, D) array
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    si = griddata(points, S.ravel(), (x / sc, y / sc), method='linear')
    ti = griddata(points, T.ravel(), (x / sc, y / sc), method='linear')

    return si, ti
