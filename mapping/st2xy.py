"""
Python port of st2xy.m
Convert ST-coordinates to Cartesian XY coordinates.
"""
import numpy as np


def st2xy(s, t, mapp, refsize=None):
    """
    Convert ST-coordinates to Cartesian XY coordinates.

    Parameters
    ----------
    s, t : array-like
        ST-coordinates.
    mapp : dict
        Mapping parameters.
    refsize : tuple, optional
        Reference image size for scaling.

    Returns
    -------
    x, y : ndarray
        Cartesian coordinates.
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    
    m_orig, n_orig = mapp['size']
    if refsize is None:
        sc = np.array([1.0, 1.0])
    else:
        sc = np.array(refsize) / np.array([m_orig, n_orig])

    # Transform (MATLAB lines 23-27)
    theta = (mapp['theta'][1] - mapp['theta'][0]) * t + mapp['theta'][0]
    PA = t * np.polyval(mapp['P1'], s) + np.polyval(mapp['P2'], s)
    PB = t * np.polyval(mapp['P3'], s) + np.polyval(mapp['P4'], s)
    
    x = np.round(PA * np.cos(theta) - PB * np.sin(theta) + mapp['x0'])
    y = np.round(PA * np.sin(theta) + PB * np.cos(theta) + mapp['y0'])

    # Re-scale and flip (MATLAB lines 31-40)
    # MATLAB: round(sc(2)*(x-1) + 1). 
    # Python 0-based: round(sc[1]*x)
    x = np.round(sc[1] * x)
    y = np.round(sc[0] * y)

    n_ref = int(round(sc[1] * n_orig))
    m_ref = int(round(sc[0] * m_orig))

    if mapp.get('flip', False):
        x = n_ref - x

    # Clamp to bounds
    x = np.clip(x, 0, n_ref - 1)
    y = np.clip(y, 0, m_ref - 1)

    return x, y
