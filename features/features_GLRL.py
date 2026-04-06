"""
Python port of features_GLRL.m
Gray-Level Run Length (GLRL) features.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from features.grayrlmatrix import grayrlmatrix
from features.grayrlprops import grayrlprops


def features_GLRL(im, flist, mask=None, par=None):
    """
    Compute GLRL texture features.

    Parameters
    ----------
    im : ndarray
        Grayscale input image.
    flist : list
        List of feature names ('rSRE', 'rLRE', etc.).
    mask : ndarray, optional
        Binary mask.
    par : dict, optional
        Parameters: 'nlevels' (default 256).

    Returns
    -------
    f : ndarray
        Feature vector.
    """
    if par is None:
        par = {'nlevels': 256}

    im = np.asarray(im, dtype=float)
    if mask is None:
        mask = np.ones_like(im, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    f = np.zeros(len(flist))
    
    masked_data = im[mask]
    if len(masked_data) == 0:
        return f

    gmin = np.min(masked_data)
    gmax = np.max(masked_data)
    
    offset = [1, 2, 3, 4] # 0, 45, 90, 135 degrees
    
    glrlms = grayrlmatrix(im, offset, par['nlevels'], [gmin, gmax], mask)
    props = grayrlprops(glrlms, np.sum(mask))

    for n, feat in enumerate(flist):
        feat = feat.upper()
        if feat == 'RSRE':
            f[n] = np.mean(props['SRE'])
        elif feat == 'RLRE':
            f[n] = np.mean(props['LRE'])
        elif feat == 'RGLN':
            f[n] = np.mean(props['GLN'])
        elif feat == 'RRPE':
            f[n] = np.mean(props['RP'])
        elif feat == 'RRLN':
            f[n] = np.mean(props['RLN'])
        elif feat == 'RLGR':
            f[n] = np.mean(props['LGRE'])
        elif feat == 'RHGR':
            f[n] = np.mean(props['HGRE'])
        else:
             # handle case sensitivity if needed
             pass

    return f
