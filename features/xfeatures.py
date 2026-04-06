"""
Python port of xfeatures.m
Dispatcher for all radiomic feature extractors.
"""
import numpy as np
import sys
import os
from skimage.transform import resize

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from features.features_GLHA import features_GLHA
from features.features_GLCM import features_GLCM
from features.features_GLRL import features_GLRL
from features.features_GLSM import features_GLSM
from features.features_FDIM import features_FDIM


def xfeatures(im, fname, mask=None, psize=0.4):
    """
    Extract image features.

    Parameters
    ----------
    im : ndarray or list of ndarray
        Grayscale image(s).
    fname : str or list of str
        Features to compute.
    mask : ndarray, optional
        Binary mask.
    psize : float
        Pixel size in mm.

    Returns
    -------
    f : ndarray
        Feature vector.
    """
    # Iterative processing (MATLAB line 34)
    if isinstance(im, list):
        if not isinstance(fname, list):
            fname = [fname]
        P = len(im)
        Q = len(fname)
        results = np.zeros((P, Q))
        for p in range(P):
            results[p, :] = xfeatures(im[p], fname, None, psize)
        return results

    if not isinstance(fname, list):
        fname = [fname]

    im = np.asarray(im, dtype=float)
    if mask is None:
        mask = np.ones_like(im, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    if mask.shape != im.shape:
        mask = resize(mask, im.shape, order=0, preserve_range=True).astype(bool)

    # Feature lists (MATLAB line 53-71)
    list1 = ['imin', 'imax', 'iavg', 'ient', 'istd', 'ip05', 'ip95', 'iba1', 'iba2', 'ip30', 'ip70', 'iske', 'ikur', 'iran']
    list2 = ['cene', 'ccor', 'ccon', 'chom', 'cent']
    list3 = ['rsre', 'rlre', 'rgln', 'rrpe', 'rrln', 'rlgr', 'rhgr']
    list4 = ['sgra', 'slap', 'swas', 'swav', 'swar', 'stev']
    list5 = ['fdim']

    def find_matches(ref, target):
        indices = [i for i, f in enumerate(target) if f.lower() in [r.lower() for r in ref]]
        matched_vals = [target[i] for i in indices]
        return matched_vals, indices

    flist1, i1 = find_matches(list1, fname)
    flist2, i2 = find_matches(list2, fname)
    flist3, i3 = find_matches(list3, fname)
    flist4, i4 = find_matches(list4, fname)
    flist5, i5 = find_matches(list5, fname)

    # Compute features by group
    f_all = np.zeros(len(fname))
    
    if flist1:
        f1 = features_GLHA(im, flist1, mask)
        f_all[i1] = f1
        
    if flist2:
        f2 = features_GLCM(im, flist2, mask)
        f_all[i2] = f2
        
    if flist3:
        f3 = features_GLRL(im, flist3, mask)
        f_all[i3] = f3
        
    if flist4:
        f4 = features_GLSM(im, flist4, mask)
        f_all[i4] = f4
        
    if flist5:
        # FDIM returns a scalar, but we might have it as scalar or list
        f5 = features_FDIM(im, mask, psize)
        f_all[i5] = f5

    return f_all
