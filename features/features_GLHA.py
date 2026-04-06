"""
Python port of features_GLHA.m
Gray-level Histogram Analysis (GLHA) features.
"""
import numpy as np
from scipy import stats


def features_GLHA(im, flist, mask=None):
    """
    Compute Gray-level histogram analysis features.

    Parameters
    ----------
    im : ndarray
        Grayscale input image.
    flist : list
        List of feature names to compute (e.g., 'imin', 'iavg', 'iske').
    mask : ndarray, optional
        Binary mask of the region of interest.

    Returns
    -------
    f : ndarray
        Column vector with computed features.
    """
    im = np.asarray(im, dtype=float)
    if mask is None:
        mask = np.ones_like(im, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    # Flatten pixels within the mask (MATLAB line 22)
    x = im[mask]
    if len(x) == 0:
        return np.zeros(len(flist))

    f = np.zeros(len(flist))
    eps = np.finfo(float).eps

    for n, feat in enumerate(flist):
        feat = feat.lower()
        if feat == 'imin':
            f[n] = np.min(x)
        elif feat == 'imax':
            f[n] = np.max(x)
        elif feat == 'iavg':
            f[n] = np.mean(x)
        elif feat == 'ient':
            # Histogram entropy (MATLAB line 33-35)
            c, _ = np.histogram(x, bins=256)
            p = c / np.sum(c)
            p = p[p > 0]
            f[n] = -np.sum(p * np.log2(p))
        elif feat == 'istd':
            f[n] = np.std(x, ddof=1) # MATLAB std uses N-1
        elif feat == 'ip05':
            f[n] = np.percentile(x, 5)
        elif feat == 'ip95':
            f[n] = np.percentile(x, 95)
        elif feat == 'ip30':
            f[n] = np.percentile(x, 30)
        elif feat == 'ip70':
            f[n] = np.percentile(x, 70)
        elif feat == 'iba1':
            p05 = np.percentile(x, 5)
            p95 = np.percentile(x, 95)
            u = np.mean(x)
            f[n] = (p95 - u + eps) / (u - p05 + eps)
        elif feat == 'iba2':
            p30 = np.percentile(x, 30)
            p70 = np.percentile(x, 70)
            u = np.mean(x)
            f[n] = (p70 - u + eps) / (u - p30 + eps)
        elif feat == 'iske':
            f[n] = stats.skew(x, bias=False) # MATLAB skewness uses bias correction
        elif feat == 'ikur':
            # MATLAB kurtosis subtracts nothing (Pearson's kurtosis)
            # scipy.stats.kurtosis subtracts 3 by default (Fisher's kurtosis)
            f[n] = stats.kurtosis(x, fisher=False, bias=False)
        elif feat == 'iran':
            f[n] = np.ptp(x) # peak-to-peak (range)
        else:
            raise ValueError(f"Unknown feature {feat}")

    return f
