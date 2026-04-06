"""
Python port of phistogram.m
Computes texton histograms from micro-pattern samples.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpatterns.mpSampling import mpSampling


def phistogram(imdata, P, params):
    """
    Texton histogram of input image.

    Parameters
    ----------
    imdata : dict
        Image structure.
    P : ndarray
        N x NP array of prototypes.
    params : dict
        Parameters for mpSampling.
    """
    # Extract samples
    S, _ = mpSampling(imdata, params)
    
    # Compute distances (MATLAB line 34-42)
    # S is (mpsize^2) x nsamples
    # P is (mpsize^2) x nprototypes
    
    nsamples = S.shape[1]
    nprototypes = P.shape[1]
    
    # Vectorized distance computation
    # (S-p)^2 = S^2 - 2*S*p + p^2
    S_sq = np.sum(S**2, axis=0) # 1 x nsamples
    P_sq = np.sum(P**2, axis=0) # 1 x nprototypes
    
    # dist^2 = S_sq + P_sq.T - 2 * P.T * S
    dist_sq = S_sq[np.newaxis, :] + P_sq[:, np.newaxis] - 2 * np.dot(P.T, S)
    dist_sq[dist_sq < 0] = 0
    
    # Find closest prototype
    p_index = np.argmin(dist_sq, axis=0) # nsamples array
    
    # Generate histogram (relative frequencies)
    h, _ = np.histogram(p_index, bins=np.arange(nprototypes + 1))
    h = h.astype(float)
    if np.sum(h) > 0:
        h /= np.sum(h)
        
    return h
