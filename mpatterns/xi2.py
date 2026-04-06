"""
Python port of xi2.m
Chi-squared kernel for histogram comparison.
"""
import numpy as np


def xi2(U, V):
    """
    Xi-squared kernel.
    G = 1 - 0.5 * sum((U-V)^2 / (U+V))

    Parameters
    ----------
    U : ndarray
        M x D array of histograms.
    V : ndarray
        N x D array of histograms.
    """
    m = U.shape[0]
    n = V.shape[0]
    G = np.zeros((m, n))
    eps = np.finfo(float).eps
    
    for i in range(m):
        u = U[i, :]
        # Vectorized over V
        num = (u - V)**2
        den = u + V + eps
        G[i, :] = 1.0 - 0.5 * np.sum(num / den, axis=1)
        
    return G
