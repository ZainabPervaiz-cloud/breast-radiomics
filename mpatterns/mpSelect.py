"""
Python port of mpSelect.m
Select micro-pattern prototypes using K-Means and multivariate normal pruning.
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import warnings


def mpSelect(M, C, np_val=40, pmax=0.2):
    """
    Select micro-pattern prototypes.

    Parameters
    ----------
    M : ndarray
        D x K array of patches.
    C : ndarray
        1 x K vector of class labels.
    np_val : int
        Number of prototypes per class.
    pmax : float
        Percentage of samples to keep after pruning.
    """
    P1 = M[:, C == 1]
    P0 = M[:, C == 0]
    
    if pmax < 1.0:
        # Prune using anomaly detection (Gaussian density)
        def prune(P, n_target):
            if P.shape[1] <= n_target:
                return P
            mu = np.mean(P, axis=1)
            # Regularize covariance to avoid singularity
            cov = np.cov(P) + np.eye(P.shape[0]) * 1e-6
            try:
                prob = multivariate_normal.pdf(P.T, mean=mu, cov=cov)
            except:
                # Fallback to mean distance if cov is still singular
                dist = np.sum((P.T - mu)**2, axis=1)
                prob = 1.0 / (dist + 1e-12)
            
            idx = np.argsort(prob)[::-1]
            return P[:, idx[:n_target]]

        n_max = int(round(pmax * M.shape[1] / 2.0))
        P1 = prune(P1, n_max)
        P0 = prune(P0, n_max)
        
    # K-Means clustering
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km0 = KMeans(n_clusters=np_val, n_init=10, max_iter=300).fit(P0.T)
        c0 = km0.cluster_centers_
        
        km1 = KMeans(n_clusters=np_val, n_init=10, max_iter=300).fit(P1.T)
        c1 = km1.cluster_centers_
        
    P_out = np.hstack([c0.T, c1.T])
    CP_out = np.hstack([np.zeros(c0.shape[0], dtype=bool), np.ones(c1.shape[0], dtype=bool)])
    
    return P_out, CP_out
