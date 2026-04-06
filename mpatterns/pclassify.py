"""
Python port of pclassify.m
Classifies mammography images using KNN on texton histograms.
"""
import numpy as np
from scipy import stats
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpatterns.phistogram import phistogram


def pclassify(dataset, model, K):
    """
    Classify using KNN on histograms.

    Parameters
    ----------
    dataset : dict or list of dict
    model : dict
    K : int
    """
    if isinstance(dataset, list):
        m = len(dataset)
        c_all = np.zeros(m)
        d_all = np.zeros((m, 2))
        l_all = np.zeros((m, K))
        
        print("Classifying images...")
        for n in range(m):
            cn, dn, ln = pclassify(dataset[n], model, K)
            c_all[n] = cn
            d_all[n, :] = dn
            l_all[n, :] = ln
        return c_all, d_all, l_all

    # 1. Compute histogram
    h = phistogram(dataset, model['P'], model['params'])
    
    # 2. Chi-squared distance (MATLAB line 41)
    # Xi = 0.5*sum( ((model.H - h).^2+eps)./(model.H + h +eps), 2);
    eps = np.finfo(float).eps
    H_model = model['H']
    
    # Broadcast h across H_model rows
    num = (H_model - h)**2
    den = H_model + h + eps
    Xi = 0.5 * np.sum(num / den, axis=1)
    
    # 3. Find K nearest neighbors
    idx_sorted = np.argsort(Xi)
    k_nn_idx = idx_sorted[:K]
    
    # 4. Features: min distance to each class in top K (MATLAB line 47-48)
    # Closest from negative class among top K? 
    # Actually MATLAB: Xi(~model.class(i(1:K)))
    k_classes = model['class'][k_nn_idx]
    k_distances = Xi[k_nn_idx]
    
    d = np.zeros(2)
    # Min distance to negative class samples in K-NN set
    neg_mask = (k_classes == 0)
    if np.any(neg_mask):
        d[0] = np.min(k_distances[neg_mask])
    else:
        d[0] = 0.0 # fallback if no negative in K-NN? 
        
    pos_mask = (k_classes == 1)
    if np.any(pos_mask):
        d[1] = np.min(k_distances[pos_mask])
    else:
        d[1] = 0.0
        
    l = k_classes
    c = stats.mode(l)[0][0] # Majority class in K-NN
    
    return c, d, l
