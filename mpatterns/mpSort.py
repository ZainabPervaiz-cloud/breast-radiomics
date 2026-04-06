"""
Python port of mpSort.m
Sorts micro-pattern prototypes according to Fisher's linear discriminant score.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def mpSort(P, C):
    """
    Sort prototypes by LDA score.

    Parameters
    ----------
    P : ndarray
        D x N array of prototypes.
    C : ndarray
        1 x N vector of class labels.
    """
    # Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(P.T, C)
    
    # Decision function (logit equivalent)
    scores = lda.decision_function(P.T)
    # Sort by ascending scores (MATLAB line 23)
    idx = np.argsort(scores)
    
    return P[:, idx]
