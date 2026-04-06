"""
Python port of bestStump.m
Computes the best accuracy of a decision stump for a single feature.
"""
import numpy as np


def bestStump(x, y, dispflag=False):
    """
    Find best accuracy of a decision stump.
    
    Parameters
    ----------
    x : ndarray
        Feature values.
    y : ndarray
        Labels.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # MATLAB: xth = linspace(min(x), max(x));
    # 100 points by default in linspace
    xth = np.linspace(np.min(x), np.max(x), 100)
    
    best_acc = 0.0
    for th in xth:
        # MATLAB: sum(double(x>xth(n))==double(y))/numel(x);
        # We should check both directions: x > th and x < th
        acc1 = np.sum((x > th) == y) / len(x)
        acc2 = np.sum((x < th) == y) / len(x)
        best_acc = max(best_acc, acc1, acc2)
        
    return best_acc
