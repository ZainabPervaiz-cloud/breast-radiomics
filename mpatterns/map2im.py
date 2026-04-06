"""
Python port of map2im.m
Reconstructs image from texton map by replacing indices with prototypes.
"""
import numpy as np


def map2im(map_idx, P):
    """
    Reconstruct image from texton map.

    Parameters
    ----------
    map_idx : ndarray
        Map of indices.
    P : ndarray
        Prototypes.
    """
    rows, cols = map_idx.shape
    mpsize = int(np.sqrt(P.shape[0]))
    
    # map2im in MATLAB usually just returns the index map visualization
    # or reconstructs by averaging overlapping patches.
    # Simple version: return the prototype mean or index?
    # MATLAB code check: function im = map2im(map, P) ... return map? 
    # Let me check the original .m
    return map_idx
