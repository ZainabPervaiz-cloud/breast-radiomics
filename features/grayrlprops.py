"""
Python port of grayrlprops.m
Computes GLRL statistics from matrices.
"""
import numpy as np


def grayrlprops(glrlms, npixels):
    """
    Computes GLRL statistics.

    Parameters
    ----------
    glrlms : list
        List of GLRL matrices.
    npixels : int
        Total number of pixels in the region.

    Returns
    -------
    stats : dict
        Dict with 'SRE', 'LRE', 'GLN', 'RLN', 'RP', 'LGRE', 'HGRE'.
    """
    num_dirs = len(glrlms)
    SRE = np.zeros(num_dirs)
    LRE = np.zeros(num_dirs)
    GLN = np.zeros(num_dirs)
    RLN = np.zeros(num_dirs)
    RP = np.zeros(num_dirs)
    LGRE = np.zeros(num_dirs)
    HGRE = np.zeros(num_dirs)

    for n in range(num_dirs):
        p = glrlms[n]
        if p is None or np.sum(p) == 0:
            continue
            
        nl, max_len = p.shape
        # i is gray level (row index + 1), j is run length (col index + 1)
        j, i = np.meshgrid(np.arange(1, max_len + 1), np.arange(1, nl + 1))
        
        n_runs = np.sum(p)
        
        # 1. Short Run Emphasis (SRE)
        SRE[n] = (1.0 / n_runs) * np.sum(p / (j**2))
        
        # 2. Long Run Emphasis (LRE)
        LRE[n] = (1.0 / n_runs) * np.sum(p * (j**2))
        
        # 3. Gray-Level Nonuniformity (GLN)
        GLN[n] = (1.0 / n_runs) * np.sum(np.sum(p, axis=1)**2)
        
        # 4. Run Length Nonuniformity (RLN)
        RLN[n] = (1.0 / n_runs) * np.sum(np.sum(p, axis=0)**2)
        
        # 5. Run Percentage (RP)
        RP[n] = n_runs / npixels
        
        # 6. Low Gray-Level Run Emphasis (LGRE)
        LGRE[n] = (1.0 / n_runs) * np.sum(p / (i**2))
        
        # 7. High Gray-Level Run Emphasis (HGRE)
        HGRE[n] = (1.0 / n_runs) * np.sum(p * (i**2))

    return {
        'SRE': SRE,
        'LRE': LRE,
        'GLN': GLN,
        'RLN': RLN,
        'RP': RP,
        'LGRE': LGRE,
        'HGRE': HGRE
    }
