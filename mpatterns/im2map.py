"""
Python port of im2map.m
Converts image to texton map by assigning each pixel to the closest prototype.
"""
import numpy as np
from skimage.transform import resize
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpatterns.mpSampling import mpSampling


def im2map(im, P, params):
    """
    Convert image to texton map.

    Parameters
    ----------
    im : ndarray
    P : ndarray
        Prototypes.
    params : dict
    """
    mpsize = params['mpsize']
    nprototypes = P.shape[1]
    
    # We need to process EVERY pixel in the image as a patch
    # This is compute intensive.
    # MATLAB uses a tiled approach or filtering.
    # Here, we'll use a sliding window view.
    
    rows, cols = im.shape
    delta = mpsize // 2
    
    # Padding to keep output same size
    padded = np.pad(im, delta, mode='edge')
    
    # Create sliding windows
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, (mpsize, mpsize))
    
    # windows shape is (rows, cols, mpsize, mpsize)
    # Reshape to (rows*cols, mpsize^2)
    flat_patches = windows.reshape(-1, mpsize**2).T # (mpsize^2) x (rows*cols)
    
    # Optional Webber's law
    if params.get('nflag', False):
        m_sq_sum = np.sqrt(np.sum(flat_patches**2, axis=0))
        eps = np.finfo(float).eps
        flat_patches = flat_patches * np.log(1.0 + m_sq_sum / 0.03) / (m_sq_sum + eps)

    # Distances to prototypes
    # flat_patches is (D, N_pix), P is (D, N_prot)
    # dist^2 = S2 + P2 - 2*P'S
    S2 = np.sum(flat_patches**2, axis=0) # 1 x N_pix
    P2 = np.sum(P**2, axis=0) # 1 x N_prot
    
    dist_sq = S2[np.newaxis, :] + P2[:, np.newaxis] - 2 * np.dot(P.T, flat_patches)
    dist_sq[dist_sq < 0] = 0
    
    map_idx = np.argmin(dist_sq, axis=0)
    return map_idx.reshape(rows, cols)
