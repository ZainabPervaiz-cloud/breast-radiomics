"""
Python port of features_GLCM.m
Gray-level Co-occurrence Matrix (GLCM) features.
"""
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def features_GLCM(im, flist, mask=None, par=None):
    """
    Compute GLCM texture features.

    Parameters
    ----------
    im : ndarray
        Grayscale input image.
    flist : list
        List of feature names ('cene', 'ccor', 'ccon', 'chom', 'cent').
    mask : ndarray, optional
        Binary mask.
    par : dict, optional
        Parameters: 'length' (default 1), 'nlevels' (default 128).

    Returns
    -------
    f : ndarray
        Feature vector.
    """
    if par is None:
        par = {'length': 1, 'nlevels': 128}

    im = np.asarray(im, dtype=float)
    if mask is None:
        mask = np.ones_like(im, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    # MATLAB line 31: GrayLimits [min, max]
    masked_data = im[mask]
    if len(masked_data) == 0:
        return np.zeros(len(flist))

    gmin = np.min(masked_data)
    gmax = np.max(masked_data)

    # Scale image to integers [0, nlevels-1] matching MATLAB graycomatrix behavior
    # MATLAB graycomatrix handles NaN by ignoring them if NumLevels is set and GrayLimits are provided.
    # In skimage, we should scale manually if we want exact parity with MATLAB's scaling.
    if gmax == gmin:
        im_scaled = np.zeros_like(im, dtype=int)
    else:
        im_scaled = np.round((im - gmin) / (gmax - gmin) * (par['nlevels'] - 1)).astype(int)
    
    # Set pixels outside mask to a value outside [0, nlevels-1] or handle them
    # skimage.feature.graycomatrix doesn't take a mask. 
    # Workaround: set outside to a special value and then crop the matrix.
    # Or more simply, since we want exact parity with the MATLAB logic which handles NaN:
    # MATLAB: im(~mask) = NaN.
    
    # Define offsets (MATLAB line 27)
    # [0 1; -1 1; -1 0; -1 -1] -> [ (0,1), (-1,1), (-1,0), (-1,-1) ]
    # In skimage, offsets are (row, col)
    # MATLAB offset [r c] means row shift r, col shift c.
    # [0 1] is right, [-1 1] is up-right, [-1 0] is up, [-1 -1] is up-left.
    offsets = [0, 1, 2, 3] # mapping to the 4 directions
    # Distance is par['length']
    sk_offsets = [par['length']]
    sk_angles = [0, -np.pi/4, -np.pi/2, -3*np.pi/4] # 0, 45, 90, 135 (negative because y-axis is down)
    
    # Actually let's use the explicit list of (r, c)
    # skimage graycomatrix(image, distances, angles)
    # We'll use a manual implementation if skimage doesn't support mask easily.
    # But wait, we can just mask the image by setting outside to nlevels and then ignoring that row/col.
    
    im_mask = im_scaled.copy()
    im_mask[~mask] = par['nlevels'] # Value outside the range
    
    g = graycomatrix(im_mask, distances=[par['length']], 
                     angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                     levels=par['nlevels'] + 1, 
                     symmetric=True, normed=False)
    
    # Slice to remove the 'extra' level
    g = g[:par['nlevels'], :par['nlevels'], :, :]

    f = np.zeros(len(flist))
    for n, feat in enumerate(flist):
        feat = feat.lower()
        if feat == 'cene': # Energy
            p = graycoprops(g, 'energy')
            f[n] = np.mean(p)
        elif feat == 'ccor': # Correlation
            p = graycoprops(g, 'correlation')
            f[n] = np.mean(p)
        elif feat == 'ccon': # Contrast
            p = graycoprops(g, 'contrast')
            f[n] = np.mean(p)
        elif feat == 'chom': # Homogeneity
            p = graycoprops(g, 'homogeneity')
            f[n] = np.mean(p)
        elif feat == 'cent': # Entropy
            # MATLAB lines 50-56
            # Normalize each GLCM
            for i in range(4): # for each direction
                gi = g[:, :, 0, i]
                total = np.sum(gi)
                if total > 0:
                    pi = gi / total
                    pi = pi[pi > 0]
                    E = -np.sum(pi * np.log2(pi))
                    f[n] += E
            f[n] /= 4.0
        else:
            raise ValueError(f"Unknown feature {feat}")

    return f
