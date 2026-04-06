"""
Python port of mpSampling.m
Random micro-pattern sampling of mammography images.
"""
import numpy as np
from skimage.transform import resize
from skimage.morphology import erosion
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc.getinfo import getinfo
from misc.ffdmRead import ffdmRead
from segmentation.seg2mask import seg2mask


def mpSampling(imdata, params):
    """
    Random micro-pattern sampling.

    Parameters
    ----------
    imdata : dict or list of dict
        Image data structure(s).
    params : dict
        'nsamples', 'mpsize', 'rflag', 'nflag'.
    """
    resolution = 0.05
    
    # Recursive case (MATLAB line 32)
    if isinstance(imdata, list):
        no_images = len(imdata)
        M_all = np.zeros((params['mpsize']**2, params['nsamples'] * no_images))
        C_all = np.zeros(params['nsamples'] * no_images, dtype=bool)
        
        for n in range(no_images):
            m0 = n * params['nsamples']
            m1 = (n + 1) * params['nsamples']
            M_sub, C_sub = mpSampling(imdata[n], params)
            M_all[:, m0:m1] = M_sub
            C_all[m0:m1] = C_sub
        return M_all, C_all

    # Single image case
    # Load image
    info = getinfo(imdata['path'])
    im_norm, _ = ffdmRead(imdata['path'], info)
    
    # Re-scale to 0.05mm
    scale = info['psize'] / resolution
    im_res = resize(im_norm, (int(round(im_norm.shape[0]*scale)), int(round(im_norm.shape[1]*scale))), preserve_range=True)
    
    # Load mask
    mask, _ = seg2mask(imdata['contour'], imdata['cwall'])
    mask_res = resize(mask, im_res.shape, order=0, preserve_range=True).astype(bool)
    
    # Erode mask (MATLAB line 66)
    mask_res = erosion(mask_res, np.ones((params['mpsize'] + 1, params['mpsize'] + 1)))
    
    # Find valid indici
    idx = np.where(mask_res.flatten())[0]
    if len(idx) < params['nsamples']:
        # if not enough pixels, just take what we have
        indices = idx
    else:
        # Fixed seed for parity (MATLAB line 72: rng(1))
        np.random.seed(1)
        indices = np.random.choice(idx, params['nsamples'], replace=False)
    
    # Sub-indices
    y_idx, x_idx = np.unravel_index(indices, mask_res.shape)
    
    M = np.zeros((params['mpsize']**2, len(indices)))
    delta = params['mpsize'] // 2
    
    for n in range(len(indices)):
        # Patch extraction
        patch = im_res[y_idx[n]-delta : y_idx[n]+delta+1, x_idx[n]-delta : x_idx[n]+delta+1]
        if patch.size == params['mpsize']**2:
            M[:, n] = patch.flatten()
        else:
            # handle border cases if erosion missed something
            pass
            
    # Webber's law / Normalization (MATLAB line 87)
    if params.get('nflag', False):
        m_sq_sum = np.sqrt(np.sum(M**2, axis=0))
        # MATLAB: M = M.*log(1+C/0.03)./(C+eps);
        eps = np.finfo(float).eps
        M = M * np.log(1.0 + m_sq_sum / 0.03) / (m_sq_sum + eps)
        
    C = np.full(M.shape[1], imdata['class'], dtype=bool)
    
    return M, C
