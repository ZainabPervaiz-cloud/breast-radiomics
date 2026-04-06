"""
Python port of ffdmFeatures.m
Top-level feature extraction pipeline with multi-scale and multi-sampling.
"""
import numpy as np
from skimage.transform import resize
from skimage.morphology import erosion
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc.getinfo import getinfo
from misc.ffdmRead import ffdmRead
from misc.isright import isright
from misc.sqmax import sqmax
from misc.rcenters import rcenters
from misc.xySampling import xySampling
from segmentation.segBreast import segBreast
from mapping.refpoints import refpoints
from mapping.stmap import stmap
from mapping.st2mask import st2mask
from features.xfeatures import xfeatures


def ffdmFeatures(impath, res=0.07, scales=None, normal=None, sampl=None):
    """
    Extract full set of clinical features from FFDM.

    Parameters
    ----------
    impath : str
        Path to DICOM.
    res : float
        Analysis resolution in mm/pixel.
    scales : list of str
        ['1.0', '0.5', '0.25'].
    normal : list of str
        ['none', 'zscore'].
    sampl : list of str
        ['full', 'RA', 'multi', 'SQ'].
    """
    if scales is None: scales = ['1.0', '0.5', '0.25']
    if normal is None: normal = ['none', 'zscore']
    if sampl is None: sampl = ['full', 'RA', 'multi', 'SQ']

    flist = ['imin', 'imax', 'iavg', 'ient', 'istd', 'ip05', 'ip95', 'iba1', 'iba2', 'ip30', 'ip70', 'iske', 'ikur', 'iran',
             'cene', 'ccor', 'ccon', 'chom', 'cent',
             'rsre', 'rlre', 'rgln', 'rrpe', 'rrln', 'rlgr', 'rhgr',
             'sgra', 'slap', 'swas', 'swav', 'swar', 'stev', 'fdim']

    # 1. Image loading
    image_info = getinfo(impath)
    im_norm, im_raw = ffdmRead(impath, image_info)
    
    if image_info['israw']:
        im = im_raw # process raw if requested? MATLAB logic: im=imn in raw case?
        # Re-check MATLAB line 61: if israw, imr = imresize(im, ...), im = imn;
        # It seems it always uses normalized image 'imn' for features.
    im = im_norm
    
    psize = image_info['psize']
    im_target = resize(im, (int(round(im.shape[0] * psize / res)), int(round(im.shape[1] * psize / res))), preserve_range=True)

    # 2. Base Masks
    # segBreast on 0.25 scale for speed (MATLAB line 70)
    im_small = resize(im_norm, (int(im_norm.shape[0] * 0.25), int(im_norm.shape[1] * 0.25)), preserve_range=True)
    mask_small, contour_small, cwall_small = segBreast(im_small, image_info['ismlo'])
    
    mask0 = resize(mask_small, im_target.shape, order=0, preserve_range=True).astype(bool)
    mask0 = erosion(mask0, np.ones((31, 31))) # MATLAB line 76

    # 3. Mapping
    rpts = refpoints(contour_small, cwall_small, image_info['ismlo'])
    mapp = stmap(contour_small, rpts)
    mask_RA0 = st2mask(mapp, [0.1, 0.5], [0.1, 0.9])
    mask_RA0 = resize(mask_RA0, im_target.shape, order=0, preserve_range=True).astype(bool)
    
    mask_SQ0, _ = sqmax(mask0)

    # 4. Flip side (MATLAB line 93)
    if isright(im_norm):
        im_target = np.fliplr(im_target)
        mask0 = np.fliplr(mask0)
        mask_RA0 = np.fliplr(mask_RA0)
        mask_SQ0 = np.fliplr(mask_SQ0)

    # 5. Pipeline
    no_feats = len(flist) * len(scales) * len(sampl) * len(normal)
    f_total = np.zeros(no_feats)
    f_info = {'fnames': [], 'scal': [], 'norm': [], 'samp': [], 'eflag': 'success'}

    k = 0
    for nn in normal:
        if nn.lower() == 'none':
            im_n = im_target.copy()
        elif nn.lower() == 'zscore':
            masked_data = im_target[mask0]
            mu = np.mean(masked_data); std = np.std(masked_data, ddof=1)
            im_n = (im_target - mu) / (std + 1e-12)
        
        imin = np.min(im_n); imax = np.max(im_n)
        
        for ns_val in scales:
            scale_f = float(ns_val)
            if scale_f == 1.0:
                im_x = im_n
            else:
                im_x = resize(im_n, (int(round(im_n.shape[0]*scale_f)), int(round(im_n.shape[1]*scale_f))), preserve_range=True)
                im_x[np.isnan(im_x)] = imin
                im_x = np.clip(im_x, imin, imax)
            
            m_full = resize(mask0, im_x.shape, order=0, preserve_range=True).astype(bool)
            m_ra = resize(mask_RA0, im_x.shape, order=0, preserve_range=True).astype(bool)
            m_sq = resize(mask_SQ0, im_x.shape, order=0, preserve_range=True).astype(bool)
            
            for s_type in sampl:
                k0 = k * len(flist)
                k1 = (k + 1) * len(flist)
                k += 1
                
                f_info['fnames'].extend(flist)
                f_info['norm'].extend([nn] * len(flist))
                f_info['scal'].extend([ns_val] * len(flist))
                f_info['samp'].extend([s_type] * len(flist))
                
                if s_type.lower() == 'full':
                    f_total[k0:k1] = xfeatures(im_x, flist, m_full, res)
                elif s_type.lower() == 'ra':
                    f_total[k0:k1] = xfeatures(im_x, flist, m_ra, res)
                elif s_type.lower() == 'sq':
                    f_total[k0:k1] = xfeatures(im_x, flist, m_sq, res)
                elif s_type.lower() == 'multi':
                    x_c, y_c = rcenters(128, 63, m_full)
                    _, r_patches, _ = xySampling(im_x, x_c, y_c, 63)
                    if len(r_patches) > 0:
                        patch_feats = xfeatures(r_patches, flist, None, res)
                        f_total[k0:k1] = np.nanmean(patch_feats, axis=0)
                    else:
                        f_total[k0:k1] = 0.0

    return f_total, f_info
