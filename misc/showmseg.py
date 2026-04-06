"""
Python port of showmseg.m
Display manual segmentation result overlaid on image.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc.getinfo import getinfo
from misc.ffdmRead import ffdmRead
from misc.showseg import showseg
from segmentation.seg2mask import seg2mask
from skimage.morphology import remove_small_objects
from skimage.transform import resize


def showmseg(segdata):
    """
    Display manual segmentation overlaid on mammogram.

    Parameters
    ----------
    segdata : dict
        Segmentation data with keys:
        'path', 'contour', 'cwall', 'gap1', 'gap2', 'th1', 'th2'

    Returns
    -------
    ims : ndarray
        RGB image with overlay.
    mask1 : ndarray
        Dense tissue mask.
    pd : float
        Percent density.
    """
    impath = segdata['path'][0]
    info = getinfo(impath)
    imn, _ = ffdmRead(impath, info)

    mask0 = seg2mask(segdata['contour'][0], segdata['cwall'][0])
    maske = seg2mask(
        segdata['contour'][0],
        segdata['cwall'][0],
        gap1=segdata.get('gap1'),
        gap2=segdata.get('gap2')
    )

    mask1 = imn > segdata['th1']
    mask1 = resize(mask1, maske.shape, order=0, preserve_range=True).astype(bool)
    mask1 = mask1 & maske
    mask1 = remove_small_objects(mask1, min_size=segdata['th2'])

    pd = mask1.sum() / max(mask0.sum(), 1)

    # Enhance contrast (MATLAB: adapthisteq)
    im_eq = equalize_adapthist(imn)
    ims = showseg(im_eq, mask0)
    ims = showseg(ims, mask1, alpha=0.25)

    return ims, mask1, pd
