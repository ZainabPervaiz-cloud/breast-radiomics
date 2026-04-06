"""
Python port of ffdmForeground.m
Detect breast foreground in FFDM image using histogram analysis and morphology.
"""
import numpy as np
from scipy.signal import find_peaks, convolve
from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation, closing, disk, binary_erosion, binary_dilation, binary_closing
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from segmentation.getcontour import getcontour


def ffdmForeground(im, cflag=False):
    """
    Detect breast foreground in FFDM image.

    Parameters
    ----------
    im : ndarray
        MxN grayscale mammography image.
    cflag : bool
        True activates contour cutting using curvature. Default False.

    Returns
    -------
    mask : ndarray
        MxN binary mask with breast foreground.
    contour : dict
        Structure with contour data.
    """
    nbins = 1000
    im = np.asarray(im, dtype=float)

    # find intensity threshold
    xth = _getLower(im.ravel(), nbins)

    # find mask
    mask0 = (im >= max(xth, 0.0))

    # remove artifacts and holes in the mask
    mask = _cleanMask(mask0)

    # compute contour
    contour = getcontour(mask, cflag)
    contour['th'] = xth

    # Cut off points below the detected contour tip
    # matches mask(round(max(contour.y)):end,:) = false;
    y_cut = int(round(np.max(contour['y'])))
    mask[y_cut:, :] = False

    return mask, contour


def _getLower(x, nbins):
    x_min, x_max = x.min(), x.max()
    xi = np.linspace(x_min, x_max, nbins)
    
    # Histogram
    counts, _ = np.histogram(x, bins=nbins, range=(x_min, x_max))
    n = counts.astype(float)

    # Smooth histogram (gausswin 25)
    from scipy.signal import windows
    gw = windows.gaussian(25, 25/6) # Approximates MATLAB gausswin(25) sigma
    n = convolve(n, gw, mode='same')
    n = n / n.max()

    mean_x = np.mean(x)
    # xsup constraint (lines 61)
    xsup = min(mean_x, max(np.percentile(x, 30), 0.2 * x_max))

    # Find peaks below xsup
    mask_peaks = (xi <= xsup)
    ipeaks, _ = find_peaks(n * mask_peaks, height=0.35)

    def gauss1(x, a, b, c):
        return a * np.exp(-((x - b) / c)**2)

    if len(ipeaks) == 1:
        select = (n > 0.35) & (xi < xsup)
        sel_xi = xi[select]
        sel_n = n[select]
        if len(sel_xi) > 3:
            popt, _ = curve_fit(gauss1, sel_xi, sel_n, p0=[n[ipeaks[0]], xi[ipeaks[0]], (sel_xi.max()-sel_xi.min())/2])
            a1, b1, c1 = popt
            xth = b1 + np.sqrt(c1**2 * (np.log(a1) - np.log(0.05)))
        else:
            xth = xi[ipeaks[0]]
            
    elif len(ipeaks) > 1:
        # find minimum between peaks
        subset_n = n[min(ipeaks):max(ipeaks)+1]
        i_min_relative = np.argmin(subset_n)
        i_min = i_min_relative + min(ipeaks)
        
        select = (xi >= xi[i_min]) & (n > 0.35) & (xi < xsup)
        sel_xi = xi[select]
        sel_n = n[select]
        if len(sel_xi) > 3:
            ipeak2 = ipeaks[1] # Use second peak for fit initialization
            popt, _ = curve_fit(gauss1, sel_xi, sel_n, p0=[n[ipeak2], xi[ipeak2], (sel_xi.max()-sel_xi.min())/4])
            a1, b1, c1 = popt
            xth = b1 + np.sqrt(c1**2 * (np.log(a1) - np.log(0.05)))
        else:
            xth = xi[ipeaks[1]]
            
    else: # no peaks
        n_max = n.max()
        i_th = np.where(n < 0.05 * n_max)[0]
        if len(i_th) > 0:
            xth = xi[i_th[0]]
        else:
            xth = xi[0]

    return xth


def _cleanMask(mask0):
    mask0 = mask0.copy()
    mask0[0, :] = False
    mask0[-1, :] = False
    
    # Erode (MATLAB ones(5) is square 5x5)
    se = np.ones((5, 5))
    mask0 = binary_erosion(mask0, se)

    # Keep biggest region
    lbl = label(mask0)
    props = regionprops(lbl)
    if not props:
        return mask0
    
    biggest_idx = np.argmax([p.area for p in props])
    mask = (lbl == (biggest_idx + 1))

    # Morphological clean (dilate, close, fill)
    mask = binary_dilation(mask, se)
    mask = binary_closing(mask, se)
    mask = binary_fill_holes(mask)
    
    return mask
