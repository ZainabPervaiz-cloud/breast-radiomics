"""
Python port of features_GLSM.m
Gray-level Sharpness Measure (GLSM) and Wavelet-based features.
"""
import numpy as np
import pywt
from skimage import filters
from scipy.ndimage import convolve
from skimage.transform import resize
import sys
import os

def features_GLSM(im, flist, mask=None):
    """
    Compute Gray-level sharpness and wavelet features.

    Parameters
    ----------
    im : ndarray
        Grayscale input image.
    flist : list
        List of feature names ('sgra', 'slap', 'swas', 'swav', 'swar', 'stev').
    mask : ndarray, optional
        Binary mask.

    Returns
    -------
    f : ndarray
        Column vector with computed features.
    """
    im = np.asarray(im, dtype=float)
    if mask is None:
        mask = np.ones_like(im, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    f = np.zeros(len(flist))
    eps = np.finfo(float).eps

    for n, feat in enumerate(flist):
        feat = feat.lower()
        
        if feat == 'sgra':
            # Sharpness gradient (MATLAB imgradient returns magnitude)
            # Sobel magnitude is common
            gx = filters.sobel(im)
            f[n] = np.mean(gx[mask])
            
        elif feat == 'slap':
            # Sharpness laplacian (MATLAB h = [-1 2 -1])
            h = np.array([[-1, 2, -1]])
            ly = convolve(im, h, mode='nearest')
            lx = convolve(im, h.T, mode='nearest')
            fx = np.abs(lx) + np.abs(ly)
            f[n] = np.mean(fx[mask])
            
        elif feat in ['swas', 'swav', 'swar']:
            # Wavelet-based features using db6
            coeffs = pywt.wavedec2(im, 'db6', level=3 if feat == 'swar' else 1)
            
            if feat == 'swas':
                # Sum of Wavelet Absolute Sub-bands (Level 1)
                coeffs1 = list(pywt.wavedec2(im, 'db6', level=1))
                cA1, d1 = coeffs1
                cH1, cV1, cD1 = d1
                
                # Reconstruct each sub-band separately
                def recon_sub(h, v, d):
                    return pywt.waverec2([np.zeros_like(cA1), (h, v, d)], 'db6')

                H = recon_sub(cH1, np.zeros_like(cV1), np.zeros_like(cD1))
                V = recon_sub(np.zeros_like(cH1), cV1, np.zeros_like(cD1))
                D = recon_sub(np.zeros_like(cH1), np.zeros_like(cV1), cD1)
                
                H = H[:im.shape[0], :im.shape[1]]
                V = V[:im.shape[0], :im.shape[1]]
                D = D[:im.shape[0], :im.shape[1]]
                
                fx = np.abs(H) + np.abs(V) + np.abs(D)
                f[n] = np.mean(fx[mask])
                
            elif feat == 'swav':
                # Wavelet Absolute Variance (Level 1)
                coeffs1 = list(pywt.wavedec2(im, 'db6', level=1))
                cA1, d1 = coeffs1
                cH1, cV1, cD1 = d1
                
                def recon_sub(h, v, d):
                    return pywt.waverec2([np.zeros_like(cA1), (h, v, d)], 'db6')

                H = recon_sub(cH1, np.zeros_like(cV1), np.zeros_like(cD1))
                V = recon_sub(np.zeros_like(cH1), cV1, np.zeros_like(cD1))
                D = recon_sub(np.zeros_like(cH1), np.zeros_like(cV1), cD1)
                
                H = H[:im.shape[0], :im.shape[1]]
                V = V[:im.shape[0], :im.shape[1]]
                D = D[:im.shape[0], :im.shape[1]]
                f[n] = np.var(H[mask], ddof=1) + np.var(V[mask], ddof=1) + np.var(D[mask], ddof=1)
                
            elif feat == 'swar':
                # Wavelet Area Ratio (Level 3)
                coeffs3 = pywt.wavedec2(im, 'db6', level=3)
                cA3 = coeffs3[0]
                d3 = [np.zeros_like(x) for x in coeffs3[1]]
                d2 = [np.zeros_like(x) for x in coeffs3[2]]
                d1 = [np.zeros_like(x) for x in coeffs3[3]]
                
                # H1 component
                d1_h = list(d1); d1_h[0] = coeffs3[3][0] # cH1
                H1 = pywt.waverec2([np.zeros_like(cA3), d3, d2, tuple(d1_h)], 'db6')
                
                # V1 component
                d1_v = list(d1); d1_v[1] = coeffs3[3][1] # cV1
                V1 = pywt.waverec2([np.zeros_like(cA3), d3, d2, tuple(d1_v)], 'db6')
                
                # D1 component
                d1_d = list(d1); d1_d[2] = coeffs3[3][2] # cD1
                D1 = pywt.waverec2([np.zeros_like(cA3), d3, d2, tuple(d1_d)], 'db6')
                
                # A components
                A1 = pywt.waverec2([cA3, coeffs3[1], coeffs3[2], tuple(d1)], 'db6')
                A2 = pywt.waverec2([cA3, coeffs3[1], tuple(d2), tuple(d1)], 'db6')
                A3 = pywt.waverec2([cA3, tuple(d3), tuple(d2), tuple(d1)], 'db6')

                H1 = H1[:im.shape[0], :im.shape[1]]
                V1 = V1[:im.shape[0], :im.shape[1]]
                D1 = D1[:im.shape[0], :im.shape[1]]
                A1 = A1[:im.shape[0], :im.shape[1]]
                A2 = A2[:im.shape[0], :im.shape[1]]
                A3 = A3[:im.shape[0], :im.shape[1]]
                
                WH = H1**2 + V1**2 + D1**2
                wh_mean = np.mean(WH[mask])
                wl_mean = np.mean(np.abs(A1[mask]) + np.abs(A2[mask]) + np.abs(A3[mask]))
                f[n] = (wh_mean + eps) / (wl_mean + eps)
                
        elif feat == 'stev':
            # Standard Deviation of Gradient
            gx = filters.sobel(im)
            f[n] = np.std(gx[mask], ddof=1)

        elif feat == 'suni': # uniformity
            gx = filters.sobel(im)
            c, _ = np.histogram(gx[mask], bins=256)
            p = c / np.sum(c)
            f[n] = np.sum(p**2)
            
        elif feat == 'ssmo': # smoothness
             gx = filters.sobel(im)
             f[n] = 1.0 / (1.0 + np.var(gx[mask], ddof=1))
             
        elif feat == 'sske': # skewness
             gx = filters.sobel(im)
             from scipy import stats
             f[n] = stats.skew(gx[mask], bias=False)
             
        elif feat == 'sent': # entropy
             gx = filters.sobel(im)
             c, _ = np.histogram(gx[mask], bins=256)
             p = c / np.sum(c); p = p[p>0]
             f[n] = -np.sum(p * np.log2(p))
             
        elif feat == 'sfdi': # fractal dimension
             psize = 0.4
             eps_vals = psize * np.array([1, 2, 4, 8, 16, 32, 64, 128])
             scales = psize * (1.0 / eps_vals)
             areas = []
             for scale in scales:
                 im_e = resize(im, (int(im.shape[0]*scale), int(im.shape[1]*scale)), preserve_range=True)
                 mask_e = resize(mask, (int(mask.shape[0]*scale), int(mask.shape[1]*scale)), order=0, preserve_range=True).astype(bool)
                 if np.any(mask_e):
                     diff_x = np.abs(np.diff(im_e, axis=1))
                     diff_y = np.abs(np.diff(im_e, axis=0))
                     # Adjust mask for diff
                     ax_val = np.sum(diff_x[mask_e[:, 1:]]) + np.sum(diff_y[mask_e[1:, :]])
                     # MATLAB line 89: A(n) = epsilon(n)^2 + epsilon(n)*sum(Ax(:));
                     # This is a crude area estimate.
                     areas.append(ax_val)
                 else:
                     areas.append(1e-8)
             
             p = np.polyfit(np.log(eps_vals), np.log(areas), 1)
             f[n] = -p[0]
        else:
             raise ValueError(f"Unknown feature {feat}")
             
    return f
