"""
Python port of features_FDIM.m
Fractal Dimension (FD) features: Box-counting and Spectral Beta.
"""
import numpy as np
from skimage.morphology import erosion
from skimage.transform import resize
from scipy.fftpack import fft2, fftshift
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc.sqmax import sqmax


def features_FDIM(im, mask=None, psize=0.4, ftype='boxc'):
    """
    Compute fractal dimension features.

    Parameters
    ----------
    im : ndarray
        Grayscale input image.
    mask : ndarray, optional
        Binary mask.
    psize : float
        Pixel size in mm.
    ftype : str
        'boxc' for box counting or 'beta' for spectral power coefficient.

    Returns
    -------
    f : float
        Fractal dimension.
    """
    im = np.asarray(im, dtype=float)
    if mask is None:
        mask = np.ones_like(im, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    if ftype == 'boxc':
        # MATLAB line 32: erode with ones(2)
        mask_e = erosion(mask, np.ones((2, 2)))
        Lmax = max(im.shape)
        nmax = int(np.floor(np.log2(Lmax)) - 2)
        n_vals = np.arange(0, min(8, nmax + 1)) # MATLAB 0:1:min([7,nmax])
        epsilon = psize * (2.0 ** n_vals)
        scales = psize * (1.0 / epsilon)
        
        A = np.zeros(len(epsilon))
        for i, scale in enumerate(scales):
            # Scale image and mask
            new_size = (int(round(im.shape[0] * scale)), int(round(im.shape[1] * scale)))
            if new_size[0] < 2 or new_size[1] < 2:
                A[i] = 1e-8
                continue
            im_e = resize(im, new_size, preserve_range=True)
            mask_ei = resize(mask_e, new_size, order=0, preserve_range=True).astype(bool)
            
            # MATLAB line 45-46: diff shifts
            # diff(im_e, 1, 2) is horizontal diff
            # diff_x = [diff(im_e, 1, 2), zeros(size(im_e, 1), 1)];
            dx = np.abs(np.diff(im_e, axis=1))
            dy = np.abs(np.diff(im_e, axis=0))
            
            # Mask indexing needs to be adjusted for diff
            # MATLAB logic: Ax = abs(diff_x(mask_e)) + abs(diff_y(mask_e));
            # This is sum of absolute gradients within the mask.
            ax_sum = np.sum(dx[mask_ei[:, :-1]]) + np.sum(dy[mask_ei[:-1, :]])
            
            A[i] = (epsilon[i]**2) + (epsilon[i] * ax_sum)

        # Log-log regression (MATLAB line 52)
        p = np.polyfit(np.log(epsilon + 1e-12), np.log(A + 1e-12), 1)
        return -p[0]

    elif ftype == 'beta':
        # Spectral Beta method (MATLAB line 54)
        nbands = 16
        # sc = mean(size(mask)/size(im)); No, MATLAB size(mask) is same as size(im)
        # security ring of 1cm (10mm)
        dx = int(round(10.0 / psize))
        
        # Border cleanup (MATLAB line 58-61)
        mask_c = mask.copy()
        mask_c[0, :] = False
        mask_c[-1, :] = False
        mask_c[:, 0] = False
        mask_c[:, -1] = False
        
        # Erosion with disk (MATLAB line 62)
        from skimage.morphology import disk
        mask_c = erosion(mask_c, disk(dx))
        
        # Extract largest square (MATLAB line 63)
        _, rect = sqmax(mask_c)
        # rect is [x, y, w, h]
        x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        if w < 4:
            return 0.0
            
        imc = im[y:y+h+1, x:x+w+1]
        L = imc.shape[0] - 1
        
        # Hanning window (MATLAB line 67)
        win1d = np.hanning(L + 1)
        win2d = np.outer(win1d, win1d)
        win2d = win2d / np.max(win2d)
        
        imc_win = (imc - np.mean(imc)) * win2d
        
        # FFT2 (MATLAB line 72)
        F = fftshift(fft2(imc_win))
        P = np.abs(F) ** 2
        
        # Frequency grid
        fs = 1.0 / psize
        freqs = np.linspace(-0.5 * fs, 0.5 * fs, L + 1)
        u, v = np.meshgrid(freqs, freqs)
        f_radial = np.sqrt(u**2 + v**2)
        
        flim = np.linspace(0, 0.5 * fs, nbands + 1)
        Pn = np.zeros(nbands)
        fn = np.zeros(nbands)
        
        for n in range(nbands):
            select = (f_radial < flim[n+1]) & (f_radial > flim[n])
            if np.any(select):
                Pn[n] = np.mean(P[select])
                fn[n] = 0.5 * (flim[n] + flim[n + 1])
        
        # Log-log regression (MATLAB line 86)
        valid = (Pn > 0) & (fn > 0)
        if np.sum(valid) < 2:
            return 0.0
        p = np.polyfit(np.log(fn[valid]), np.log(Pn[valid]), 1)
        return -p[0]
    
    return 0.0
