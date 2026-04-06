"""
Python port of xy2im.m
Map discrete scalar values at XY locations into an image using interpolation.
"""
import numpy as np
from scipy.interpolate import griddata
from skimage.transform import resize


def xy2im(x, y, f, im, mask):
    """
    Interpolate scalar values at XY points into an image.

    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates (0-based in Python).
    f : array-like
        Scalar values at (x, y).
    im : ndarray
        Base image to be resized and filled.
    mask : ndarray
        ROI mask.

    Returns
    -------
    imout : ndarray
        Interpolated image.
    """
    m_mask, n_mask = mask.shape
    
    # Resize base image to mask size
    # MATLAB: imresize(im, size(mask))
    im_resized = resize(im, (m_mask, n_mask), preserve_range=True)
    
    # Coordinates in mask
    # MATLAB uses 1-based indexing: y = floor(sc(1)*(y-1) + 1)
    # Our Python coordinates are already scaled if using modular components,
    # but let's match the logic if we assumed x,y were in original im scale.
    # If x, y are already in mask scale, then sc=[1,1].
    sc = np.array(mask.shape) / np.array(im.shape)
    y_idx = np.floor(sc[0] * np.asarray(y)).astype(int)
    x_idx = np.floor(sc[1] * np.asarray(x)).astype(int)

    # Grid for interpolation
    yi, xi = np.indices((m_mask, n_mask))
    
    # Create mask of points to keep (original mask minus the sampled points)
    # In MATLAB: mask(i) = false; im(i) = f;
    # Then it interpolates everywhere EXCEPT where mask is True? 
    # Wait, line 12: griddata(xi(~mask(:)), yi(~mask(:)), im(~mask(:)), xi, yi)
    # If mask is True, skip. If False, use.
    
    # Copy mask and unset at (y,x)
    working_mask = mask.copy()
    valid_coords = (y_idx >= 0) & (y_idx < m_mask) & (x_idx >= 0) & (x_idx < n_mask)
    working_mask[y_idx[valid_coords], x_idx[valid_coords]] = False
    
    working_im = im_resized.copy()
    working_im[y_idx[valid_coords], x_idx[valid_coords]] = f[valid_coords]
    
    # Points to interpolate FROM (where mask is False)
    train_y, train_x = np.where(~working_mask)
    train_vals = working_im[train_y, train_x]
    
    # Points to interpolate TO (everywhere)
    imout = griddata((train_x, train_y), train_vals, (xi, yi), method='linear', fill_value=0.0)
    
    return imout
