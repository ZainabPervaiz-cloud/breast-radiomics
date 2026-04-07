"""
Python port of showseg.m
Overlay segmentation mask on image.
"""
import numpy as np
from skimage.morphology import binary_dilation, square
from skimage.segmentation import find_boundaries


def showseg(im, mask, alpha=0, color=None):
    """
    Overlay segmentation on image.

    Parameters
    ----------
    im : ndarray
        MxN grayscale or MxNx3 RGB image.
    mask : ndarray
        MxN binary segmentation mask.
    alpha : float
        Overlay alpha. 0 = edges only.
    color : str or array-like
        'r', 'g', 'b', or [R, G, B] (0-255). Default: red [255, 0, 0].

    Returns
    -------
    imout : ndarray
        MxNx3 uint8 image with overlay.
    """
    border = 4  # border width in pixels

    if color is None:
        # Default to Green [0, 255, 0] if alpha > 0, to match MATLAB's showseg default
        color = [0, 255, 0] if alpha > 0 else [255, 0, 0]
    elif color == 'r':
        color = [255, 0, 0]
    elif color == 'g':
        color = [0, 255, 0]
    elif color == 'b':
        color = [0, 0, 255]
    color = np.array(color, dtype=np.uint8)

    # Resize image to match mask if needed
    if im.shape[:2] != mask.shape[:2]:
        from skimage.transform import resize
        im = resize(im, mask.shape[:2], preserve_range=True).astype(im.dtype)

    # Convert to double then uint8
    im = np.asarray(im, dtype=float)
    im_min, im_max = im.min(), im.max()
    if im_max > im_min:
        im_n = ((im - im_min) / (im_max - im_min) * 255).astype(np.uint8)
    else:
        im_n = np.zeros_like(im, dtype=np.uint8)

    mask = np.asarray(mask, dtype=bool)

    # Build RGB planes
    if im_n.ndim == 3:
        imR = im_n[:, :, 0].copy()
        imG = im_n[:, :, 1].copy()
        imB = im_n[:, :, 2].copy()
    else:
        imR = im_n.copy()
        imG = im_n.copy()
        imB = im_n.copy()

    if alpha == 0 or alpha is None:
        # Show mask edges only (border dilation)
        perim = find_boundaries(mask, mode='outer')
        perim = binary_dilation(perim, square(border))
        imR[perim] = color[0]
        imG[perim] = color[1]
        imB[perim] = color[2]
    else:
        # Colored overlay
        c = color
        if np.array_equal(c, [0, 255, 0]):  # green
            imG[mask] = np.clip(imG[mask].astype(int) + int(alpha * 255), 0, 255).astype(np.uint8)
        elif np.array_equal(c, [255, 0, 0]):  # red
            imR[mask] = np.clip(imR[mask].astype(int) + int(alpha * 255), 0, 255).astype(np.uint8)
        elif np.array_equal(c, [0, 0, 255]):  # blue
            imB[mask] = np.clip(imB[mask].astype(int) + int(alpha * 255), 0, 255).astype(np.uint8)
        else:
            imR[mask] = np.clip(imR[mask].astype(int) + int(alpha * c[0]), 0, 255).astype(np.uint8)
            imG[mask] = np.clip(imG[mask].astype(int) + int(alpha * c[1]), 0, 255).astype(np.uint8)
            imB[mask] = np.clip(imB[mask].astype(int) + int(alpha * c[2]), 0, 255).astype(np.uint8)

    imout = np.stack([imR, imG, imB], axis=2)
    return imout
