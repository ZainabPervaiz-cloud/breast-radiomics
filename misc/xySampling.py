"""
Python port of xySampling.m
Sample image regions using XY coordinates.
"""
import numpy as np


def xySampling(im, x, y, wsize, dispflag=False):
    """
    Sample image regions at given XY coordinates.

    Parameters
    ----------
    im : ndarray
        MxN grayscale image.
    x : array-like
        P-element vector of x-coordinates (1-based, like MATLAB).
    y : array-like
        P-element vector of y-coordinates (1-based, like MATLAB).
    wsize : int
        Window size of each region.
    dispflag : bool
        If True, display sampled regions. Default False.

    Returns
    -------
    f : ndarray
        Px1 vector of intensity values at each (x,y).
    r : list of ndarray
        Px1 list of image patches (only returned when needed).
    mask : ndarray
        MxN label mask (region index at each pixel).
    """
    im = np.asarray(im, dtype=float)
    # Convert to 0-based indexing
    x = np.asarray(x, dtype=int) - 1
    y = np.asarray(y, dtype=int) - 1

    delta = int(np.floor(0.5 * wsize))
    nrois = len(x)
    mask = np.zeros(im.shape, dtype=int)

    r = []
    f = np.zeros(nrois)

    for n in range(nrois):
        xi, yi = int(x[n]), int(y[n])
        patch = im[yi - delta: yi + delta + 1, xi - delta: xi + delta + 1]
        r.append(patch)
        f[n] = im[yi, xi]
        mask[yi - delta: yi + delta + 1, xi - delta: xi + delta + 1] = n + 1  # 1-based label

    if dispflag:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots()
        ax.imshow(im, cmap='gray')
        for n in range(nrois):
            xi, yi = int(x[n]), int(y[n])
            rect = mpatches.Rectangle(
                (xi - delta, yi - delta), wsize, wsize,
                linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.plot(xi, yi, 'r+')
        plt.show()

    return f, r, mask
