"""
Python port of showmp.m
Visualizes micro-pattern prototypes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def showmp(P, C, psize, save_path=None):
    """
    Visualize MP prototypes.

    Parameters
    ----------
    P : ndarray
        D x N matrix of prototypes.
    C : ndarray
        1 x N class labels.
    psize : tuple
        (nx, ny) grid dimensions.
    save_path : str, optional
        Path to save the visualization.
    """
    L = 700
    edge = 10
    
    side_y = int(np.floor((L / 2 - (psize[1] + 1) * edge) / psize[1]))
    side_x = int(np.floor((L - (psize[0] + 1) * edge) / psize[0]))
    side = min(side_x, side_y)
    
    LX = side * psize[0] + edge * (psize[0] + 1)
    LY = side * psize[1] + edge * (psize[1] + 1)
    
    def build_grid(P_sub):
        grid = 0.9 * np.ones((LY, LX))
        npix = int(np.sqrt(P_sub.shape[0]))
        k = 0
        for m in range(psize[1]):
            yo = edge * (m + 1) + side * m
            yf = yo + side
            for n in range(psize[0]):
                if k >= P_sub.shape[1]: break
                xo = edge * (n + 1) + side * n
                xf = xo + side
                patch = P_sub[:, k].reshape(npix, npix)
                # Rescale patch to [0, 1]
                p_min, p_max = np.min(patch), np.max(patch)
                if p_max > p_min:
                    patch = (patch - p_min) / (p_max - p_min)
                else:
                    patch = np.zeros_like(patch)
                grid[yo:yf, xo:xf] = resize(patch, (side, side), preserve_range=True)
                k += 1
        return grid

    im_up = build_grid(P[:, ~C])
    im_do = build_grid(P[:, C])
    
    im_total = np.vstack([im_up, im_do])
    
    if save_path:
        plt.figure(figsize=(10, 10))
        plt.imshow(im_total, cmap='gray')
        plt.axis('off')
        plt.title('Micro-pattern Prototypes (Top: Neg, Bottom: Pos)')
        plt.savefig(save_path)
        plt.close()
        
    return im_total
