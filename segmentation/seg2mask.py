"""
Python port of seg2mask.m
Convert segmentation data (contour, cwall) to binary masks.
"""
import numpy as np
from skimage.draw import polygon2mask
from skimage.morphology import erosion, dilation
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def seg2mask(contour, cpts=None, rpoints=None, cgap=None, sgap=None):
    """
    Convert segmentation data to masks.

    Parameters
    ----------
    contour : dict
        Structure with 'x', 'y' and 'size'.
    cpts : ndarray, optional
        100x2 array with [x, y] coordinates of chest wall points.
    rpoints : dict, optional
        Reference points.
    cgap : int, optional
        Chest wall gap.
    sgap : int, optional
        Skin gap.

    Returns
    -------
    mask1 : ndarray
        MxN full breast mask.
    mask2 : ndarray, optional
        MxN RA region mask (only if rpoints provided).
    """
    m, n = contour['size']
    xs = contour['x']
    ys = contour['y']

    # Close the contour by adding points (MATLAB lines 36-37)
    # x = [0; x(1); x(:); x(end); 0];
    # y = [min(y); min(y); y(:);  y(end); y(end)];
    poly_x = np.concatenate([[0], [xs[0]], xs, [xs[-1]], [0]])
    poly_y = np.concatenate([[np.min(ys)], [np.min(ys)], ys, [ys[-1]], [ys[-1]]])
    
    # polygon2mask expects (y, x)
    points = np.column_stack([poly_y, poly_x])
    mask1 = polygon2mask((m, n), points)

    # Apply skin gap (MATLAB lines 41-43)
    if sgap is not None and sgap != 0:
        mask1 = erosion(mask1, np.ones((sgap, sgap)))

    # Chest wall handling (MATLAB lines 46-60)
    if cpts is not None:
        cx = cpts[:, 0]
        cy = cpts[:, 1]
        poly_cx = np.concatenate([[0], cx])
        poly_cy = np.concatenate([[0], cy])
        c_points = np.column_stack([poly_cy, poly_cx])
        mask2_c = polygon2mask((m, n), c_points)
        mask2_c[:, :2] = True # matches mask2(1:end,1:2) = true;

        if cgap is not None and cgap != 0:
            mask2_c = dilation(mask2_c, np.ones((cgap, cgap)))
        
        mask1 = mask1 & (~mask2_c)

    # RA mask (requires mapping module)
    # Since Phase 3 (mapping) isn't implemented yet, we leave this as a check
    # But for parity, it should attempt to call stmap and st2mask if needed.
    mask_ra = None
    if rpoints is not None:
        # Local imports to avoid circular dependency if Phase 3 uses this
        from mapping.stmap import stmap
        from mapping.st2mask import st2mask
        
        # Adjust reference points (MATLAB lines 67-75)
        for pt_key in ['p0', 'p1', 'p2']:
            pt = rpoints[pt_key]
            dist = np.sqrt((pt['x'] - xs)**2 + (pt['y'] - ys)**2)
            idx = np.argmin(dist)
            rpoints[pt_key]['x'] = xs[idx]
            rpoints[pt_key]['y'] = ys[idx]
        
        slims = [.2, .8]
        tlims = [.1, .9]
        mapp = stmap(contour, rpoints)
        mask_ra = st2mask(mapp, slims, tlims)
        
        # remove borders
        mask_ra[:, 0] = False
        mask_ra[:, -1] = False
        mask_ra[0, :] = False
        mask_ra[-1, :] = False

    # Remove borders for mask1
    mask1[:, 0] = False
    mask1[:, -1] = False
    mask1[0, :] = False
    mask1[-1, :] = False

    if contour.get('flip', False):
        mask1 = np.fliplr(mask1)
        if mask_ra is not None:
            mask_ra = np.fliplr(mask_ra)

    if rpoints is not None:
        return mask1, mask_ra
    else:
        return mask1
