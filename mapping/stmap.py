"""
Python port of stmap.m
Compute parameters of the polar-based (ST) mapping system for breast mammography.
"""
import numpy as np


def stmap(contour, rpoints):
    """
    Compute parameters of ST mapping.

    Parameters
    ----------
    contour : dict
        Breast contour data ('x', 'y', 'size').
    rpoints : dict
        Reference points ('p0', 'p1', 'p2').

    Returns
    -------
    mapp : dict
        Mapping parameters.
    """
    d_degree = 5
    xp = np.asarray(contour['x'], dtype=float)
    yp = np.asarray(contour['y'], dtype=float)
    x0 = float(rpoints['p0']['x'])
    y0 = float(rpoints['p0']['y'])

    # Find indices of reference points
    i0 = np.argmin(np.sqrt((xp - x0)**2 + (yp - y0)**2))
    i1 = np.argmin(np.sqrt((xp - rpoints['p1']['x'])**2 + (yp - rpoints['p1']['y'])**2))
    i2 = np.argmin(np.sqrt((xp - rpoints['p2']['x'])**2 + (yp - rpoints['p2']['y'])**2))

    # Auxiliary curves (MATLAB lines 51-54)
    # Using step -1 for i0:-1:i1 in Python slice: i0 to i1-1 steps of -1
    if i0 >= i1:
        x_phy0 = xp[i0:i1-1:-1] if i1 > 0 else xp[i0::-1]
        y_phy0 = yp[i0:i1-1:-1] if i1 > 0 else yp[i0::-1]
    else:
        # Case where indices are inverted
        x_phy0 = xp[i0:i1+1]
        y_phy0 = yp[i0:i1+1]

    x_phy1 = xp[i0:i2+1]
    y_phy1 = yp[i0:i2+1]

    # Surface distance (normalized to [0,1])
    s0 = np.cumsum(np.sqrt(np.diff(x_phy0)**2 + np.diff(y_phy0)**2))
    s1 = np.cumsum(np.sqrt(np.diff(x_phy1)**2 + np.diff(y_phy1)**2))
    
    # Prepend 0 if not empty
    s0 = np.concatenate(([0.0], s0))
    s1 = np.concatenate(([0.0], s1))
    
    max_s0 = np.max(s0) if len(s0) > 0 else 1.0
    max_s1 = np.max(s1) if len(s1) > 0 else 1.0
    s0 = s0 / (max_s0 + 1e-12)
    s1 = s1 / (max_s1 + 1e-12)

    # Fit polynomials (MATLAB pfit: no constant term)
    def pfit(x, y, n):
        if len(x) < 2:
            return np.zeros(n)
        # V = [x^n, x^{n-1}, ..., x^1]
        V = np.column_stack([x**(n-i) for i in range(n)])
        P, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        return P

    # Px/Py are coefficients from highest power to lowest (but constant is added as 0 separately)
    # Note: MATLAB [pfit; 0] makes the last element (constant term) 0.
    # In Python np.polyval, the last element is the constant term.
    Px0 = np.concatenate([pfit(s0, x_phy0 - x0, d_degree), [0.0]])
    Px1 = np.concatenate([pfit(s1, x_phy1 - x0, d_degree), [0.0]])
    Py0 = np.concatenate([pfit(s0, y_phy0 - y0, d_degree), [0.0]])
    Py1 = np.concatenate([pfit(s1, y_phy1 - y0, d_degree), [0.0]])

    # Extreme angles (MATLAB line 69-70)
    theta0 = np.arctan2(np.polyval(Py0, 1.0), np.polyval(Px0, 1.0)) + 2.0 * np.pi
    theta1 = np.arctan2(np.polyval(Py1, 1.0), np.polyval(Px1, 1.0))

    # Coefficients
    c0, s0c = np.cos(-theta0), np.sin(-theta0)
    c1, s1c = np.cos(-theta1), np.sin(-theta1)

    mapp = {
        'P1': c1 * Px1 - s1c * Py1 - c0 * Px0 + s0c * Py0,
        'P2': c0 * Px0 - s0c * Py0,
        'P3': s1c * Px1 + c1 * Py1 - s0c * Px0 - c0 * Py0,
        'P4': s0c * Px0 + c0 * Py0,
        'x0': x0,
        'y0': y0,
        'x1': rpoints['p1']['x'],
        'y1': rpoints['p1']['y'],
        'x2': rpoints['p2']['x'],
        'y2': rpoints['p2']['y'],
        'theta': [float(theta0), float(theta1)],
        'size': contour['size'],
        'flip': bool(contour.get('flip', False))
    }
    return mapp
