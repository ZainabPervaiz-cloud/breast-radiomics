"""
Python port of showmap.m
Visualize the ST coordinate grid overlaid on a mammogram.
"""
import numpy as np
import matplotlib.pyplot as plt


def showmap(im, mapp):
    """
    Show ST coordinate mapping grid on image.

    Parameters
    ----------
    im : ndarray
        MxN grayscale input image.
    mapp : dict
        Mapping parameters.
    """
    m_grid = 30
    n_grid = 30
    
    im = np.asarray(im, dtype=float)
    if mapp.get('flip', False):
        im = np.fliplr(im)
        
    ax = plt.gca()
    ax.imshow(im, cmap='gray')
    ax.set_autoscale_on(False)
    
    s_vals = np.linspace(0, 1)
    P1 = np.polyval(mapp['P1'], s_vals)
    P2 = np.polyval(mapp['P2'], s_vals)
    P3 = np.polyval(mapp['P3'], s_vals)
    P4 = np.polyval(mapp['P4'], s_vals)
    
    # Show t-grid (lines of constant t)
    t_ticks = np.linspace(0, 1, m_grid)
    theta_ticks = (mapp['theta'][1] - mapp['theta'][0]) * t_ticks + mapp['theta'][0]
    for m in range(m_grid):
        pa = t_ticks[m] * P1 + P2
        pb = t_ticks[m] * P3 + P4
        x = pa * np.cos(theta_ticks[m]) - pb * np.sin(theta_ticks[m])
        y = pa * np.sin(theta_ticks[m]) + pb * np.cos(theta_ticks[m])
        ax.plot(x + mapp['x0'], y + mapp['y0'], 'g', linewidth=0.5)

    # Show s-grid (lines of constant s)
    s_ticks = np.linspace(0, 1, n_grid)
    t_vals = np.linspace(0, 1)
    theta_vals = (mapp['theta'][1] - mapp['theta'][0]) * t_vals + mapp['theta'][0]
    for n in range(n_grid):
        pa = t_vals * np.polyval(mapp['P1'], s_ticks[n]) + np.polyval(mapp['P2'], s_ticks[n])
        pb = t_vals * np.polyval(mapp['P3'], s_ticks[n]) + np.polyval(mapp['P4'], s_ticks[n])
        x = pa * np.cos(theta_vals) - pb * np.sin(theta_vals)
        y = pa * np.sin(theta_vals) + pb * np.cos(theta_vals)
        ax.plot(x + mapp['x0'], y + mapp['y0'], 'g', linewidth=0.5)

    # Show reference points
    ax.plot([mapp['x0'], mapp['x1'], mapp['x2']], 
             [mapp['y0'], mapp['y1'], mapp['y2']], 
             'yo', markersize=6, markeredgewidth=2)
    
    ax.set_title("ST Coordinate Mapping Grid")
