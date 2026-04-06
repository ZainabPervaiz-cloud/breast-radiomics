"""
Python port of rscore.m
Computes risk score and provides visualization of risk distribution.
"""
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage.transform import resize
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc.getinfo import getinfo
from misc.ffdmRead import ffdmRead
from segmentation.segBreast import segBreast
from features.xfeatures import xfeatures


def rscore(impath, model, dflag=False, save_plot=None):
    """
    Compute risk score of mammogram.

    Parameters
    ----------
    impath : str
        Path to DICOM.
    model : dict
        Risk model from rmodel.
    dflag : bool
        Whether to generate visualization.
    save_path : str, optional
        Where to save plot.
    """
    start_time = time.time()
    
    # 1. Feature extraction
    res = 0.1
    flist = ['imin', 'imax', 'iavg', 'ient', 'istd', 'ip05', 'ip95', 'iba1', 'iba2', 'ip30', 'ip70', 'iske', 'ikur', 'iran',
             'cene', 'ccor', 'ccon', 'chom', 'cent',
             'rsre', 'rlre', 'rgln', 'rrpe', 'rrln', 'rlgr', 'rhgr',
             'sgra', 'slap', 'swas', 'swav', 'swar', 'stev', 'fdim']
    
    info = getinfo(impath)
    im_norm, im_raw = ffdmRead(impath, info)
    
    # segBreast on 0.25 scale (MATLAB line 35)
    im_small = resize(im_norm, (int(im_norm.shape[0]*0.25), int(im_norm.shape[1]*0.25)), preserve_range=True)
    mask_small, _, _ = segBreast(im_small, info['ismlo'])
    
    # process raw image (MATLAB line 36 uses 'im' which is raw)
    im_target = resize(im_raw if info['israw'] else im_norm, 
                       (int(round(im_norm.shape[0] * info['psize'] / res)), 
                        int(round(im_norm.shape[1] * info['psize'] / res))), 
                       preserve_range=True)
    
    f = xfeatures(im_target, flist, mask_small, res)
    
    # 2. Sensor info (MATLAB line 40-50)
    target = info.get('target', '').upper()
    filt = info.get('filter', '').upper()
    sensor = -1
    if target == 'MO' and filt == 'RH': sensor = 0
    elif target == 'TU' and filt == 'AL': sensor = 1
    elif target == 'RH' and filt == 'RH': sensor = 2
    elif target == 'RH' and filt == 'SI': sensor = 3
    
    # 3. Assemble feature vector (MATLAB line 53)
    # x = [f, info.KVP, info.H, info.cforce, sensor]
    x_vec = np.hstack([f, info['KVP'], info.get('H', 0), info.get('cforce', 0), sensor])
    
    # 4. Predict (MATLAB line 56)
    r = model['lr'].predict_proba(x_vec.reshape(1, -1))[0, 1]
    
    elapsed = time.time() - start_time
    fname = os.path.basename(impath)
    print(f"File  : {fname}")
    print(f"Score : {r:.3f}")
    print(f"Age   : {info.get('age', 'N/A')}")
    print(f"Time  : {elapsed:.3f}")
    
    # 5. Visualization (MATLAB line 65-165)
    if dflag:
        plt.figure(figsize=(12, 6))
        
        # Left: Image
        plt.subplot(1, 2, 1)
        im_disp = resize(im_norm, (int(im_norm.shape[0]*0.25), int(im_norm.shape[1]*0.25)), preserve_range=True)
        plt.imshow(im_disp, cmap='gray')
        plt.axis('off')
        plt.text(30, im_disp.shape[0]-50, 
                 f"View: {info['side']}{info['view']}\nAge: {info['age']}\nR-score: {r:.2f}", 
                 color='lime', fontweight='bold')
        
        # Right: Risk distribution
        plt.subplot(1, 2, 2)
        
        def plot_density(scores, label, color):
            kde = gaussian_kde(scores, bw_method=0.1)
            x_range = np.linspace(0, 1, 100)
            y_vals = kde(x_range)
            y_vals /= np.max(y_vals)
            y_vals *= 0.3
            plt.fill_between(x_range, -y_vals, y_vals, color=color, alpha=0.3, label=label)
            # Medians (MATLAB boxplot-lite)
            median = np.median(scores)
            plt.scatter(median, 0, color='white', edgecolors='gray', s=30, zorder=5)
            
        # Class 0: Low Risk (Negative)
        plot_density(model['scores'][model['class'] == 0], "Low risk", "green")
        # Class 1: High Risk (Positive)
        plot_density(model['scores'][model['class'] == 1], "High risk", "red")
        
        # Risk line
        plt.axvline(r, color='gray', linewidth=2)
        plt.scatter(r, -0.35, marker='^', color='gray')
        plt.scatter(r, 0.35, marker='v', color='gray')
        plt.text(r, -0.4, f"{r:.2f}", ha='center', fontsize=12)
        
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        plt.yticks([])
        plt.xlabel('Risk score')
        plt.legend(loc='lower left')
        
        if save_plot:
            plt.savefig(save_plot)
        plt.close()
        
    return r
