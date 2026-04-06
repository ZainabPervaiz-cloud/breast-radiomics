"""
Python port of showcase.m
Display mammogram images from a dataset by patient/study ID.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from misc.getinfo import getinfo
from misc.ffdmRead import ffdmRead


def showcase(dataset, id_no, study_no):
    """
    Display mammogram images for a specific patient/study.

    Parameters
    ----------
    dataset : dict
        Dataset with keys 'id', 'study', 'path' (lists/arrays).
    id_no : int
        Patient ID number.
    study_no : int
        Study number.
    """
    ids = np.asarray(dataset['id'])
    studies = np.asarray(dataset['study'])
    paths = dataset['path']

    indices = np.where((ids == id_no) & (studies == study_no))[0]

    for i in indices:
        path = paths[i]
        info = getinfo(path)
        imn, _ = ffdmRead(path, info)
        # Resize to 10% (matches MATLAB imresize(im, .1))
        scale = 0.1
        im_small = zoom(imn, scale)
        plt.figure()
        plt.imshow(im_small, cmap='gray')
        plt.title(f"{info.get('side', '?')}{info.get('view', '?')}")
        plt.axis('off')
    plt.show()
