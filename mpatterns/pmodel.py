"""
Python port of pmodel.m
Builds a statistical population model from an image dataset.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpatterns.phistogram import phistogram


def pmodel(dataset, P, params):
    """
    Build MP-based statistical model.

    Parameters
    ----------
    dataset : list of dict
        Dataset structure.
    P : ndarray
        Prototypes.
    params : dict
        Parameters.
    """
    no_images = len(dataset)
    no_prototypes = P.shape[1]
    
    H = np.zeros((no_images, no_prototypes))
    classes = np.array([d['class'] for d in dataset])
    
    print("Building model...")
    for n in range(no_images):
        H[n, :] = phistogram(dataset[n], P, params)
        if (n + 1) % 5 == 0 or n == no_images - 1:
            print(f"Progress: {100 * (n + 1) / no_images:.0f}%", end='\r')
    print("\nModel complete.")
    
    model = {
        'H': H,
        'class': classes,
        'params': params,
        'P': P,
        'C': len(np.unique(classes))
    }
    return model
