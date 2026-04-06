"""
Python port of mpRank.m
Placeholder for ILFS (Infinite Latent Feature Selection).
"""
import numpy as np


def mpRank(model):
    """
    Rank MP patterns.
    Uses ILFS (Infinite Latent Feature Selection).
    Currently implemented as a placeholder returning indices by variance.
    """
    print("WARNING: ILFS is not available. Using variance-based ranking as a placeholder.")
    H = model['H']
    # Rank by variance across the dataset
    variances = np.var(H, axis=0)
    ranking = np.argsort(variances)[::-1]
    return ranking
