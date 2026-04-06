"""
Python port of mpdiff.m
Computes discriminative power of micro-patterns.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mpatterns.mpSampling import mpSampling
from mpatterns.bestStump import bestStump


def mpdiff(model, dataset=None):
    """
    Discriminative power of MP patterns.

    Parameters
    ----------
    model : dict
    dataset : list of dict, optional
    """
    if dataset is not None:
        if isinstance(dataset, list):
            no_images = len(dataset)
            num_prototypes = model['P'].shape[1]
            D = np.zeros((num_prototypes, num_prototypes, no_images))
            
            print("Computing discriminative power...")
            for n in range(no_images):
                D[:, :, n] = mpdiff(model, dataset[n])
                if (n+1) % 5 == 0 or n == no_images-1:
                    print(f"Progress: {100*(n+1)/no_images:.0f}%", end='\r')
            print()
            
            labels = np.array([d['class'] for d in dataset])
            DP = np.zeros((num_prototypes, num_prototypes))
            for m in range(num_prototypes):
                for n in range(m + 1, num_prototypes):
                    X = D[m, n, :]
                    DP[m, n] = bestStump(X, labels)
                    DP[n, m] = DP[m, n]
            return DP
        else:
            # Single image case (MATLAB line 51)
            S, _ = mpSampling(dataset, model['params'])
            no_samples = S.shape[1]
            num_prototypes = model['P'].shape[1]
            DP = np.zeros((num_prototypes, num_prototypes))
            
            P = model['P']
            # Compute distance for all samples to all prototypes
            # S is D x nsamples, P is D x nprototypes
            S_sq = np.sum(S**2, axis=0)
            P_sq = np.sum(P**2, axis=0)
            dist_sq = S_sq[np.newaxis, :] + P_sq[:, np.newaxis] - 2 * np.dot(P.T, S)
            
            # For each pair (m, n)
            for m in range(num_prototypes):
                for n in range(m + 1, num_prototypes):
                    # For each sample, find if it's closer to m or n
                    d_m = dist_sq[m, :]
                    d_n = dist_sq[n, :]
                    # MATLAB: [~, i] = min([h1; h2]); ... n1 = sum(i==1)/no_samples;
                    n1 = np.sum(d_m < d_n) / no_samples
                    n2 = np.sum(d_n <= d_m) / no_samples
                    DP[m, n] = n1 - n2
            return DP
    else:
        # MATLAB line 41
        num_prototypes = model['H'].shape[1]
        DP = np.zeros(num_prototypes)
        labels = model['class']
        for n in range(num_prototypes):
            X = model['H'][:, n]
            DP[n] = bestStump(X, labels)
        return DP
