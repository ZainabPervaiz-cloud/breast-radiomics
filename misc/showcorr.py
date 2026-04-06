"""
Python port of showcorr.m
Visualize correlation among features.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def showcorr(X, fnames):
    """
    Visualize correlation among features.

    Parameters
    ----------
    X : ndarray
        MxN matrix of N-dimensional feature vectors.
    fnames : list of str
        Feature names (length N).
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]

    # Compute correlation matrix and p-values
    R = np.corrcoef(X.T)
    p = np.ones_like(R)
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                _, pval = pearsonr(X[:, i], X[:, j])
                p[i, j] = pval

    # Sort columns by correlation with first feature (descending)
    sort_idx = np.argsort(-R[0, :])
    R_sorted = np.corrcoef(X[:, sort_idx].T)
    p_sorted = p[np.ix_(sort_idx, sort_idx)]
    fnames_sorted = [fnames[i] for i in sort_idx]

    # ---- Plot 1: Correlation heatmap ----
    fig1, ax1 = plt.subplots(figsize=(10, 9))
    im1 = ax1.imshow(R_sorted, vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(n_features))
    ax1.set_xticklabels(fnames_sorted, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(n_features))
    ax1.set_yticklabels(fnames_sorted, fontsize=8)
    plt.colorbar(im1, ax=ax1)
    plt.tight_layout()

    # ---- Plot 2: Significance map ----
    sig_map = np.zeros_like(p_sorted)
    sig_map[p_sorted < 0.05] = 0.5
    sig_map[p_sorted < 0.01] = 0.75
    # Identity -> grey out diagonal
    np.fill_diagonal(sig_map, 1.0)

    fig2, ax2 = plt.subplots(figsize=(10, 9))
    ax2.imshow(sig_map, cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(n_features))
    ax2.set_xticklabels(fnames_sorted, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(n_features))
    ax2.set_yticklabels(fnames_sorted, fontsize=8)
    plt.tight_layout()

    # Print counts
    mask_off = ~np.eye(n_features, dtype=bool)
    n_weak = np.sum((p_sorted < 0.05) & (np.abs(R_sorted) < 0.3) & mask_off) / 2
    n_moderate = np.sum((p_sorted < 0.05) & (np.abs(R_sorted) >= 0.3) & (np.abs(R_sorted) < 0.7) & mask_off) / 2
    n_large = np.sum((p_sorted < 0.05) & (np.abs(R_sorted) >= 0.7) & mask_off) / 2
    print(f'Weak correlation: {int(round(n_weak))}')
    print(f'Moderate correlation: {int(round(n_moderate))}')
    print(f'Large correlation: {int(round(n_large))}')

    plt.show()
