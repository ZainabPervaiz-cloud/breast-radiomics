"""
Python port of getperf.m
Compute binary classification performance metrics: AUC and OPERA.
"""
import numpy as np
from scipy.stats import norm
from scipy.special import logit, expit


def getperf(labels, scores, dispflag=False):
    """
    Compute binary classification performance.

    Parameters
    ----------
    labels : array-like
        Boolean vector with correct labels (True/False or 1/0).
    scores : array-like
        Scalar vector with prediction scores.
    dispflag : bool
        Print results table if True.

    Returns
    -------
    auc : float
        Area under the ROC curve.
    or_ : float
        Odds per standard deviation (OPERA).
    auc_CI : ndarray
        [lower, upper] 95% CI for AUC.
    or_CI : ndarray
        [lower, upper, pvalue] 95% CI for OR.
    """
    from scipy.stats import logistic
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()

    # Z-score the scores (matches MATLAB zscore)
    x = (scores - scores.mean()) / (scores.std(ddof=1) + 1e-12)

    # Logistic regression for OR (GLM with binomial link)
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(fit_intercept=True, penalty=None, solver='lbfgs', max_iter=1000)
        lr.fit(x.reshape(-1, 1), labels)
        B1 = lr.coef_[0][0]
        # Approximate SE via asymptotic covariance
        p_hat = lr.predict_proba(x.reshape(-1, 1))[:, 1]
        W = np.diag(p_hat * (1 - p_hat))
        X_mat = np.column_stack([np.ones_like(x), x])
        cov = np.linalg.pinv(X_mat.T @ W @ X_mat)
        se_B1 = np.sqrt(cov[1, 1])
        B_lo = B1 - 1.965 * se_B1
        B_hi = B1 + 1.965 * se_B1
        # p-value: Wald test
        z_stat = B1 / se_B1
        pval = 2 * (1 - norm.cdf(abs(z_stat)))
    except ImportError:
        # Minimal fallback without sklearn
        B1, se_B1, B_lo, B_hi, pval = 0.0, 1.0, -1.96, 1.96, 1.0

    or_ = np.exp(B1)
    or_CI = np.array([np.exp(B_lo), np.exp(B_hi), pval])

    # AUC via Mann-Whitney U (non-parametric, matches DeLong)
    pos = x[labels == 1]
    neg = x[labels == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        auc = 0.5
        auc_CI = np.array([0.5, 0.5])
    else:
        # Standard Mann-Whitney
        from scipy.stats import mannwhitneyu
        stat, _ = mannwhitneyu(pos, neg, alternative='two-sided')
        auc = stat / (n1 * n0)
        # Hanley-McNeil SE approximation
        q1 = auc / (2 - auc)
        q2 = 2 * auc**2 / (1 + auc)
        se_auc = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
        auc_CI = np.array([auc - 1.96 * se_auc, auc + 1.96 * se_auc])

    if dispflag:
        print(f'{"":10s} {"value":>8s}  {"95% CI":>15s}  {"pvalue":>8s}')
        print(f'{"OR":10s} {or_:8.3f}  [{or_CI[0]:.3f}, {or_CI[1]:.3f}]  {or_CI[2]:8.4f}')
        print(f'{"AUC":10s} {auc:8.3f}  [{auc_CI[0]:.3f}, {auc_CI[1]:.3f}]  {"nan":>8s}')

    return auc, or_, auc_CI, or_CI
