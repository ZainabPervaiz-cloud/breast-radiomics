"""
Python port of rmodel.m
Trains a logistic regression model for breast cancer risk assessment.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def rmodel(x_train, y_train):
    """
    Train risk model using logistic regression.

    Parameters
    ----------
    x_train : ndarray
        N x M feature matrix.
    y_train : ndarray
        Class labels.
    """
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    
    # MATLAB: fitglm(x_train, y_train, 'binomial', 'link', 'logit')
    # Equivalent to LogisticRegression with no penalty for exact parity, 
    # but scikit-learn defaults to L2. We'll use penalty='none' for exactness.
    lr = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
    lr.fit(x_train, y_train)
    
    # Predict probabilities (scores)
    scores = lr.predict_proba(x_train)[:, 1]
    
    model = {
        'lr': lr,
        'class': y_train,
        'scores': scores
    }
    return model
