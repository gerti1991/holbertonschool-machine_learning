#!/usr/bin/env python3
"""Bayesian Information Criterion Module"""

import numpy as np
def maximization(X, g):
    """
    Maximization step of the EM algorithm for a GMM.

    X: numpy.ndarray of shape (n, d) containing the dataset.
    g: numpy.ndarray of shape (k, n) containing the posterior probabilities.

    Returns:
    pi: numpy.ndarray of shape (k,) containing the updated priors for each cluster.
    m: numpy.ndarray of shape (k, d) containing the updated centroids (means).
    S: numpy.ndarray of shape (k, d, d) containing the updated covariance matrices.
    """
    if not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray):
        return None, None, None

    n, d = X.shape  # number of data points (n), number of features (d)
    k, _ = g.shape  # number of clusters (k)

    # Ensure g has the shape (k, n)
    if g.shape != (k, n):
        return None, None, None

    # 1. Calculate updated priors (pi)
    pi = np.sum(g, axis=1) / n

    # 2. Calculate updated means (m)
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    # 3. Calculate updated covariance matrices (S)
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]  # difference between data points and cluster mean
        S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    return pi, m, S
