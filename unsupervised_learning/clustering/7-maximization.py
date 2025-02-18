#!/usr/bin/env python3
""" That calculates the maximization"""
import numpy as np


def maximization(X, g):
    """
    Performs the maximization step of the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): Data set of shape (n, d).
        g (numpy.ndarray): Posterior probabilities of shape (k, n).

    Returns:
        pi (numpy.ndarray): Updated priors of shape (k,).
        m (numpy.ndarray): Updated means of shape (k, d).
        S (numpy.ndarray): Updated covariance matrices of shape (k, d, d).
        or None, None, None if input validation fails.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray):
        return None, None, None
    if X.ndim != 2 or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    k, n_g = g.shape
    if n != n_g:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    # Update priors (pi)
    N_k = np.sum(g, axis=1)  # Shape (k,)
    pi = N_k / n  # Shape (k,)

    # Update means (m)
    m = np.dot(g, X) / N_k[:, np.newaxis]  # Shape (k, d)

    # Update covariance matrices (S)
    S = np.zeros((k, d, d))  # Shape (k, d, d)
    for i in range(k):
        diff = X - m[i]  # Shape (n, d)
        weighted_diff = g[i, :, np.newaxis] * diff  # Shape (n, d)
        S[i] = np.dot(weighted_diff.T, diff) / N_k[i]  # Shape (d, d)

    return pi, m, S
