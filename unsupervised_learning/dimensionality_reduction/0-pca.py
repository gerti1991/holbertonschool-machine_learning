#!/usr/bin/env python3
"""Performs PCA on a dataset."""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA to retain specified variance fraction.

    Args:
        X: numpy.ndarray of shape (n, d), centered dataset
        var: float, fraction of variance to retain (default 0.95)

    Returns:
        W: numpy.ndarray of shape (d, nd), weights matrix maintaining
           var fraction of X's variance
    """
    cov = np.cov(X, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    cum_var = np.cumsum(eig_vals) / np.sum(eig_vals)
    n_components = np.argmax(cum_var >= var) + 1
    return eig_vecs[:, :n_components]
