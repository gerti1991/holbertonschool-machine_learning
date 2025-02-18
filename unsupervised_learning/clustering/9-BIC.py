#!/usr/bin/env python3
"""Bayesian Information Criterion Module"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Determines the best number of clusters for a GMM using BIC.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    n, d = X.shape

    # First, if kmax is provided, validate it.
    if kmax is not None:
        if not isinstance(kmax, int) or kmax < kmin:
            return None, None, None, None
    else:
        kmax = n
        if kmax < kmin:
            return None, None, None, None

    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    log_likelihoods = np.zeros(kmax - kmin + 1)
    bics = np.zeros(kmax - kmin + 1)
    best_idx = 0
    best_bic = float("inf")
    best_result = None
    best_k = kmin

    for i, k in enumerate(range(kmin, kmax + 1)):
        result = expectation_maximization(X, k, iterations, tol, verbose)
        if result is None or len(result) < 5 or result[4] is None:
            return None, None, None, None

        pi, m, S, g, log_likelihood = result

        # Number of parameters: means + covariances + mixing coefficients
        p = (k * d) + (k * d * (d + 1)) // 2 + (k - 1)
        bic = p * np.log(n) - 2 * log_likelihood

        log_likelihoods[i] = log_likelihood
        bics[i] = bic

        if bic < best_bic:
            best_bic = bic
            best_idx = i
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics
