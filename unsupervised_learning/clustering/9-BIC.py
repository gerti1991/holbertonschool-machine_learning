#!/usr/bin/env python3
"""Bayesian Information Criterion Module"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Test
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    best_idx = 0
    best_bic = float("inf")
    best_result = None
    best_k = kmin

    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)

        p = (k * d) + (k * d * (d + 1)) // 2 + (k - 1)
        bic = p * np.log(n) - 2 * log_likelihood

        l[i] = log_likelihood
        b[i] = bic

        if bic < best_bic:
            best_bic = bic
            best_idx = i
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
