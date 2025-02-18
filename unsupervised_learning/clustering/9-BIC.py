import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
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

    results = [expectation_maximization(X, k, iterations, tol, verbose) for k in range(kmin, kmax + 1)]

    for i, (pi, m, S, g, log_likelihood) in enumerate(results):
        k = kmin + i

        # Number of parameters: 
        # k * d for means + k * d * (d + 1) / 2 for covariance + (k - 1) for mixing coefficients
        p = (k * d) + (k * d * (d + 1)) // 2 + (k - 1)

        l[i] = log_likelihood
        b[i] = p * np.log(n) - 2 * log_likelihood

    best_idx = np.argmin(b)

    best_k = kmin + best_idx
    best_result = results[best_idx][:3]

    return best_k, best_result, l, b
