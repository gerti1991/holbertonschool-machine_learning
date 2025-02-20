#!/usr/bin/env python3
"""That performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """
    Perform Principal Component Analysis (PCA) on the dataset X
    to retain a fraction of the total variance specified by var.

    Parameters:
    - X: numpy.ndarray of shape (n, d) where n is the number of data points
         and d is the number of dimensions of each data point.
    - var: Fraction of variance to retain (default is 0.95).

    Returns:
    - W: numpy.ndarray of shape (d, nd) where nd is the new dimensionality
         that maintains var fraction of X's original variance
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate explained variance ratio
    explained_variance = eigenvalues / np.sum(eigenvalues)

    # Calculate cumulative variance
    cumulative_variance = np.cumsum(explained_variance)

    # Find number of components needed for desired variance
    num_components = np.argmax(cumulative_variance >= var) + 1

    # Select top eigenvectors
    W = eigenvectors[:, :num_components]

    return W


# For debugging purposes, you can add this to see full transformation
if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.normal(size=50)
    b = np.random.normal(size=50)
    c = np.random.normal(size=50)
    d = 2 * a
    e = -5 * b
    f = 10 * c

    X = np.array([a, b, c, d, e, f]).T
    m = X.shape[0]
    X_m = X - np.mean(X, axis=0)

    # Full transformation with 3 components
    W_full = pca(X_m, var=1.0)  # var=1.0 to get all components (up to d=6)
    W_full = W_full[:, :3]  # Take first 3 components
    T = np.matmul(X_m, W_full)
    print(T)
    print(T.shape)

    # Reduced transformation with 95% variance
    W_reduced = pca(X_m, var=0.95)
    print(W_reduced)
    print(W_reduced.shape)
