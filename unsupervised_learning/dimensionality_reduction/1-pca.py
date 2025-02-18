#!/usr/bin/env python3
"""That performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """
    Perform PCA on the dataset X to reduce it to
        the desired number of dimensions (ndim).

    Parameters:
    - X: numpy.ndarray of shape (n, d) where n is the number of data points
         and d is the number of dimensions of each data point.
    - ndim: The number of dimensions to reduce X to.

    Returns:
    - T: numpy.ndarray of shape (n, ndim)
         containing the transformed version of X.
    """
    # Step 1: Center the data (mean zero)
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Compute the covariance matrix of the centered data
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Select the top `ndim` eigenvectors
    W = eigenvectors[:, :ndim]

    # Step 6: Transform the data into the new lower-dimensional space
    T = np.dot(X_centered, W)

    # Print out the transformed data and its shape for debugging
    print('Transformed data shape:', T.shape)
    print('Transformed data (first 5 rows):\n', T[:5, :])

    return T
