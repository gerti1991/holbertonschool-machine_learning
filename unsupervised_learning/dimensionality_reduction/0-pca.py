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

    # Remove the forced minimum of 3 components that was causing issues
    # num_components = max(num_components, 3) <- Remove this line

    # Select top eigenvectors
    W = eigenvectors[:, :num_components]

    return W
