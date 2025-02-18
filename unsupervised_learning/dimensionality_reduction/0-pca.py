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
    - W: numpy.ndarray of shape (d, nd) where nd is the number of dimensions
         after reduction (which keeps the specified variance).
    """
    # Step 1: Center the data (mean zero)
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Calculate the explained variance
    explained_variance = eigenvalues / np.sum(eigenvalues)

    # Step 6: Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)

    # Print out the cumulative variance to debug
    print(f"Cumulative Variance Explained: {cumulative_variance}")

    # Step 7: Find the number of components to retain the desired variance
    num_components = np.argmax(cumulative_variance >= var) + 1

    # If not enough components are retained.
    num_components = max(num_components, 3)

    # Step 8: Select the eigenvectors corresponding to the largest eigenvalues
    W = eigenvectors[:, :num_components]

    # Return the weight matrix W that contains the principal components
    return W
