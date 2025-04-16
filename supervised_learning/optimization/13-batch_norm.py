#!/usr/bin/env python3
"""Module implementing batch normalization for neural networks"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using
    batch normalization.

    Parameters:
    Z is a numpy.ndarray of shape (m, n) that should be normalized
        m is the number of data points
        n is the number of features in Z
    gamma is a numpy.ndarray of shape (1, n) containing the scales
          used for batch normalization
    beta is a numpy.ndarray of shape (1, n) containing the offsets
         used for batch normalization
    epsilon is a small number used to avoid division by zero

    Returns:
    The normalized Z matrix
    """
    # Calculate the mean of each feature across all data points
    mean = np.mean(Z, axis=0, keepdims=True)

    # Calculate the variance of each feature across all data points
    var = np.var(Z, axis=0, keepdims=True)

    # Normalize Z (Z_norm = (Z - mean) / sqrt(var + epsilon))
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)

    # Scale and shift the normalized values
    Z_norm = gamma * Z_norm + beta

    return Z_norm
