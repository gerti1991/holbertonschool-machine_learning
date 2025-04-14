#!/usr/bin/env python3
"""
Module for normalization constants calculation
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Parameters:
        X (numpy.ndarray): Array of shape (m, nx) to normalize
            m is the number of data points
            nx is the number of features

    Returns:
        tuple: Mean and standard deviation of each feature, respectively
    """
    # Calculate mean along the vertical axis (axis=0)
    # This gives the mean of each feature
    mean = np.mean(X, axis=0)

    # Calculate standard deviation along the vertical axis (axis=0)
    # This gives the standard deviation of each feature
    std_dev = np.std(X, axis=0)

    return mean, std_dev
