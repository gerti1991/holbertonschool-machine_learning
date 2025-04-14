#!/usr/bin/env python3
"""
Module for data normalization
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters:
        X (numpy.ndarray): Array of shape (d, nx) to normalize
            d is the number of data points
            nx is the number of features
        m (numpy.ndarray): Array of shape (nx,) containing the mean of all
            features of X
        s (numpy.ndarray): Array of shape (nx,) containing the standard
            deviation of all features of X

    Returns:
        numpy.ndarray: The normalized X matrix
    """
    # Normalize the matrix by subtracting the mean and dividing by the
    # standard deviation
    X_normalized = (X - m) / s

    return X_normalized
