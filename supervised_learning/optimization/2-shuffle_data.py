#!/usr/bin/env python3
"""
Module for data shuffling
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters:
        X (numpy.ndarray): Array of shape (m, nx) to shuffle
            m is the number of data points
            nx is the number of features
        Y (numpy.ndarray): Array of shape (m, ny) to shuffle
            m is the same number of data points as in X
            ny is the number of features in Y

    Returns:
        tuple: The shuffled X and Y matrices
    """
    # Generate a random permutation of indices
    permutation = np.random.permutation(X.shape[0])

    # Shuffle both matrices using the same permutation
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    return X_shuffled, Y_shuffled
