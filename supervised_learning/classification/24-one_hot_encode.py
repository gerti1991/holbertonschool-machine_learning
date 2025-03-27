#!/usr/bin/env python3
"""
Module containing the function one_hot_encode
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Args:
        Y (numpy.ndarray): Vector of shape (m,) containing numeric class labels
                          where m is the number of examples
        classes (int): Maximum number of classes found in Y

    Returns:
        numpy.ndarray: One-hot encoding of Y with shape (classes, m),
        or None on failure
    """
    try:
        # Check if Y is a numpy.ndarray with shape (m,)
        if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
            return None

        # Check if classes is a positive integer
        if not isinstance(classes, int) or classes <= 0:
            return None

        # Initialize the one-hot matrix
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))

        # Set the appropriate positions to 1
        for i in range(m):
            if 0 <= Y[i] < classes:  # Check if the class label is valid
                one_hot[Y[i], i] = 1
            else:
                return None  # Invalid class label

        return one_hot
    except Exception:
        return None
