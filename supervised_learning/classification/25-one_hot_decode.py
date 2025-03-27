#!/usr/bin/env python3
"""
Module containing the function one_hot_decode
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels

    Args:
        one_hot (numpy.ndarray): One-hot encoded numpy.ndarray with
                                shape (classes, m)
                                where classes is the maximum number of classes
                                and m is the number of examples

    Returns:
        numpy.ndarray: Array with shape (m,) containing the numeric labels
                      for each example, or None on failure
    """
    try:
        # Check if one_hot is a valid numpy array with at least 2 dimensions
        if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
            return None

        # Return the indices of the 1s in each column
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
