#!/usr/bin/env python3
"""
Module for creating mini-batches for neural network training
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for training a neural network using mini-batch
    gradient descent.

    Parameters:
        X (numpy.ndarray): Array of shape (m, nx) representing input data
            m is the number of data points
            nx is the number of features in X
        Y (numpy.ndarray): Array of shape (m, ny) representing the labels
            m is the same number of data points as in X
            ny is the number of classes for classification tasks
        batch_size (int): The number of data points in a batch

    Returns:
        list: Mini-batches containing tuples (X_batch, Y_batch)
    """
    # Get the number of data points
    m = X.shape[0]

    # Shuffle the data to ensure randomness in batches
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    # List to store the mini-batches
    mini_batches = []

    # Calculate the number of complete mini-batches
    complete_batches = m // batch_size

    # Create the complete mini-batches
    for i in range(complete_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]

        mini_batches.append((X_batch, Y_batch))

    # Handle the final batch if there are remaining data points
    if m % batch_size != 0:
        start_idx = complete_batches * batch_size

        X_batch = X_shuffled[start_idx:]
        Y_batch = Y_shuffled[start_idx:]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
