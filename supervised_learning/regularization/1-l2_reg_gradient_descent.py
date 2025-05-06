#!/usr/bin/env python3
"""
Module for L2 regularized gradient descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y: One-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: Dictionary of weights and biases of the neural network
        cache: Dictionary of outputs from each layer of the neural network
        alpha: Learning rate
        lambtha: L2 regularization parameter
        L: Number of layers in the neural network

    Returns:
        None (updates weights and biases in place)
    """
    m = Y.shape[1]

    for i in range(L, 0, -1):
        # Current layer activation
        if i == L:
            # For softmax output layer
            dZ = cache["A" + str(i)] - Y
        else:
            # For tanh hidden layers
            dZ = np.matmul(weights["W" + str(i+1)].T, dZ) * (1 - np.square(
                cache["A" + str(i)]))

        # Current layer weights and biases
        W = weights["W" + str(i)]
        # Gradient with L2 regularization
        dW = (1 / m) * np.matmul(dZ, cache["A" + str(i-1)].T) + \
            (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        # Update weights and biases
        weights["W" + str(i)] = weights["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db
