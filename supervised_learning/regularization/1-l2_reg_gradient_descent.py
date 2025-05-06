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
    # Get the number of training examples
    m = Y.shape[1]

    # Initialize dZ for the output layer (layer L)
    # For softmax with cross-entropy loss:
    # dZ[L] = A[L] - Y (difference between predictions and actual labels)
    dZ = cache['A' + str(L)] - Y

    # Go backwards through the layers
    for i in range(L, 0, -1):
        # Current layer activation
        A = cache['A' + str(i)]
        # Previous layer activation
        A_prev = cache['A' + str(i - 1)]

        # Current layer weights and biases
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        # Gradient of cost with respect to W with L2 regularization
        # Standard gradient: (1/m) * dZ * A_prev.T
        # L2 regularization term: (lambtha/m) * W
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W

        # Gradient of cost with respect to b
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases using gradient descent
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db

        # Compute dZ for the previous layer (if not at the input layer)
        if i > 1:
            # For tanh activation: dZ[i-1] = W[i].T * dZ[i] * (1 - A[i-1]²)
            dZ = np.matmul(weights['W' + str(i)].T, dZ) * (1 - np.power(
                A_prev, 2))
