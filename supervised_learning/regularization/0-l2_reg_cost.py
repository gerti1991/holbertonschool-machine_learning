#!/usr/bin/env python3
"""
Module for L2 regularization cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: The cost of the network without L2 regularization
        lambtha: The regularization parameter
        weights: A dictionary of the weights and biases of the neural network
        L: The number of layers in the neural network
        m: The number of data points used

    Returns:
        The cost of the network accounting for L2 regularization
    """
    # Initialize the L2 regularization cost
    l2_cost = 0

    # Sum the squares of all weights (not biases)
    for i in range(1, L + 1):
        weight_key = 'W' + str(i)
        if weight_key in weights:
            l2_cost += np.sum(np.square(weights[weight_key]))

    # Calculate the L2 regularization term: (λ/2m) * sum of squared weights
    l2_term = (lambtha / (2 * m)) * l2_cost

    # Return the original cost plus the L2 regularization term
    return cost + l2_term
