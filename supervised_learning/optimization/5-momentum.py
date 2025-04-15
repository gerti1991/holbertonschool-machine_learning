#!/usr/bin/env python3
"""
Module for momentum optimization algorithm
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum
    optimization algorithm.

    Parameters:
        alpha (float): The learning rate
        beta1 (float): The momentum weight
        var (numpy.ndarray): The variable to be updated
        grad (numpy.ndarray): The gradient of var
        v (numpy.ndarray or float): The previous first moment of var

    Returns:
        tuple: The updated variable and the new moment, respectively
    """
    # Calculate the new momentum (velocity)
    # v_t = beta1 * v_{t-1} + (1 - beta1) * gradient
    v_new = beta1 * v + (1 - beta1) * grad

    # Update the variable using momentum
    # var = var - alpha * v_new
    var_updated = var - alpha * v_new

    return var_updated, v_new
