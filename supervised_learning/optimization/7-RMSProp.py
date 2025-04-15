#!/usr/bin/env python3
"""
Module for RMSProp optimization algorithm
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
        alpha (float): The learning rate
        beta2 (float): The RMSProp weight
        epsilon (float): A small number to avoid division by zero
        var (numpy.ndarray): The variable to be updated
        grad (numpy.ndarray): The gradient of var
        s (numpy.ndarray or float): The previous second moment of var

    Returns:
        tuple: The updated variable and the new moment, respectively
    """
    # Update the second moment (s)
    # s_t = beta2 * s_{t-1} + (1 - beta2) * (grad)^2
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update the variable using RMSProp formula
    # var = var - alpha * grad / (sqrt(s_new) + epsilon)
    var_updated = var - alpha * grad / (np.sqrt(s_new) + epsilon)

    return var_updated, s_new
