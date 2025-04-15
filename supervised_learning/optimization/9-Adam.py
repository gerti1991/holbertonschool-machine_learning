#!/usr/bin/env python3
"""
Module for Adam optimization algorithm
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Parameters:
        alpha (float): The learning rate
        beta1 (float): The weight used for the first moment
        beta2 (float): The weight used for the second moment
        epsilon (float): A small number to avoid division by zero
        var (numpy.ndarray): The variable to be updated
        grad (numpy.ndarray): The gradient of var
        v (numpy.ndarray or float): The previous first moment of var
        s (numpy.ndarray or float): The previous second moment of var
        t (int): The time step used for bias correction

    Returns:
        tuple: The updated variable, the new first moment, and the new second
               moment, respectively
    """
    # Update first moment (momentum)
    # v_t = beta1 * v_{t-1} + (1 - beta1) * grad
    v_new = beta1 * v + (1 - beta1) * grad

    # Update second moment (RMSProp)
    # s_t = beta2 * s_{t-1} + (1 - beta2) * (grad)^2
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Bias correction for first moment
    # v_corrected = v_t / (1 - beta1^t)
    v_corrected = v_new / (1 - beta1 ** t)

    # Bias correction for second moment
    # s_corrected = s_t / (1 - beta2^t)
    s_corrected = s_new / (1 - beta2 ** t)

    # Update variable using Adam formula
    # var = var - alpha * v_corrected / (sqrt(s_corrected) + epsilon)
    var_updated = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var_updated, v_new, s_new
