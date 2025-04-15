#!/usr/bin/env python3
"""
Module for learning rate decay using inverse time decay
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    Parameters:
        alpha (float): The original learning rate
        decay_rate (float): The weight used to determine the rate at which
                           alpha will decay
        global_step (int): The number of passes of gradient descent that
                          have elapsed
        decay_step (int): The number of passes of gradient descent that should
                         occur before alpha is decayed further

    Returns:
        float: The updated value for alpha
    """
    # Calculate the number of times to decay the learning rate
    # This implements the stepwise decay by integer division
    step_factor = np.floor(global_step / decay_step)

    # Calculate the new learning rate using inverse time decay formula:
    # alpha / (1 + decay_rate * number_of_decay_steps)
    alpha_updated = alpha / (1 + decay_rate * step_factor)

    return alpha_updated