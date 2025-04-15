#!/usr/bin/env python3
"""
Module for learning rate decay using inverse time decay in TensorFlow
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time
    decay.

    Parameters:
        alpha is the original learning rate
        decay_rate is the weight used to determine the rate at which alpha
                  will decay
        decay_step is the number of passes of gradient descent that should
                   occur before alpha is decayed further

    Returns:
        The learning rate decay operation
    """
    # Use the inverse_time_decay function from tf.keras.optimizers.schedules
    # staircase=True ensures the decay happens in a stepwise fashion
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
