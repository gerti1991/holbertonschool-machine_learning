#!/usr/bin/env python3
"""
Module for momentum optimization using TensorFlow
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm in
    TensorFlow.

    Parameters:
        alpha (float): The learning rate
        beta1 (float): The momentum weight

    Returns:
        tensorflow.keras.optimizers.SGD: An optimizer for TensorFlow models
    """
    # Create a momentum optimizer using TensorFlow's SGD with momentum
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer
