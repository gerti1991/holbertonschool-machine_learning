#!/usr/bin/env python3
"""Module implementing batch normalization for neural networks in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    Parameters:
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used
                on the output of the layer

    Returns:
    A tensor of the activated output for the layer
    """
    # Define the kernel initializer
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create the base layer (Dense layer without activation)
    base_layer = tf.keras.layers.Dense(
        units=n, kernel_initializer=initializer, use_bias=False)

    # Get the output of the base layer
    Z = base_layer(prev)

    # Calculate mean and variance for batch normalization
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Define gamma and beta as trainable variables
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')

    # Apply batch normalization
    epsilon = 1e-7
    Z_norm = tf.nn.batch_normalization(
        Z, mean, variance, offset=beta, scale=gamma, variance_epsilon=epsilon)

    # Apply the activation function
    if activation is not None:
        output = activation(Z_norm)
    else:
        output = Z_norm

    return output
