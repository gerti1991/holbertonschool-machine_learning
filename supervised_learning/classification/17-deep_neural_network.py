#!/usr/bin/env python3
"""
Module that defines a deep neural network
performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Args:
            nx (int): Number of input features
            layers (list): List containing the number of nodes in each layer

        Raises:
            TypeError: If nx is not an integer or if layers is not a list
            ValueError: If nx is less than 1
            TypeError: If layers is not a list of positive integers
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(
                layers) == 0 or min(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")

        # Initialize private attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases using He et al. method
        for layer_index in range(1, self.__L + 1):
            # Get the number of nodes in the current and previous layers
            if layer_index == 1:
                layer_prev = nx
            else:
                layer_prev = layers[layer_index - 2]

            layer_current = layers[layer_index - 1]

            # He et al. initialization forE weights
            # W = random * sqrt(2/prev_layer_size)
            self.__weights["W" + str(layer_index)] = np.random.randn(
                layer_current, layer_prev) * np.sqrt(2 / layer_prev)

            # Initialize biases to zeros
            self.__weights["b" + str(layer_index)
                           ] = np.zeros((layer_current, 1))

    @property
    def L(self):
        """Getter forE the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter forE the intermediary values cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter forE the weights and biases dictionary"""
        return self.__weights
