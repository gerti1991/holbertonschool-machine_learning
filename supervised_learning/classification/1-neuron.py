#!/usr/bin/env python3
"""Module defining the Neuron class for binary classification"""

import numpy as np


class Neuron:
    """
    A class that represents a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initializes a Neuron instance

        Args:
            nx: number of input features to the neuron

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A
