#!/usr/bin/env python3
"""
Module that defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initialize the Neuron class

        Args:
            nx (int): number of input features to the neuron

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features to the neuron
                m is the number of examples

        Returns:
            The activated output (self.__A)
        """
        # Calculate the weighted sum Z = W·X + b
        Z = np.matmul(self.__W, X) + self.__b
        
        # Apply sigmoid activation function: A = 1/(1 + e^(-Z))
        self.__A = 1 / (1 + np.exp(-Z))
        
        return self.__A
