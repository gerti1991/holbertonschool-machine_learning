#!/usr/bin/env python3
"""
Module that defines a neural network with one hidden layer
performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """
    Class that defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network

        Args:
            nx (int): Number of input features
            nodes (int): Number of nodes in the hidden layer

        Raises:
            TypeError: If nx or nodes is not an integer
            ValueError: If nx or nodes is less than 1
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights, biases and activations as private attributes
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = np.zeros((1, 1))
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for the weights vector for the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Getter for the bias for the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Getter for the activated output for the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Getter for the weights vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Getter for the bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Getter for the activated output for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples

        Returns:
            tuple: The output of the hidden layer and the output layer
        """
        # Calculate the weighted sum for the hidden layer
        Z1 = np.matmul(self.__W1, X) + self.__b1

        # Apply sigmoid activation function for the hidden layer
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Calculate the weighted sum for the output layer
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2

        # Apply sigmoid activation function for the output layer
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            float: The cost
        """
        m = Y.shape[1]
        # Calculate cost using logistic regression formula
        # J(θ) = -1/m * Σ[y*log(a) + (1-y)*log(1-a)]
        # Using 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
