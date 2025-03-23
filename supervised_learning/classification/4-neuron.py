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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y (numpy.ndarray): Correct labels for input data, shape (1, m)
            A (numpy.ndarray): Activated output of neuron, shape (1, m)
        Returns:
            The cost
        """
        m = Y.shape[1]
        # Calculate cost using logistic regression formula
        # J(θ) = -1/m * Σ[y*log(a) + (1-y)*log(1-a)]
        # Using 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
        
    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions
        
        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features to the neuron
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)
                
        Returns:
            numpy.ndarray: Predicted labels for each example (1, m)
            float: Cost of the network
        """
        # Get the output of the neural network (forward propagation)
        A = self.forward_prop(X)
        
        # Calculate the cost
        cost = self.cost(Y, A)
        
        # Convert probabilities to binary predictions (1 if >= 0.5, 0 otherwise)
        prediction = np.where(A >= 0.5, 1, 0)
        
        return prediction, cost