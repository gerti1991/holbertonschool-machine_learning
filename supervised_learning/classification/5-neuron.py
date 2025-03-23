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
        """Getter method that returns the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter method that returns the bias"""
        return self.__b

    @property
    def A(self):
        """Getter method that returns the activated output"""
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
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
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
            numpy.ndarray: Predicted labels
            float: Cost of the network
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)
            alpha (float): Learning rate

        Returns:
            None
        """
        m = X.shape[1]

        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
