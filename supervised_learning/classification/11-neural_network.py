#!/usr/bin/env python3
"""
Module that defines a neural network with one hidden layer
"""
import numpy as np


class NeuralNetwork:
    """
    Class that defines a neural network with one hidden layer
    """

    def __init__(self, nx, nodes):
        """Initialize neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
            
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = np.zeros((1, 1))
        self.__A2 = 0

    @property
    def W1(self):
        """Gets W1"""
        return self.__W1

    @property
    def b1(self):
        """Gets b1"""
        return self.__b1

    @property
    def A1(self):
        """Gets A1"""
        return self.__A1

    @property
    def W2(self):
        """Gets W2"""
        return self.__W2

    @property
    def b2(self):
        """Gets b2"""
        return self.__b2

    @property
    def A2(self):
        """Gets A2"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates neural network outputs"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates cost"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
