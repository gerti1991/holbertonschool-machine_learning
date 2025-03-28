#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class with multiple
activation functions
"""
import numpy as np
DNN27 = __import__('27-deep_neural_network').DeepNeuralNetwork


class DeepNeuralNetwork(DNN27):
    """
    DeepNeuralNetwork class for multiclass classification
    with different activation functions
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize the deep neural network with activation function choice

        Args:
            nx (int): Number of input features
            layers (list): List containing the number of nodes in each layer
            activation (str): Type of activation function in hidden layers
                              'sig' for sigmoid or 'tanh' for hyperbolic
                              tangent

        Raises:
            ValueError: If activation is not 'sig' or 'tanh'
        """
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        super().__init__(nx, layers)
        self.__activation = activation

    @property
    def activation(self):
        """Getter for the activation function type"""
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)

        Returns:
            tuple: (A, cache) - output of the neural network and cached values
        """
        # For sigmoid activation (default case), use parent implementation
        if self.__activation == 'sig':
            return DNN27.forward_prop(self, X)

        # For tanh activation
        self._DeepNeuralNetwork__cache['A0'] = X
        A = X

        for layer_idx in range(1, self._DeepNeuralNetwork__L + 1):
            W = self._DeepNeuralNetwork__weights['W' + str(layer_idx)]
            b = self._DeepNeuralNetwork__weights['b' + str(layer_idx)]
            Z = np.matmul(W, A) + b

            if layer_idx == self._DeepNeuralNetwork__L:
                # Softmax for output layer
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Tanh for hidden layers
                A = np.tanh(Z)

            self._DeepNeuralNetwork__cache['A' + str(layer_idx)] = A

        return A, self._DeepNeuralNetwork__cache

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): One-hot matrix of correct labels (classes, m)
            cache (dict): Dictionary containing intermediary values
            alpha (float): Learning rate
        """
        # For sigmoid activation, use parent class implementation
        if self.__activation == 'sig':
            return DNN27.gradient_descent(self, Y, cache, alpha)

        # For tanh activation
        m = Y.shape[1]
        weights_copy = self._DeepNeuralNetwork__weights.copy()

        for layer in range(self._DeepNeuralNetwork__L, 0, -1):
            A_current = cache["A" + str(layer)]
            A_prev = cache["A" + str(layer - 1)]

            if layer == self._DeepNeuralNetwork__L:
                dZ = A_current - Y
            else:
                W_next = weights_copy["W" + str(layer + 1)]
                dZ_next = dZ
                # Tanh derivative: 1 - A^2
                dZ = np.matmul(W_next.T, dZ_next) * \
                    (1 - np.power(A_current, 2))

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self._DeepNeuralNetwork__weights["W" + str(layer)] = \
                weights_copy["W" + str(layer)] - alpha * dW
            self._DeepNeuralNetwork__weights["b" + str(layer)] = \
                weights_copy["b" + str(layer)] - alpha * db
