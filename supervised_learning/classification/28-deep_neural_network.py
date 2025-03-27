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
        # Validate activation type
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        # Call parent class constructor
        super().__init__(nx, layers)

        # Set the activation type
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
        # Store input in cache as A0
        self._DeepNeuralNetwork__cache['A0'] = X

        # Forward propagation through each layer
        A = X
        for layer_idx in range(1, self._DeepNeuralNetwork__L + 1):
            # Get weights and biases from weights dictionary
            W = self._DeepNeuralNetwork__weights['W' + str(layer_idx)]
            b = self._DeepNeuralNetwork__weights['b' + str(layer_idx)]

            # Calculate Z = W·A + b
            Z = np.matmul(W, A) + b

            # Apply appropriate activation function
            if layer_idx == self._DeepNeuralNetwork__L:
                # Softmax activation for output layer (multiclass)
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Use specified activation function for hidden layers
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))  # Sigmoid
                else:
                    A = np.tanh(Z)  # Hyperbolic tangent

            # Store activation in cache
            self._DeepNeuralNetwork__cache['A' + str(layer_idx)] = A

        # Return output (last layer activation) and cache
        return A, self._DeepNeuralNetwork__cache

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): One-hot matrix of correct labels (classes, m)
            cache (dict): Dictionary containing intermediary values
            alpha (float): Learning rate
        """
        m = Y.shape[1]
        weights_copy = self._DeepNeuralNetwork__weights.copy()

        # Backpropagation - working from output layer to input layer
        for layer in range(self._DeepNeuralNetwork__L, 0, -1):
            # Get activations for current and previous layer
            A_current = cache["A" + str(layer)]
            A_prev = cache["A" + str(layer - 1)]

            # Calculate gradients differently for output layer and hidden
            # layers
            if layer == self._DeepNeuralNetwork__L:
                # For output layer with softmax: dZ = A - Y
                dZ = A_current - Y
            else:
                # For hidden layers, use appropriate derivative based on
                # activation
                W_next = weights_copy["W" + str(layer + 1)]
                dZ_next = dZ

                # Calculate derivative based on activation function
                if self.__activation == 'sig':
                    # Sigmoid derivative: A * (1 - A)
                    dZ = np.matmul(W_next.T, dZ_next) * \
                        A_current * (1 - A_current)
                else:
                    # Tanh derivative: 1 - A^2
                    dZ = np.matmul(W_next.T, dZ_next) * \
                        (1 - np.power(A_current, 2))

            # Calculate weight and bias gradients
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Update weights and biases
            w_key = "W" + str(layer)
            b_key = "b" + str(layer)
            self._DeepNeuralNetwork__weights[w_key] = weights_copy[w_key] - alpha * dW
            self._DeepNeuralNetwork__weights[b_key] = weights_copy[b_key] - alpha * db
