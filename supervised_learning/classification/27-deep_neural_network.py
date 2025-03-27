#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class for multiclass classification
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
DNN26 = __import__('26-deep_neural_network').DeepNeuralNetwork


class DeepNeuralNetwork(DNN26):
    """
    DeepNeuralNetwork class for multiclass classification
    """

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

            # Apply activation function
            if layer_idx == self._DeepNeuralNetwork__L:
                # Use softmax activation for output layer (multiclass)
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Use sigmoid for hidden layers for compatibility with
                # gradient_descent
                A = 1 / (1 + np.exp(-Z))

            # Store activation in cache
            self._DeepNeuralNetwork__cache['A' + str(layer_idx)] = A

        # Return output (last layer activation) and cache
        return A, self._DeepNeuralNetwork__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using categorical cross-entropy

        Args:
            Y (numpy.ndarray): One-hot matrix of correct labels (classes, m)
            A (numpy.ndarray): Activated output (classes, m)

        Returns:
            float: Cost
        """
        m = Y.shape[1]
        # Categorical cross-entropy for multiclass
        # Add small epsilon (1e-15) to avoid log(0)
        cost = -np.sum(Y * np.log(A + 1e-15)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions for multiclass

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): One-hot matrix of correct labels (classes, m)

        Returns:
            tuple: (prediction, cost) - the prediction and the cost
        """
        # Perform forward propagation
        A, _ = self.forward_prop(X)

        # Calculate cost
        cost = self.cost(Y, A)

        # For multiclass, create one-hot predictions (1 at max probability)
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return prediction, cost
