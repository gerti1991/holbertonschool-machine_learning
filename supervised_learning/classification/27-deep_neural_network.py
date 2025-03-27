#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class for multiclass classification
"""
import numpy as np
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
        self._DeepNeuralNetwork__cache["A0"] = X

        for layer in range(1, self._DeepNeuralNetwork__L + 1):
            W = self._DeepNeuralNetwork__weights["W{}".format(layer)]
            b = self._DeepNeuralNetwork__weights["b{}".format(layer)]
            A_prev = self._DeepNeuralNetwork__cache["A{}".format(layer - 1)]
            Z = np.matmul(W, A_prev) + b

            if layer == self._DeepNeuralNetwork__L:
                # Softmax activation for multiclass classification
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                A = np.maximum(0, Z)  # ReLU for hidden layers

            self._DeepNeuralNetwork__cache["A{}".format(layer)] = A

        return self._DeepNeuralNetwork__cache["A{}".format(
            self._DeepNeuralNetwork__L)], self._DeepNeuralNetwork__cache

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
        # Use categorical cross-entropy for multiclass
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
        A, _ = self.forward_prop(X)

        # For multiclass, create a one-hot matrix with 1 at max probability
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): One-hot matrix of correct labels (classes, m)
            cache (dict): Dictionary containing all intermediary values
            alpha (float): Learning rate
        """
        m = Y.shape[1]

        # For softmax + cross-entropy, the initial error is simply (A - Y)
        dZ = cache["A{}".format(self._DeepNeuralNetwork__L)] - Y

        for layer in range(self._DeepNeuralNetwork__L, 0, -1):
            A_prev = cache["A{}".format(layer - 1)]

            # Compute gradients
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Store current weights for computing next dZ
            W = self._DeepNeuralNetwork__weights["W{}".format(layer)]

            # Update weights and biases
            self._DeepNeuralNetwork__weights["W{}".format(layer)] -= alpha * dW
            self._DeepNeuralNetwork__weights["b{}".format(layer)] -= alpha * db

            # Compute dZ for previous layer (if not at the first layer)
            if layer > 1:
                dA = np.matmul(W.T, dZ)
                A = cache["A{}".format(layer - 1)]
                # ReLU derivative: 1 if A > 0, 0 otherwise
                dZ = dA * (A > 0)
