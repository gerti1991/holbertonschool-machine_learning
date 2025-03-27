#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork class for binary classification
    """

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork class

        Args:
            nx (int): Number of input features
            layers (list): List containing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer in range(self.__L):
            if not isinstance(layers[layer], int) or layers[layer] <= 0:
                raise TypeError("layers must be a list of positive integers")

            w_key = "W{}".format(layer + 1)
            b_key = "b{}".format(layer + 1)

            # Initialize weights using He et al. method
            if layer == 0:
                self.__weights[w_key] = np.random.randn(
                    layers[layer], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[w_key] = np.random.randn(
                    layers[layer], layers[layer - 1]) * \
                    np.sqrt(2 / layers[layer - 1])

            # Initialize biases to zeros
            self.__weights[b_key] = np.zeros((layers[layer], 1))

    @property
    def L(self):
        """Getter for L (number of layers)"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)

        Returns:
            tuple: (A, cache) - output of the neural network and cached values
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            A_prev = self.__cache["A{}".format(layer - 1)]
            Z = np.matmul(W, A_prev) + b

            # Use sigmoid activation for the output layer, ReLU for hidden
            # layers
            if layer == self.__L:
                A = 1 / (1 + np.exp(-Z))  # Sigmoid
            else:
                A = np.maximum(0, Z)  # ReLU

            self.__cache["A{}".format(layer)] = A

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            float: Cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): Correct labels with shape (1, m)

        Returns:
            tuple: (prediction, cost) - the prediction and the cost
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            cache (dict): Dictionary containing all intermediary values
            alpha (float): Learning rate
        """
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(layer - 1)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            W = self.__weights["W{}".format(layer)]

            if layer > 1:
                dA = np.matmul(W.T, dZ)
                A = cache["A{}".format(layer - 1)]
                dZ = dA * (A > 0)  # Derivative of ReLU

            self.__weights["W{}".format(layer)] -= alpha * dW
            self.__weights["b{}".format(layer)] -= alpha * db

    def train(
            self,
            X,
            Y,
            iterations=5000,
            alpha=0.05,
            verbose=True,
            graph=True,
            step=100):
        """
        Trains the deep neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over
            alpha (float): Learning rate
            verbose (bool): Whether to print training information
            graph (bool): Whether to plot training information
            step (int): Step for printing/plotting information

        Returns:
            tuple: (prediction, cost) - the final prediction and cost
        """
        # Parameter validation
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iterations_list = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            # Print and store cost at specified steps
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    iterations_list.append(i)

            # Perform gradient descent (except for the last iteration)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        # Plot learning curve if requested
        if graph and iterations_list:
            plt.plot(iterations_list, costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Args:
            filename (str): File to which the object should be saved
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Args:
            filename (str): File from which the object should be loaded

        Returns:
            DeepNeuralNetwork: The loaded object, or None if filename
            doesn't exist
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
