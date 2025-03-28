#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class with multiple
activation functions
"""
import numpy as np


class DeepNeuralNetwork:
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
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for layer_index in range(1, self.__L + 1):
            if not isinstance(layers[layer_index - 1], int) or \
               layers[layer_index - 1] <= 0:
                raise TypeError("layers must be a list of positive integers")

            w_key = "W{}".format(layer_index)
            b_key = "b{}".format(layer_index)

            if layer_index == 1:
                self.__weights[w_key] = np.random.randn(
                    layers[layer_index - 1], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[w_key] = np.random.randn(
                    layers[layer_index - 1], layers[layer_index - 2]) * \
                    np.sqrt(2 / layers[layer_index - 2])

            self.__weights[b_key] = np.zeros((layers[layer_index - 1], 1))

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
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            A_prev = self.__cache["A{}".format(layer - 1)]
            Z = np.matmul(W, A_prev) + b

            if layer == self.__L:
                # Softmax activation for output layer
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Use specified activation for hidden layers
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))  # Sigmoid
                else:
                    A = np.tanh(Z)  # Tanh

            self.__cache["A{}".format(layer)] = A

        return self.__cache["A{}".format(self.__L)], self.__cache

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
        # Add small epsilon (1e-15) to avoid log(0)
        cost = -np.sum(Y * np.log(A + 1e-15)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): One-hot matrix of correct labels (classes, m)

        Returns:
            tuple: (prediction, cost) - the prediction and the cost
        """
        A, _ = self.forward_prop(X)

        # Create one-hot prediction matrix (1 at position of max probability)
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): One-hot matrix of correct labels (classes, m)
            cache (dict): Dictionary containing intermediary values
            alpha (float): Learning rate
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()

        for layer in range(self.__L, 0, -1):
            A_current = cache["A" + str(layer)]
            A_prev = cache["A" + str(layer - 1)]

            if layer == self.__L:
                dZ = A_current - Y
            else:
                W_next = weights_copy["W" + str(layer + 1)]
                dZ_next = dZ

                if self.__activation == 'sig':
                    # Sigmoid derivative: A * (1 - A)
                    dZ = np.matmul(W_next.T, dZ_next) * \
                        A_current * (1 - A_current)
                else:
                    # Tanh derivative: 1 - A^2
                    dZ = np.matmul(W_next.T, dZ_next) * \
                        (1 - np.power(A_current, 2))

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W" + str(layer)] = \
                weights_copy["W" + str(layer)] - alpha * dW
            self.__weights["b" + str(layer)] = \
                weights_copy["b" + str(layer)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network

        Args:
            X: Input data with shape (nx, m)
            Y: Correct labels with shape (classes, m)
            iterations: Number of iterations to train over
            alpha: Learning rate
            verbose: Whether to print training information
            graph: Whether to plot training information
            step: Interval for displaying information

        Returns:
            tuple: (prediction, cost) - the final prediction and cost
        """
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

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    iterations_list.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            import matplotlib.pyplot as plt
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
        import pickle
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
        import pickle
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
