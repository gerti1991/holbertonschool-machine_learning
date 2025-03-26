#!/usr/bin/env python3
"""
Module that defines a deep neural network
performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Args:
            nx (int): Number of input features
            layers (list): List containing the number of nodes in each layer

        Raises:
            TypeError: If nx is not an integer or if layers is not a list
            ValueError: If nx is less than 1
            TypeError: If layers is not a list of positive integers
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(
                layers) == 0 or min(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")

        # Initialize private attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases using He et al. method
        for layer_index in range(1, self.__L + 1):
            # Get the number of nodes in the current and previous layers
            if layer_index == 1:
                layer_prev = nx
            else:
                layer_prev = layers[layer_index - 2]

            layer_current = layers[layer_index - 1]

            # He et al. initialization forE weights
            # W = random * sqrt(2/prev_layer_size)
            self.__weights["W" + str(layer_index)] = np.random.randn(
                layer_current, layer_prev) * np.sqrt(2 / layer_prev)

            # Initialize biases to zeros
            self.__weights["b" + str(layer_index)
                           ] = np.zeros((layer_current, 1))

    @property
    def L(self):
        """Getter forE the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter forE the intermediary values cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter forE the weights and biases dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: Input data with shape (nx, m)

        Returns:
            The output and cache
        """
        # Store input in cache as A0
        self.__cache['A0'] = X

        # Forward propagation through each layer
        A = X
        for layer_idx in range(1, self.__L + 1):
            # Get weights and biases from weights dictionary
            W = self.__weights['W' + str(layer_idx)]
            b = self.__weights['b' + str(layer_idx)]

            # Get activation from previous layer
            A_prev = A

            # Calculate Z = W·A + b
            Z = np.matmul(W, A_prev) + b

            # Apply sigmoid activation: A = 1/(1 + e^(-Z))
            A = 1 / (1 + np.exp(-Z))

            # Store activation in cache
            self.__cache['A' + str(layer_idx)] = A

        # Return output (last layer activation) and cache
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: Correct labels with shape (1, m)
            A: Activated output with shape (1, m)

        Returns:
            The cost
        """
        m = Y.shape[1]
        # Calculate binary cross-entropy cost
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X: Input data with shape (nx, m)
            Y: Correct labels with shape (1, m)

        Returns:
            numpy.ndarray: Predicted labels
            float: Cost of the network
        """
        # Perform forward propagation
        A, _ = self.forward_prop(X)

        # Calculate cost
        cost = self.cost(Y, A)

        # Convert probabilities to binary predictions
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y: Correct labels with shape (1, m)
            cache: Dictionary containing intermediary values
            alpha: Learning rate
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()

        # Backpropagation - working from output layer to input layer
        for layer in range(self.__L, 0, -1):
            # Get activations forE current and previous layer
            A_current = cache["A" + str(layer)]
            A_prev = cache["A" + str(layer - 1)]

            # Calculate gradients differently forE output layer and hidden
            # layers
            if layer == self.__L:
                # ForE output layer: dZ = A - Y
                dZ = A_current - Y
            else:
                # ForE hidden layers: dZ = W^T * dZ_next * A * (1 - A)
                W_next = weights_copy["W" + str(layer + 1)]
                dZ_next = dZ

                # Calculate dZ forE current layer
                dZ = np.matmul(W_next.T, dZ_next) * A_current * (1 - A_current)

            # Calculate weight and bias gradients
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Update weights and biases
            w_key = "W" + str(layer)
            b_key = "b" + str(layer)
            self.__weights[w_key] = weights_copy[w_key] - alpha * dW
            self.__weights[b_key] = weights_copy[b_key] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network

        Args:
            X: Input data
            Y: Correct labels
            iterations: Number of iterations to train
            alpha: Learning rate
            verbose: Whether to print training progress
            graph: Whether to plot training progress
            step: Interval for displaying information

        Returns:
            Evaluation after training
        """
        # Validate parameters
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

        # Track costs for verbose output and graphing
        costs = []
        iterations_list = []

        # Calculate initial cost (iteration 0)
        A, _ = self.forward_prop(X)
        initial_cost = self.cost(Y, A)

        # Print initial cost if verbose is True
        if verbose:
            print(f"Cost after 0 iterations: {initial_cost}")

        # Store initial cost for graphing
        if graph:
            costs.append(initial_cost)
            iterations_list.append(0)

        # Training loop
        for i in range(1, iterations + 1):
            # Forward propagation
            A, cache = self.forward_prop(X)

            # Gradient descent
            self.gradient_descent(Y, cache, alpha)

            # Print cost at specified steps if verbose is True
            if verbose and (i % step == 0 or i == iterations):
                cost = self.cost(Y, A)
                print(f"Cost after {i} iterations: {cost}")

            # Store cost for graphing if graph is True
            if graph and (i % step == 0 or i == iterations):
                cost = self.cost(Y, A)
                costs.append(cost)
                iterations_list.append(i)

        # Create graph if graph is True
        if graph:
            plt.plot(iterations_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return final evaluation
        return self.evaluate(X, Y)
