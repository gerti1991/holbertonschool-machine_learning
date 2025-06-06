#!/usr/bin/env python3
"""
Module that defines a neural network with one hidden layer
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """Evaluates network predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Updates weights and biases with gradient descent"""
        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network

        Args:
            X: Input data
            Y: Correct labels
            iterations: Number of iterations to train
            alpha: Learning rate
            verbose: Whether to print training progress
            graph: Whether to plot training progress
            step: Interval for displaying/plotting progress

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

        # Calculate initial cost (before any training)
        A1, A2 = self.forward_prop(X)
        initial_cost = self.cost(Y, A2)

        # Print initial cost if verbose
        if verbose:
            print(f"Cost after 0 iterations: {initial_cost}")

        # Store initial cost for graphing
        if graph:
            costs.append(initial_cost)
            iterations_list.append(0)

        # Training loop
        for i in range(1, iterations + 1):
            # Forward propagation and gradient descent
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

            # Print cost at specified steps if verbose is True
            if verbose and (i % step == 0 or i == iterations):
                cost = self.cost(Y, A2)
                print(f"Cost after {i} iterations: {cost}")

            # Store cost for graphing if graph is True
            if graph and (i % step == 0 or i == iterations):
                costs.append(self.cost(Y, A2))
                iterations_list.append(i)

        # Create graph if graph is True
        if graph:
            plt.plot(iterations_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Return evaluation of the training data
        return self.evaluate(X, Y)
