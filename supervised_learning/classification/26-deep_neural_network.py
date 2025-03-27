#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class with persistence capabilities
"""
import pickle
DNN23 = __import__('23-deep_neural_network').DeepNeuralNetwork


class DeepNeuralNetwork(DNN23):
    """
    DeepNeuralNetwork class for binary classification
    with added persistence methods
    """

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
