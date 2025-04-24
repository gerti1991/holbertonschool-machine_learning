#!/usr/bin/env python3
"""
Module containing function to calculate precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                  represent the correct labels and column indices represent
                  the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the precision of
        each class
    """
    # Get the number of classes from the confusion matrix shape
    classes = confusion.shape[0]

    # Initialize the precision array
    precision_values = np.zeros(classes)

    # Calculate precision for each class
    for i in range(classes):
        # True positives are on the diagonal of the confusion matrix
        true_positives = confusion[i, i]

        # All predicted positives for this class (sum of the entire column)
        all_predicted_positives = np.sum(confusion[:, i])

        # Precision = true positives / all predicted positives
        if all_predicted_positives > 0:  # Avoid division by zero
            precision_values[i] = true_positives / all_predicted_positives

    return precision_values
