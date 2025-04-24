#!/usr/bin/env python3
"""
Module containing function to calculate sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                  represent the correct labels and column indices represent
                  the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity of
        each class
    """
    # Get the number of classes from the confusion matrix shape
    classes = confusion.shape[0]

    # Initialize the sensitivity array
    sensitivity_values = np.zeros(classes)

    # Calculate sensitivity for each class
    for i in range(classes):
        # True positives are on the diagonal of the confusion matrix
        true_positives = confusion[i, i]

        # All actual positives for this class (sum of the entire row)
        all_actual_positives = np.sum(confusion[i, :])

        # Sensitivity = true positives / all actual positives
        if all_actual_positives > 0:  # Avoid division by zero
            sensitivity_values[i] = true_positives / all_actual_positives

    return sensitivity_values
