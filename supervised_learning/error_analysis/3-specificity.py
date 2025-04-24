#!/usr/bin/env python3
"""
Module containing function to calculate specificity
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                  represent the correct labels and column indices represent
                  the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity of
        each class
    """
    # Get the number of classes from the confusion matrix shape
    classes = confusion.shape[0]

    # Initialize the specificity array
    specificity_values = np.zeros(classes)

    # Calculate specificity for each class
    for i in range(classes):
        # True negatives: sum of all elements except those in row i and column
        # i
        true_negatives = np.sum(confusion) - np.sum(confusion[i, :]) - \
            np.sum(confusion[:, i]) + confusion[i, i]

        # False positives: sum of column i excluding the true positive
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]

        # Specificity = true negatives / (true negatives + false positives)
        if true_negatives + false_positives > 0:  # Avoid division by zero
            specificity_values[i] = true_negatives / \
                (true_negatives + false_positives)

    return specificity_values
