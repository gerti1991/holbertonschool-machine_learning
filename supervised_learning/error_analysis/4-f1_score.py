#!/usr/bin/env python3
"""
Module containing function to calculate F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                  represent the correct labels and column indices represent
                  the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score of
        each class
    """
    # Get the sensitivity (recall) from the previously created function
    sens = sensitivity(confusion)

    # Get the precision from the previously created function
    prec = precision(confusion)

    # Calculate F1 score for each class: 2 * (precision * sensitivity)
    # / (precision + sensitivity)
    # Handle division by zero cases
    f1_scores = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        if prec[i] + sens[i] > 0:  # Avoid division by zero
            f1_scores[i] = 2 * (prec[i] * sens[i]) / (prec[i] + sens[i])

    return f1_scores
