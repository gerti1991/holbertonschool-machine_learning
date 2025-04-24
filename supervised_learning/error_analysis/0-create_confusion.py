#!/usr/bin/env python3
"""
Module containing function to create confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes) containing
                the correct labels for each data point
        logits: one-hot numpy.ndarray of shape (m, classes) containing
                the predicted labels

    Returns:
        confusion matrix as a numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
    """
    # Get the true class indices (convert from one-hot to class index)
    true_classes = np.argmax(labels, axis=1)

    # Get the predicted class indices (convert from one-hot to class index)
    pred_classes = np.argmax(logits, axis=1)

    # Get the number of classes
    classes = labels.shape[1]

    # Create and populate confusion matrix
    confusion = np.zeros((classes, classes))

    # For each pair of (true, predicted) class indices, increment the
    # corresponding cell in the confusion matrix
    for true_class, pred_class in zip(true_classes, pred_classes):
        confusion[true_class, pred_class] += 1

    return confusion
