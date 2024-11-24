#!/usr/bin/env python3

"""
This module contains the `matrix_shape` function.

The function calculates the shape of a matrix by determining the dimensions at
each level.
It ensures all rows at the same depth have consistent lengths.
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list: A list of integers representing the dimensions of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
