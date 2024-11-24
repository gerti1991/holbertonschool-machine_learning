#!/usr/bin/env python3

"""
Returns the transpose of a 2D matrix.

Args:
    matrix (list of lists): A 2D matrix to transpose.

Returns:
    list of lists: A new 2D matrix representing the transpose
        of the input matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
        matrix (list of lists): A 2D matrix to transpose.

    Returns:
        list of lists: A new 2D matrix representing the transpose
         of the input matrix.
    """
    shape = []

    for row in matrix:
        shape.append(len(row))

    if len(set(shape)) != 1:
        return "Error you put a list with different shapes"
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
