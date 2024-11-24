#!/usr/bin/env python3

def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix (list): A list of lists representing the matrix.
    Returns:
        list: A list of integers representing the dimensions of the matrix.
        If rows are inconsistent in length, the function
        returns "Inconsistent row lengths".
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
