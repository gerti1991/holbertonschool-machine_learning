#!/usr/bin/env python3

"""
This module contains the `matrix_shape` function.
The function calculates the shape of a matrix by
 determining the dimensions at each level.
It ensures all rows at the same depth have consistent lengths.
"""


def matrix_shape(matrix):

    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
