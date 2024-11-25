#!/usr/bin/env python3
"""
tetete
"""


def add_matrices2D(arr1, arr2):
    """
     adds two matrices element-wise
    """
    add_array = []
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        return None
    r = 0
    c = 0
    for row in arr1:
        temp = []
        for i in row:
            temp.append(i + arr2[r][c])
            c += 1
        add_array.append(temp)
        c = 0
        r += 1
    return add_array
