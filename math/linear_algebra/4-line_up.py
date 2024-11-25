#!/usr/bin/env python3

"""
This module contains the `add_arrays` function.

The function adds two arrays of equal length element-wise
and returns a new array.
If the input arrays are not of the same length, the function returns `None`.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list of int/float): The first array.
        arr2 (list of int/float): The second array.

    Returns:
        list of int/float: A new array containing the element-wise sums of the
        input arrays.
        None: If the input arrays are not of the same length.
    """
    add_arr = []
    if len(arr1) != len(arr2):
        return None
    else:
        count = 0
        while count < len(arr1):
            add_arr.append(int(arr1[count]) + int(arr2[count]))
            count += 1
    return add_arr
