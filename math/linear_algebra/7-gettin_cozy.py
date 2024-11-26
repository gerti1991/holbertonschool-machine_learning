#!/usr/bin/env python3
"""
hshshsh
"""


def cat_matrices2D(arr1, arr2, axis=0):
    """
    jsjsjjs
    """
    if axis == 0:
        if len(arr1[0]) != len(arr2[0]):
            return None
        cont = arr1.copy() + arr2.copy()
        return cont
    elif axis == 1:
        if len(arr1) != len(arr2):
            return None
        cont = [arr1[i] + arr2[i] for i in range(len(arr1))]
        return cont
    None
