#!/usr/bin/env python3
"""
jdjd
"""


def mat_mul(arr1, arr2):
    """
    dsf
    """
    if len(arr1[0]) != len(arr2):
        return None
    arr2_tr = [[row[i] for row in arr2]for i in range(len(arr2[0]))]
    mult_arr = []
    for row in arr1:
        new_row = []
        for col in arr2_tr:
            new_row.append(sum(row[i] * col[i] for i in range(len(row))))
        mult_arr.append(new_row)
    return mult_arr
