#!/usr/bin/env python3
"""
Calculates the sum of squares of integers from 1 to n using the formula
sum(i^2) = n(n + 1)(2n + 1) / 6.
"""


def summation_i_squared(n):
    """
    Calculates the sum of squares of integers from 1 to n using the formula
    sum(i^2) = n(n + 1)(2n + 1) / 6.
    """
    if isinstance(n, int) and n > 0:
        return (n * (n + 1) * (2 * n + 1)) // 6
    return None
