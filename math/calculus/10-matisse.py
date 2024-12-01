#!/usr/bin/env python3
"""
Derive happiness in oneself from a good day's work
"""


def poly_derivative(poly):
    """
    Derive happiness in oneself from a good day's work
    """
    if not isinstance(poly, list):
        return None
    if len(poly) <= 1:
        return [0]
    der_poly = [poly[i] * i for i in range(1, len(poly))]
    return der_poly
