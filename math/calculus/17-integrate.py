#!/usr/bin/env python3
"""
Integrate happiness in oneself from a good day's work
"""


def poly_integral(poly, C=0):
    """
    Integrate happiness in oneself from a good day's work
    """
    if not isinstance(poly, list):
        return None
    if len(poly) == 0:
        return None
    int_poly = [C]
    for power, coff in enumerate(poly):
        int_poly.append(coff/(power+1))
    return int_poly
