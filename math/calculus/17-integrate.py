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
        return [C]
    int_poly = [0]
    for power, coff in enumerate(poly):
        n_coff = coff/(power+1)
        if n_coff - int(n_coff) == 0:
            int_poly.append(int(n_coff))
        else:
            int_poly.append(n_coff)
    return int_poly
