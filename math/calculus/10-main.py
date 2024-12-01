#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
poly2 = []
poly3 = [10]
poly4  = [1,4]
print(poly_derivative(poly))
print(poly_derivative(poly2))
print(poly_derivative(poly3))
print(poly_derivative(poly4))