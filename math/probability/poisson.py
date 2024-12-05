#!/usr/bin/env python3
"""
Poisson module
This module defines the Poisson class for working with Poisson distributions.
"""


class Poisson:
    """Class representing a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson distribution.
        Args:
            data (list, optional): List of data points to estimate lambtha.
            lambtha (float): Expected number of occurrences in a time frame.
        Raises:
            ValueError: If lambtha is not positive or if data has fewer than
            two elements.
            TypeError: If data is not a list.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        """
        factorial
        """
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def pmf(self, k):
        """
        pmf
        """
        k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        fact = self.factorial(k)
        lam = self.lambtha
        PMF = (e**(-1*(lam))*lam**(k)) / fact
        return PMF

    def cdf(self, k):
        """
        cdf
        """
        k = int(k)
        if k < 0:
            return 0
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
