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
