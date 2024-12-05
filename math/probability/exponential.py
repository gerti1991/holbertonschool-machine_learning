#!/usr/bin/env python3
"""
Exponential module
This module defines the Exponential class for working with
Exponential distributions.
"""


class Exponential:
    """Class representing an Exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential distribution.
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
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period x.
        Args:
            x (float): Time period for which PDF is calculated.
        Returns:
            float: PDF value for x.
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * e ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period x.
        Args:
            x (float): Time period for which CDF is calculated.
        Returns:
            float: CDF value for x.
        """
        if x < 0:
            return 0
        e = 2.7182818285
        return 1 - e ** (-self.lambtha * x)
