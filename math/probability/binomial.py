#!/usr/bin/env python3
"""
Binomial distribution
"""


class Binomial:
    """
    Binomial distribution class
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Init
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def fac(self, x):
        """
        factorial function
        """
        if x == 0 or x == 1:
            return 1
        n = 1
        for i in range(2, x+1):
            n *= i
        return n

    def pmf(self, k):
        """
        pmf
        """
        k = int(k)
        n = self.n
        p = self.p
        delta = n - k
        if k < 0 or k > n:
            return 0
        n_k = self.fac(n) / (self.fac(k) * self.fac(delta))
        pmf = n_k * (p ** k) * ((1 - p) ** delta)
        return pmf

    def cdf(self, k):
        k = int(k)
        cdf = sum(self.pmf(i) for i in range(k+1))
        return cdf
