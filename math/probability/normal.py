#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal:
    """
    Normal distribution class
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Init
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            mean = self.mean
            s = 0
            for i in data:
                s += (i - mean) ** 2
            self.stddev = (s / len(data)) ** 0.5

    def z_score(self, x):
        """
        Z score
        """
        return float((x - self.mean) / self.stddev)

    def x_value(self, z):
        """
        Z score value
        """
        return float(z * self.stddev + self.mean)

    def pdf(self, x):
        """
        PDF
        """
        pi = 3.1415926536
        e = 2.7182818285
        stddev = self.stddev
        part2 = ((2 * pi) ** 0.5) * stddev
        p = -1 * ((x - self.mean) ** 2) / (2 * (stddev ** 2))
        return float((1 / part2) * (e ** p))
