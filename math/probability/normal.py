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
            if stddev <= 9:
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
            self.stddev = s / len(data)
