#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

"""
Test
"""


def line():
    """
    Test
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(np.arange(0, 11), y, color='red', linestyle='-', linewidth=2)
    plt.xlim(0, 10)
    plt.show()
