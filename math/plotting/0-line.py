#!/usr/bin/env python3
"""
Test
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Test
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(np.arange(0, 11), y, color='red')
    plt.xlim(0, 10)
    plt.show()
