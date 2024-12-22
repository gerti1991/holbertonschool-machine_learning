#!/usr/bin/env python3
"""
Test
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Test
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    labels = ['Farrah', 'Fred', 'Felicia']
    apples, bananas, oranges, peaches = fruit
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    width = 0.5
    plt.bar(labels, apples, color=colors[0], width=width, label="apples")
    plt.bar(
            labels,
            bananas,
            bottom=apples,
            color=colors[1],
            width=width,
            label="bananas"
            )
    plt.bar(
            labels,
            oranges,
            bottom=apples + bananas,
            color=colors[2],
            width=width,
            label="oranges"
            )
    plt.bar(
            labels,
            peaches,
            bottom=apples + bananas + oranges,
            color=colors[3],
            width=width,
            label="peaches"
            )
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.legend()
    plt.show()
