#!/usr/bin/env python3
"""
Test
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Test
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(
            student_grades,
            bins=[x * 10 for x in range(0, 11)],
            edgecolor='black'
            )
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks([x * 10 for x in range(0, 11)])
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()
