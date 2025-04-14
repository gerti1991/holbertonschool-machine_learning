#!/usr/bin/env python3
"""
Module for calculating the weighted moving average
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.

    Parameters:
        data (list): The list of data to calculate the moving average of
        beta (float): The weight used for the moving average

    Returns:
        list: A list containing the moving averages of data
    """
    # Initialize variables
    v = 0
    moving_averages = []

    # Loop through the data points
    for i in range(len(data)):
        # Calculate the weighted average
        v = beta * v + (1 - beta) * data[i]

        # Apply bias correction: v / (1 - beta^(t+1))
        bias_correction = 1 - beta ** (i + 1)
        moving_averages.append(v / bias_correction)

    return moving_averages
