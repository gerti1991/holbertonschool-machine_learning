#!/usr/bin/env python3
"""
Test
"""


def array(df):
    """
    Test
    """
    high_values = df['High'].tail(10).values
    close_values = df['Close'].tail(10).values
    result = [[high_values[i], close_values[i]] for i in range(10)]
    return result
