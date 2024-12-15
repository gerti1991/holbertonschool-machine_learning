#!/usr/bin/env python3
"""
Test
"""

import numpy as np

def array(df):
    """
    Test
    """
    high_values = df['High'].tail(10).values
    close_values = df['Close'].tail(10).values
    result = np.column_stack((high_values, close_values))
    return result
