#!/usr/bin/env python3
"""
Test
"""

import numpy as np


def array(df):
    """
    Test
    """
    A = (df[['High', 'Close']].tail(10)).to_numpy()
    return A
