#!/usr/bin/env python3
"""
Test
"""


def array(df):
    """
    Test
    """
    result = df[['High', 'Close']].tail(10).values
    return result
