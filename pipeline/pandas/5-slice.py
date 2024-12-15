#!/usr/bin/env python3
"""
Test
"""


def slice(df):
    """
    test
    """
    data = df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
    return data
