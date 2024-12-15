#!/usr/bin/env python3
"""
Test
"""


def array(df):
    """
    Test
    """
    last_10_rows = df[['High', 'Close']].tail(10)
    result_array = last_10_rows.values
    return result_list
