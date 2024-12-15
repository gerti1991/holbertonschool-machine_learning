#!/usr/bin/env python3
"""
Test
"""


def array(df):
    """
    Test
    """
    last_10 = df[['High', 'Close']][-10:]
    result_list = [list(row) for row in last_10]
    return result_list
