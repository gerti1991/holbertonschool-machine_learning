#!/usr/bin/env python3
"""
Test
"""

import pandas as pd


def array(df):
    """
    Test
    """
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
    A = (df[['High', 'Close']].tail(10)).to_numpy()
    return A
