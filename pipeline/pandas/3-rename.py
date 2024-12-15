#!/usr/bin/env python3
"""
Test
"""

import pandas as pd


def rename(df):
    """
    Test
    """
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
