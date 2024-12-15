#!/usr/bin/env python3
"""
Test
"""

index = __import__('10-index').index
import pandas as pd


def concat(df1, df2):
    """
    Test
    """
    df1 = index(df1)
    df2 = index(df2)
    df2_filtered = df2[df2.index <= 1417411920]
    df_concat = pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])
    return df_concat
