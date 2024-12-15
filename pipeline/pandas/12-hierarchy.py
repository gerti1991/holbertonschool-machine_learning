#!/usr/bin/env python3
"""
Test
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Test
    """
    df1 = index(df1)
    df2 = index(df2)
    df2_filtered = df2[(df2.index >= 1417411980) & (df2.index <= 1417411980)]
    df_concat = pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])
    df_concat = df_concat.swaplevel(0, 1).sort_index(level=0)
    return df_concat
