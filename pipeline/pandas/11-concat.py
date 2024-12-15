#!/usr/bin/env python3
"""
Test
"""


def concat(df1, df2):
    """
    Test
    """
    df_concat = pd.concat(
                        [df2[df2['Timestamp'] <= 1417411920], df1],
                        keys=['bitstamp', 'coinbase']
                        )
    df = df_concat
    return df
