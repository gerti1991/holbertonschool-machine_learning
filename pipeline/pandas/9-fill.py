#!/usr/bin/env python3
"""
Test
"""


def fill(df):
    """
    Test
    """
    df = df.drop(columns=['Weighted_Price'])
    df['Close'].fillna(method='ffill', inplace=True)
    df['High'].fillna(df['Close'], inplace=True)
    df['Low'].fillna(df['Close'], inplace=True)
    df['Open'].fillna(df['Close'], inplace=True)
    df['Volume_(BTC)'].fillna(0, inplace=True)
    df['Volume_(Currency)'].fillna(0, inplace=True)
    return df
