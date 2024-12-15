#!/usr/bin/env python3
"""
Test
"""


def prune(df):
    """
    Test
    """
    data = df.dropna(subset=['Close'])
    return data
