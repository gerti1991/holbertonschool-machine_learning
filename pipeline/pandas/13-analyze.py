#!/usr/bin/env python3
"""
Test
"""


def analyze(df):
    """
    Test
    """
    df = df.drop(columns=['Timestamp'])
    stats = df.describe()
    return stats
