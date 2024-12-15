#!/usr/bin/env python3
"""
Test
"""


def flip_switch(df):
    """
    Test
    """
    data = df.sort_values(by='Timestamp', ascending=False).transpose()
    return data
