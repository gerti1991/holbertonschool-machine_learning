#!/usr/bin/env python3
"""
Test
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Test
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
