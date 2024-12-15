#!/usr/bin/env python3
import pandas as pd
"""
Test
"""


def from_file(filename, delimiter):
    """
    Test
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
