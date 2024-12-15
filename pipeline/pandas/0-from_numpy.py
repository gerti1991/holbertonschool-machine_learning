#!/usr/bin/env python3
"""
Comment
"""

import pandas as pd


def from_numpy(array):
    """
    comment
    """
    columns = [chr(i) for i in range(65, 65 + array.shape[1])]
    df = pd.DataFrame(array, columns=columns)
    return df
