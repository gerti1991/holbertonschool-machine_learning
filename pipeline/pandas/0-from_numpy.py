#!/usr/bin/env python3
import pandas as pd
import numpy as np

def from_numpy(array):
    columns = [chr(i) for i in range(65, 65 + array.shape[1])]
    df = pd.DataFrame(array, columns=columns)
    return df
