#!/usr/bin/env python3
"""
This module provides a function to create a DataFrame-like structure from a numpy ndarray.
"""

def from_numpy(array):
    """
    Creates a DataFrame-like structure from a numpy ndarray.
    
    Args:
        array (np.ndarray): The numpy array to convert to a DataFrame-like structure.
    
    Returns:
        dict: A dictionary where keys are column labels and values are lists of column data.
    """
    # Generate column labels (A, B, C, ..., Z) for the "DataFrame"
    columns = [chr(65 + i) for i in range(array.shape[1])]
    
    # Create a dictionary where keys are column labels and values are lists of column data
    df_like = {columns[i]: array[:, i].tolist() for i in range(len(columns))}
    
    return df_like
