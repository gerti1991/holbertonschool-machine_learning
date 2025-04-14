#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add parent directory to path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
create_mini_batches = __import__('3-mini_batch').create_mini_batches

if __name__ == '__main__':
    # Create a small test dataset
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    Y = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Test with different batch sizes
    print("Mini-batches with batch_size=2:")
    mini_batches = create_mini_batches(X, Y, 2)
    for i, (X_batch, Y_batch) in enumerate(mini_batches):
        print(f"Mini-batch {i+1}:")
        print("X_batch:", X_batch)
        print("Y_batch:", Y_batch)
        print()
    
    print("\nMini-batches with batch_size=3:")
    mini_batches = create_mini_batches(X, Y, 3)
    for i, (X_batch, Y_batch) in enumerate(mini_batches):
        print(f"Mini-batch {i+1}:")
        print("X_batch:", X_batch)
        print("Y_batch:", Y_batch)
        print()
        
    # Test with a batch size that doesn't divide the dataset evenly
    print("\nMini-batches with batch_size=5 (uneven division):")
    mini_batches = create_mini_batches(X, Y, 5)
    for i, (X_batch, Y_batch) in enumerate(mini_batches):
        print(f"Mini-batch {i+1}:")
        print("X_batch shape:", X_batch.shape)
        print("Y_batch shape:", Y_batch.shape)
        print("X_batch:", X_batch)
        print("Y_batch:", Y_batch)
        print()