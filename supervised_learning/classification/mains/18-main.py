#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
Deep = __import__('18-deep_neural_network').DeepNeuralNetwork

# Get absolute path to project root and data file
current_dir = os.path.dirname(os.path.abspath(__file__))  # mains directory
parent_dir = os.path.dirname(current_dir)  # classification directory
supervised_dir = os.path.dirname(parent_dir)  # supervised_learning directory

# Try different possible locations for the data files
possible_data_dirs = [
    os.path.join(supervised_dir, 'data'),
    os.path.join(supervised_dir, '..', 'data'),
    os.path.join(parent_dir, 'data'),
    '../data'  # Relative path that might work
]

# Find the first directory that contains the required file
data_dir = None
for directory in possible_data_dirs:
    train_path = os.path.join(directory, 'Binary_Train.npz')
    if os.path.exists(train_path):
        data_dir = directory
        break

# If no valid directory was found, use a relative path as fallback
if data_dir is None:
    data_dir = '../data'

# Set path to data file
data_path = os.path.join(data_dir, 'Binary_Train.npz')

# Load data using absolute path
try:
    lib_train = np.load(data_path)
    X_3D, Y = lib_train['X'], lib_train['Y']
    X = X_3D.reshape((X_3D.shape[0], -1)).T
except FileNotFoundError:
    print(f"Error: Could not find data at {data_path}")
    sys.exit(1)

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
deep._DeepNeuralNetwork__weights['b1'] = np.ones((5, 1))
deep._DeepNeuralNetwork__weights['b2'] = np.ones((3, 1))
deep._DeepNeuralNetwork__weights['b3'] = np.ones((1, 1))
A, cache = deep.forward_prop(X)
print(A)
print(cache)
print(cache is deep.cache)
print(A is cache['A3'])
