#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
NN = __import__('12-neural_network').NeuralNetwork

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
nn = NN(X.shape[0], 3)
A, cost = nn.evaluate(X, Y)
print(A)
print(cost)
