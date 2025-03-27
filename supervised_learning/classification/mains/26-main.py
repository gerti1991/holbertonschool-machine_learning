#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
Deep = __import__('26-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

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
    X_train_3D, Y_train = lib_train['X'], lib_train['Y']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
except FileNotFoundError:
    print(f"Error: Could not find data at {data_path}")
    sys.exit(1)

np.random.seed(0)
deep = Deep(X_train.shape[0], [3, 1])
A, cost = deep.train(X_train, Y_train, iterations=400, graph=False)
deep.save('26-output')
del deep

saved = Deep.load('26-output.pkl')
A_saved, cost_saved = saved.evaluate(X_train, Y_train)

print(np.array_equal(A, A_saved) and cost == cost_saved)
