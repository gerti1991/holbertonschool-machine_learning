#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
NN = __import__('15-neural_network').NeuralNetwork

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

# Find the first directory that contains the required files
data_dir = None
for directory in possible_data_dirs:
    train_path = os.path.join(directory, 'Binary_Train.npz')
    dev_path = os.path.join(directory, 'Binary_Dev.npz')
    if os.path.exists(train_path) and os.path.exists(dev_path):
        data_dir = directory
        break

# If no valid directory was found, use a relative path as fallback
if data_dir is None:
    data_dir = '../data'

# Set paths to data files
data_path_train = os.path.join(data_dir, 'Binary_Train.npz')
data_path_dev = os.path.join(data_dir, 'Binary_Dev.npz')

# Load training data
try:
    lib_train = np.load(data_path_train)
    X_train_3D, Y_train = lib_train['X'], lib_train['Y']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
except FileNotFoundError:
    print(f"Error: Could not find training data at {data_path_train}")
    sys.exit(1)

# Load development data
try:
    lib_dev = np.load(data_path_dev)
    X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
    X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T
except FileNotFoundError:
    print(f"Error: Could not find development data at {data_path_dev}")
    sys.exit(1)

np.random.seed(0)
nn = NN(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
