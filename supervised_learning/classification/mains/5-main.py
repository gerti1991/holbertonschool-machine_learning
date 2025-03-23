#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
Neuron = __import__('5-neuron').Neuron

# Get absolute path to project root and data file
current_dir = os.path.dirname(os.path.abspath(__file__))  # mains directory
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))  # go up 3 levels
data_path = os.path.join(project_root, 'data', 'Binary_Train.npz')

# Load data using absolute path
lib_train = np.load(data_path)
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
neuron.gradient_descent(X, Y, A, 0.5)
print(neuron.W)
print(neuron.b)
