#!/usr/bin/env python3

import numpy as np
import sys
import os

# Force Seed - fix for Keras
SEED = 8

# Add parent directory to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
build_model = __import__('0-sequential').build_model

# Get absolute path to project root and data file
current_dir = os.path.dirname(os.path.abspath(__file__))  # mains directory
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))  # go up 3 levels

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
