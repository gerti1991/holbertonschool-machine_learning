#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os


sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
convolve_grayscale_same = __import__(
    '1-convolve_grayscale_same').convolve_grayscale_same


# Get absolute path to project root and data files
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
    mnist_path = os.path.join(directory, 'MNIST.npz')
    if os.path.exists(mnist_path):
        data_dir = directory
        break

# If no valid directory was found, use a relative path as fallback
if data_dir is None:
    data_dir = '../data'

# Set path to data file
data_path = os.path.join(data_dir, 'MNIST.npz')


if __name__ == '__main__':

    dataset = np.load(data_path)
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_same(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()