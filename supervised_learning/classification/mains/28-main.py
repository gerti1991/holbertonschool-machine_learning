#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
Deep27 = __import__('27-deep_neural_network').DeepNeuralNetwork
Deep28 = __import__('28-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

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

# Check for saved model paths
saved_model_path27 = os.path.join(parent_dir, '27-output.pkl')
if not os.path.exists(saved_model_path27):
    saved_model_path27 = '27-output.pkl'  # Try relative path

saved_model_path28 = os.path.join(data_dir, '28-saved.pkl')
if not os.path.exists(saved_model_path28):
    saved_model_path28 = '28-saved.pkl'  # Try relative path

# Load data using absolute path
try:
    lib = np.load(data_path)
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_test_3D = lib['X_test']
    Y_test = lib['Y_test']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
    X_test = X_test_3D.reshape((X_test_3D.shape[0], -1)).T
    Y_train_one_hot = one_hot_encode(Y_train, 10)
    Y_valid_one_hot = one_hot_encode(Y_valid, 10)
    Y_test_one_hot = one_hot_encode(Y_test, 10)
except FileNotFoundError:
    print(f"Error: Could not find data at {data_path}")
    sys.exit(1)

print('Sigmoid activation:')
try:
    deep27 = Deep27.load(saved_model_path27)
    if deep27 is None:
        print(f"Error: Could not load model from {saved_model_path27}")
        sys.exit(1)
except Exception as e:
    print(f"Error loading sigmoid model: {e}")
    sys.exit(1)

A_one_hot27, cost27 = deep27.evaluate(X_train, Y_train_one_hot)
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_train == A27) / Y_train.shape[0] * 100
print("Train cost:", cost27)
print("Train accuracy: {}%".format(accuracy27))
A_one_hot27, cost27 = deep27.evaluate(X_valid, Y_valid_one_hot)
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_valid == A27) / Y_valid.shape[0] * 100
print("Validation cost:", cost27)
print("Validation accuracy: {}%".format(accuracy27))
A_one_hot27, cost27 = deep27.evaluate(X_test, Y_test_one_hot)
A27 = one_hot_decode(A_one_hot27)
accuracy27 = np.sum(Y_test == A27) / Y_test.shape[0] * 100
print("Test cost:", cost27)
print("Test accuracy: {}%".format(accuracy27))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A27[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

print('\nTanh activation:')
try:
    deep28 = Deep28.load(saved_model_path28)
    if deep28 is None:
        print(f"Error: Could not load model from {saved_model_path28}")
        sys.exit(1)
except Exception as e:
    print(f"Error loading tanh model: {e}")
    sys.exit(1)

A_one_hot28, cost28 = deep28.train(X_train, Y_train_one_hot, iterations=100,
                               step=10, graph=False)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_train == A28) / Y_train.shape[0] * 100
print("Train cost:", cost28)
print("Train accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_valid, Y_valid_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_valid == A28) / Y_valid.shape[0] * 100
print("Validation cost:", cost28)
print("Validation accuracy: {}%".format(accuracy28))
A_one_hot28, cost28 = deep28.evaluate(X_test, Y_test_one_hot)
A28 = one_hot_decode(A_one_hot28)
accuracy28 = np.sum(Y_test == A28) / Y_test.shape[0] * 100
print("Test cost:", cost28)
print("Test accuracy: {}%".format(accuracy28))
deep28.save('28-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A28[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
