#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import sys

# Add parent directory to path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
create_RMSProp_op = __import__('8-RMSProp').create_RMSProp_op

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

# Check for data file location
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
if os.path.exists(os.path.join(data_dir, 'MNIST.npz')):
    # Load from data directory
    lib = np.load(os.path.join(data_dir, 'MNIST.npz'))
else:
    # Try current directory
    try:
        lib = np.load('MNIST.npz')
    except FileNotFoundError:
        print("Error: MNIST.npz file not found")
        print("Please provide the file or use the simplified test file: 8-simple-test.py")
        sys.exit(1)

X_3D = lib['X_train']
Y = lib['Y_train']
X = X_3D.reshape((X_3D.shape[0], -1))
Y_oh=one_hot(Y,10)

# Check for model file location
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/model.h5'))
if os.path.exists(model_path):
    # Load from data directory
    model = tf.keras.models.load_model(model_path, compile=False)
else:
    # Try current directory
    try:
        model = tf.keras.models.load_model('model.h5', compile=False)
    except:
        print("Error: model.h5 file not found")
        print("Please provide the file or use the simplified test file: 8-simple-test.py")
        sys.exit(1)

optimizer=create_RMSProp_op(0.001, 0.9, 1e-07)

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

total_iterations = 1000
for iteration in range(total_iterations):

    cost = train_step(X, Y_oh)

    if (iteration + 1) % 100 == 0:
        print(f'Cost after {iteration + 1} iterations: {cost}')

Y_pred_oh = model(X[:100])
Y_pred = np.argmax(Y_pred_oh, axis=1)

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(str(Y_pred[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()