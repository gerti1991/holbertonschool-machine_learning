#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import sys

# Add parent directory to path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

try:
    # Try to load the MNIST dataset
    lib = np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    # Try to load the model
    try:
        model = tf.keras.models.load_model('../data/model.h5', compile=False)
    except:
        # If the model doesn't exist, create a simple test model
        print("Model not found, creating a test model")
        inputs = tf.keras.Input(shape=(X.shape[1],))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    alpha = 0.1
    alpha_schedule = learning_rate_decay(alpha, 1, 10)
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha_schedule)

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy()(labels, predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    total_iterations = 100
    for iteration in range(total_iterations):
        current_learning_rate = alpha_schedule(iteration).numpy()
        print(current_learning_rate)
        # Skip actual training to speed up the test
        # cost = train_step(X, Y_oh)

except Exception as e:
    # If we can't load the dataset or model, just show learning rate decay without training
    print(f"Loading dataset or model failed: {e}")
    print("Showing learning rate decay without training...")

    alpha = 0.1
    alpha_schedule = learning_rate_decay(alpha, 1, 10)
    
    total_iterations = 100
    for iteration in range(total_iterations):
        current_learning_rate = alpha_schedule(iteration).numpy()
        print(current_learning_rate)