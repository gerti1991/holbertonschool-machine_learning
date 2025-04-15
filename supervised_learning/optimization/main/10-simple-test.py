#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import sys

# Add parent directory to path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
create_Adam_op = __import__('10-Adam').create_Adam_op

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

# Simple test function to verify the optimizer works without the full model
def test_optimizer():
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # Create sample data
    X = np.random.randn(100, 5).astype(np.float32)
    Y = np.random.randint(0, 5, size=(100,))
    Y_oh = one_hot(Y, 5)
    
    # Create our Adam optimizer
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-7
    optimizer = create_Adam_op(alpha, beta1, beta2, epsilon)
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train for a few steps
    history = model.fit(X, Y_oh, epochs=5, verbose=1)
    
    print("\nOptimization Test Results:")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    # Verify the optimizer parameters
    print("\nOptimizer Configuration:")
    print(f"Learning rate: {optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate}")
    print(f"Beta_1: {optimizer.beta_1}")
    print(f"Beta_2: {optimizer.beta_2}")
    print(f"Epsilon: {optimizer.epsilon}")
    
    return history.history['loss'][-1] < history.history['loss'][0]

if __name__ == "__main__":
    print("Testing Adam optimizer...")
    success = test_optimizer()
    
    if success:
        print("\n✅ Test passed: Loss decreased during training")
    else:
        print("\n❌ Test failed: Loss did not decrease during training")