#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
update_variables_RMSProp = __import__('7-RMSProp').update_variables_RMSProp

def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m

    return cost

if __name__ == '__main__':
    # Create a simpler test case
    np.random.seed(0)
    
    # Create synthetic data
    X = np.random.randn(5, 3)
    Y = np.array([[0, 1, 0, 1, 0]]).T
    
    # Initialize parameters
    W = np.random.randn(3, 1)
    b = 0
    s_dW = np.zeros((3, 1))
    s_db = 0
    
    print("Initial W:", W)
    print("Initial b:", b)
    print("Initial s_dW:", s_dW)
    print("Initial s_db:", s_db)
    
    # Perform a few iterations of RMSProp
    alpha = 0.01
    beta2 = 0.9
    epsilon = 1e-8
    
    print("\nRunning RMSProp for 5 iterations:")
    
    for i in range(5):
        # Forward propagation
        A = forward_prop(X, W, b)
        
        # Calculate cost
        cost = calculate_cost(Y, A)
        print(f"Cost after iteration {i}: {cost}")
        
        # Calculate gradients
        dW, db = calculate_grads(Y, A, W, b)
        
        # Update variables using RMSProp
        W, s_dW = update_variables_RMSProp(alpha, beta2, epsilon, W, dW, s_dW)
        b, s_db = update_variables_RMSProp(alpha, beta2, epsilon, b, db, s_db)
        
    print("\nFinal W:", W)
    print("Final b:", b)
    print("Final s_dW:", s_dW)
    print("Final s_db:", s_db)
    
    # Verify with manual calculation for the first iteration
    print("\nVerification:")
    
    # Reset to initial values
    np.random.seed(0)
    X = np.random.randn(5, 3)
    Y = np.array([[0, 1, 0, 1, 0]]).T
    W = np.random.randn(3, 1)
    b = 0
    s_dW = np.zeros((3, 1))
    s_db = 0
    
    # Forward propagation
    A = forward_prop(X, W, b)
    
    # Calculate gradients
    dW, db = calculate_grads(Y, A, W, b)
    
    # Manual RMSProp update
    s_dW_manual = beta2 * s_dW + (1 - beta2) * (dW ** 2)
    W_manual = W - alpha * dW / (np.sqrt(s_dW_manual) + epsilon)
    
    s_db_manual = beta2 * s_db + (1 - beta2) * (db ** 2)
    b_manual = b - alpha * db / (np.sqrt(s_db_manual) + epsilon)
    
    # Update with our function
    W_updated, s_dW_updated = update_variables_RMSProp(alpha, beta2, epsilon, W, dW, s_dW)
    b_updated, s_db_updated = update_variables_RMSProp(alpha, beta2, epsilon, b, db, s_db)
    
    # Compare
    print("Manual W update:", W_manual)
    print("Function W update:", W_updated)
    print("W difference:", np.sum(np.abs(W_manual - W_updated)))
    
    print("Manual b update:", b_manual)
    print("Function b update:", b_updated)
    print("b difference:", np.abs(b_manual - b_updated))