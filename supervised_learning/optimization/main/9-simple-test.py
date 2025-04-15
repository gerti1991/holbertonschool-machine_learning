#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add parent directory to path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
update_variables_Adam = __import__('9-Adam').update_variables_Adam

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
    v_dW = np.zeros((3, 1))
    v_db = 0
    s_dW = np.zeros((3, 1))
    s_db = 0
    
    print("Initial W:", W)
    print("Initial b:", b)
    print("Initial v_dW:", v_dW)
    print("Initial v_db:", v_db)
    print("Initial s_dW:", s_dW)
    print("Initial s_db:", s_db)
    
    # Perform a few iterations of Adam
    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.99
    epsilon = 1e-8
    
    print("\nRunning Adam for 5 iterations:")
    
    for t in range(1, 6):
        # Forward propagation
        A = forward_prop(X, W, b)
        
        # Calculate cost
        cost = calculate_cost(Y, A)
        print(f"Cost after iteration {t}: {cost}")
        
        # Calculate gradients
        dW, db = calculate_grads(Y, A, W, b)
        
        # Update variables using Adam
        W, v_dW, s_dW = update_variables_Adam(alpha, beta1, beta2, epsilon, W, dW, v_dW, s_dW, t)
        b, v_db, s_db = update_variables_Adam(alpha, beta1, beta2, epsilon, b, db, v_db, s_db, t)
        
    print("\nFinal W:", W)
    print("Final b:", b)
    print("Final v_dW:", v_dW)
    print("Final v_db:", v_db)
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
    v_dW = np.zeros((3, 1))
    v_db = 0
    s_dW = np.zeros((3, 1))
    s_db = 0
    t = 1
    
    # Forward propagation
    A = forward_prop(X, W, b)
    
    # Calculate gradients
    dW, db = calculate_grads(Y, A, W, b)
    
    # Manual Adam update
    v_dW_new = beta1 * v_dW + (1 - beta1) * dW
    s_dW_new = beta2 * s_dW + (1 - beta2) * (dW ** 2)
    
    v_dW_corrected = v_dW_new / (1 - beta1 ** t)
    s_dW_corrected = s_dW_new / (1 - beta2 ** t)
    
    W_manual = W - alpha * v_dW_corrected / (np.sqrt(s_dW_corrected) + epsilon)
    
    # Update with our function
    W_updated, v_dW_updated, s_dW_updated = update_variables_Adam(alpha, beta1, beta2, epsilon, W, dW, v_dW, s_dW, t)
    
    # Compare
    print("Manual W update:", W_manual)
    print("Function W update:", W_updated)
    print("W difference:", np.sum(np.abs(W_manual - W_updated)))