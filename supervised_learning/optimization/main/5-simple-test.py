#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add parent directory to path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
update_variables_momentum = __import__('5-momentum').update_variables_momentum

if __name__ == '__main__':
    # Create a simple test case
    # Initialize a variable and its gradient for testing
    var = np.array([1, 2, 3, 4, 5])
    grad = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    velocity = np.zeros((5,))
    
    print("Initial variable:", var)
    print("Initial gradient:", grad)
    print("Initial velocity:", velocity)
    
    # Apply momentum update
    alpha = 0.01  # learning rate
    beta1 = 0.9   # momentum parameter
    
    # First update
    updated_var, updated_velocity = update_variables_momentum(alpha, beta1, var, grad, velocity)
    print("\nAfter 1st update:")
    print("Updated variable:", updated_var)
    print("Updated velocity:", updated_velocity)
    
    # Second update with new gradient
    grad2 = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
    updated_var2, updated_velocity2 = update_variables_momentum(alpha, beta1, updated_var, grad2, updated_velocity)
    print("\nAfter 2nd update:")
    print("Updated variable:", updated_var2)
    print("Updated velocity:", updated_velocity2)
    
    # Verify against manual calculation for first update
    expected_velocity = beta1 * velocity + (1 - beta1) * grad
    expected_var = var - alpha * expected_velocity
    print("\nVerification (1st update):")
    print("Expected velocity:", expected_velocity)
    print("Expected variable:", expected_var)
    print("Velocity difference:", np.sum(np.abs(updated_velocity - expected_velocity)))
    print("Variable difference:", np.sum(np.abs(updated_var - expected_var)))