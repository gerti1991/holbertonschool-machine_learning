#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
learning_rate_decay = __import__('11-learning_rate_decay').learning_rate_decay

if __name__ == '__main__':
    # Initial learning rate
    alpha_init = 0.1
    
    # Number of steps to simulate
    steps = 200
    
    # Different decay rates to compare
    decay_rates = [1, 2, 0.5]
    
    # Decay step (number of iterations before applying decay)
    decay_step = 10
    
    # Create figure for the plot
    plt.figure(figsize=(12, 6))
    
    # Colors for different decay rates
    colors = ['b', 'r', 'g']
    
    # List to store alphas for each configuration
    all_alphas = []
    
    # Calculate learning rate decay for each decay rate
    for i, decay_rate in enumerate(decay_rates):
        alphas = []
        
        # Calculate alpha for each step
        for step in range(steps):
            alpha = learning_rate_decay(alpha_init, decay_rate, step, decay_step)
            alphas.append(alpha)
            
            # Print the first 100 values for the default decay_rate (1)
            if decay_rate == 1 and step < 100:
                print(f"Step {step}: alpha = {alpha}")
        
        all_alphas.append(alphas)
        
        # Plot the learning rate decay
        plt.plot(
            range(steps), 
            alphas, 
            color=colors[i], 
            label=f'decay_rate={decay_rate}'
        )
        
        # Add markers at decay steps
        decay_points_x = list(range(0, steps, decay_step))
        decay_points_y = [alphas[x] for x in decay_points_x]
        plt.scatter(
            decay_points_x, 
            decay_points_y, 
            color=colors[i], 
            marker='o'
        )
    
    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Learning Rate (alpha)', fontsize=12)
    plt.title('Inverse Time Decay Learning Rate Schedule', fontsize=14)
    plt.legend()
    
    # Show the decay steps with vertical lines
    for i in range(0, steps, decay_step):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_decay.png')
    plt.show()
    
    print("\nComparison of decay rates effect:")
    for i, decay_rate in enumerate(decay_rates):
        print(f"decay_rate={decay_rate}:")
        print(f"  After 50 steps: alpha = {all_alphas[i][49]:.6f}")
        print(f"  After 100 steps: alpha = {all_alphas[i][99]:.6f}")
        print(f"  After 200 steps: alpha = {all_alphas[i][199]:.6f}")