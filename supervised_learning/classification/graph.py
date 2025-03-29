import numpy as np
import matplotlib.pyplot as plt

# Define the input range
x = np.linspace(-5, 5, 1000)

# Define the activation functions
def binary(x, threshold=0):
    return np.where(x >= threshold, 1, 0)

def linear(x, k=1):
    return k * x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Softmax is typically for a vector; here we simulate a simplified version for a single input
# For visualization, we'll use a normalized exponential to mimic its behavior
def softmax_approx(x):
    # Simulate a 2-class softmax: [x, 0] -> [e^x/(e^x+1), 1/(e^x+1)]
    return np.exp(x) / (np.exp(x) + 1)

# Create the plot
plt.figure(figsize=(10, 6), dpi=100)
plt.title("Comparison of Common Activation Functions", fontsize=14, pad=15)
plt.xlabel("Input (x)", fontsize=12)
plt.ylabel("Output", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Plot each activation function
plt.plot(x, binary(x), label="Binary (Step)", color="red", linewidth=2)
plt.plot(x, linear(x), label="Linear (k=1)", color="green", linewidth=2)
plt.plot(x, sigmoid(x), label="Sigmoid", color="blue", linewidth=2)
plt.plot(x, tanh(x), label="Tanh", color="purple", linewidth=2)
plt.plot(x, relu(x), label="ReLU", color="orange", linewidth=2)
plt.plot(x, softmax_approx(x), label="Softmax (Simplified)", color="cyan", linewidth=2)

# Add legend
plt.legend(loc="upper left", fontsize=10)

# Set axis limits
plt.ylim(-1.5, 2.5)
plt.xlim(-5, 5)

# Add horizontal and vertical lines at y=0 and x=0
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

# Show the plot
plt.tight_layout()
plt.show()