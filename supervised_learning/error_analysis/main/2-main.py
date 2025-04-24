#!/usr/bin/env python3

import numpy as np
import sys
import os
import importlib.util

# Add the parent directory to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import the module properly
spec = importlib.util.spec_from_file_location("precision", 
    os.path.join(parent_dir, "2-precision.py"))
precision_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision_module)
precision = precision_module.precision

if __name__ == '__main__':
    # Load the confusion matrix
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(precision(confusion))