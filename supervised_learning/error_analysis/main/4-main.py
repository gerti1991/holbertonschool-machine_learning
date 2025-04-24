#!/usr/bin/env python3

import numpy as np
import sys
import os
import importlib.util

# Add the parent directory to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import the module properly
spec = importlib.util.spec_from_file_location("f1_score", 
    os.path.join(parent_dir, "4-f1_score.py"))
f1_score_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f1_score_module)
f1_score = f1_score_module.f1_score

if __name__ == '__main__':
    # Load the confusion matrix
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(f1_score(confusion))