#!/usr/bin/env python3

import numpy as np
import sys
import os
import importlib.util

# Add the parent directory to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Get the path to the data directory (relative to project root)
project_root = os.path.abspath(os.path.join(parent_dir, '../..'))
data_dir = os.path.join(project_root, 'data')

# Import the module properly when it starts with a number
spec = importlib.util.spec_from_file_location("create_confusion", 
    os.path.join(parent_dir, "0-create_confusion.py"))
create_confusion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(create_confusion_module)
create_confusion_matrix = create_confusion_module.create_confusion_matrix

if __name__ == '__main__':
    # Load the data from the data folder
    lib = np.load(os.path.join(data_dir, 'labels_logits.npz'))
    labels = lib['labels']
    logits = lib['logits']

    np.set_printoptions(suppress=True)
    confusion = create_confusion_matrix(labels, logits)
    print(confusion)
    np.savez_compressed('confusion.npz', confusion=confusion)