#!/usr/bin/env python3
"""Valid convolution module for grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on a batch of images

    Args:
        images : ndarray shape (m, h, w)
        kernel : ndarray shape (kh, kw)

    Returns:
        ndarray shape (m, h_out, w_out) of convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute output dimensions
    h_out = h - kh + 1
    w_out = w - kw + 1

    # Initialize result
    output = np.zeros((m, h_out, w_out))

    # Double loop over kernel dimensions
    for i in range(kh):
        for j in range(kw):
            # Extract the corresponding window
            window = images[:, i:i + h_out, j:j + w_out]
            # Apply the kernel coefficient
            output += window * kernel[i, j]

    return output
