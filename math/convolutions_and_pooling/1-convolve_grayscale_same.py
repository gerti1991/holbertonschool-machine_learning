#!/usr/bin/env python3
"""Same convolution module for grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    Args:
        images : ndarray shape (m, h, w) containing the images
        kernel : ndarray shape (kh, kw) containing the kernel

    Returns:
        ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Handle padding for even and odd kernels
    ph = (kh - int(kh % 2 == 1)) // 2
    pw = (kw - int(kw % 2 == 1)) // 2

    # Compute output dimensions
    h_out = h
    w_out = w

    # Apply padding
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant',
                    constant_values=0)

    # Initialize output
    output = np.zeros((m, h_out, w_out))

    # Convolution
    for i in range(h_out):
        for j in range(w_out):
            # Extract the correct window
            window = padded[:, i:i + kh, j:j + kw]
            # Multiply and sum over the correct axes
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
