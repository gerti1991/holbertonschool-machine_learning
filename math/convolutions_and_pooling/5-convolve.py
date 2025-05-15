#!/usr/bin/env python3
"""Multi-kernel convolution with advanced padding and stride management"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution with multiple kernels

    Args:
        images : ndarray (m, h, w, c)
        kernels : ndarray (kh, kw, c, nc)
        padding : 'same'/'valid'/tuple
        stride : (sh, sw)

    Returns:
        ndarray (m, h_out, w_out, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    # Check channels
    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    # Padding calculation with stride management
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Apply padding
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    # Compute output dimensions
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out, nc))

    # Optimized spatial traversal
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Vectorized window extraction
            window = padded[:, h_start:h_end, w_start:w_end, :]

            # Simultaneous calculation for all kernels
            output[:, i, j, :] = np.tensordot(
                window, kernels, axes=([1, 2, 3], [0, 1, 2]))

    return output
