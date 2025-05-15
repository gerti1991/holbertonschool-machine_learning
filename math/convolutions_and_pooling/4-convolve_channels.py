#!/usr/bin/env python3
"""Multichannel convolution with correct handling of padding and stride"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels

    Args:
        images : ndarray (m, h, w, c)
        kernel : ndarray (kh, kw, c)
        padding : 'same'/'valid'/tuple
        stride : tuple (sh, sw)

    Returns:
        ndarray (m, h_out, w_out)
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # Check channels
    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    # New padding calculation for 'same'
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

    # Output dimensions
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, h_out, w_out))

    # Loop over output positions
    for i in range(h_out):
        for j in range(w_out):
            # Positions with stride management
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Window (m, kh, kw, c)
            window = padded[:, h_start:h_end, w_start:w_end, :]

            # Product + sum over kh, kw, c
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))

    return output
