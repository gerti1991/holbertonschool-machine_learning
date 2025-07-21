#!/usr/bin/env python3
"""word embedding"""
import tensorflow as tf
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """Convert a gensim model to keras Embedding layer"""
    return model.wv.get_keras_embedding(train_embeddings=True)
