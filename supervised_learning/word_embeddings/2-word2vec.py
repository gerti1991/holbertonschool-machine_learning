#!/usr/bin/env python3
"""word embedding"""
import gensim
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """Creates and trains a Word2Vec model"""
    return Word2Vec(
        sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        epochs=epochs,
        workers=workers,
        seed=seed,
        sg=not cbow
    )
