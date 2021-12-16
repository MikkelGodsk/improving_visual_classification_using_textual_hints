import numpy as np
import os
import re

from embedding_similarities import cos_sim
from wordnet_interface import find_synset, splitting_chars

"""
    Modified from: https://keras.io/examples/nlp/pretrained_word_embeddings/
    
    Probably should have been implemented as classes
"""

glove_embeddings = {}

def get_glove(version='6B'):
    """
        Fetches glove and writes to the global(!) variable embedding.
        If we do not need glove, then there's no reason for loading it.
    """
    global glove_embeddings
    path_to_glove_file = os.path.join(
        '/work3/s184399/glove', 'glove.'+version+'.txt'
    )
    with open(path_to_glove_file) as f:
        for line in f:
            word, vector = line.split(maxsplit=1)
            vector = np.fromstring(vector, "f", sep=" ")
            glove_embeddings[word] = vector

def is_initialized():
    return glove_embeddings != {}

def glove_cosine_avg_sim(x, y):
    """
        Computes the similarity between multi-word terms. Does so by first averaging the
        word embeddings across the seperate sentences, and then computing the cosine similarity,
        as was done in https://dl.acm.org/doi/abs/10.1145/2806416.2806475
    """
    x = find_synset(x).name().split('.')[0]
    y = find_synset(y).name().split('.')[0]
    xs, ys = re.split(splitting_chars, x), re.split(splitting_chars, y)
    x_mean, y_mean = np.zeros(glove_embeddings['the'].shape), np.zeros(glove_embeddings['the'].shape)
    for i in range(len(xs)):
        try:
            x_mean += glove_embeddings[xs[i]]
        except:
            print(xs[i] + " does not exist in GloVe")
    for i in range(len(ys)):
        try:
            y_mean += glove_embeddings[ys[i]]
        except:
            print(ys[i] + " does not exist in GloVe")
    #x_mean /= len(xs)  # Not necessary for cosine similarity as it normalizes the vector.
    #y_mean /= len(ys)

    return -cos_sim(x_mean.reshape((-1, 1)).astype(np.float32),
                    y_mean.reshape((-1, 1)).astype(np.float32))


glove_similarities = {
    'cosine_avg': glove_cosine_avg_sim
}


if __name__ == '__main__':
    get_glove()
    print("Found {:d} word vectors".format(len(glove_embeddings)))
    print("Example: Embedding for cat is " + str(glove_embeddings['cat']))
    print("Similarity for cat <-> cat: {}".format(glove_cosine_avg_sim("cat", "cat")))
    print("Similarity for 'little cat' <-> 'big cat': {}".format(glove_cosine_avg_sim("little cat", "big cat")))
    print("Similarity for cat <-> dog: {}".format(glove_cosine_avg_sim('cat', 'dog')))
    print("Similarity for french fries <-> fast food: {}".format(glove_cosine_avg_sim('french fries', 'fast food')))