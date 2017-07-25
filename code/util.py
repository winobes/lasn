from __future__ import division
import numpy as np
import math
from collections import Counter

log_arr = np.vectorize(lambda x, base: 0 if x == 0 else math.log(x, base))

def entropy(P, base=2):
    """ Computes Shannon entropy of a single probability distribution. """
    return - (P * log_arr(P, base)).sum()

def jsd(Ps, weights=None, base=None):
    """ Jensen-Shannon divergence """
    n_dists = len(Ps)
    if weights == None:
        weights = np.full(len(Ps), 1/n_dists)
    if base == None:
        base = n_dists # forces 0 <= JSD <= 1
    _entropy = lambda x: entropy(x, base=base)
    return (_entropy((weights[:,np.newaxis] * Ps).sum(axis=0)) - 
           (weights * np.apply_along_axis(_entropy, 1, Ps)).sum())

def ttr(text):
    """
    Type-token ratio. Expects tokenized text (list of tokens).
    """
    if not text:
        return None
    counter = Counter(text)
    return len(counter) / sum(counter.values())

def mattr(text, window_size=500):
    """
    Implementation of the Moving Average Type-Token Ration Algorithm (Covington & McFall 2010)
    Assumes that text has been tokenized.
    """
    def decrement(counter, key):
        if counter[key] == 1:
            counter.pop(key)
        else:
            counter[key] -= 1
    text_length = len(text)
    if text_length < window_size:
        return None
    word_counts = Counter(text[:window_size])
    ttrs = [len(word_counts) / window_size]
    for i in range(text_length - window_size):
        decrement(word_counts, text[i])
        word_counts[text[i + window_size]] += 1
        ttrs.append(len(word_counts) / window_size) 
    return sum(ttrs) / len(ttrs)


def split_list(data, measure, cutoff):
    sorted_data = sorted(data, key=lambda x: measure[x], reverse=True)
    return (sorted_data[0:cutoff], sorted_data[cutoff:])


def get_lines(path):
    with open(path, 'r') as f:
        return [line[:-1] for line in f.readlines()]

