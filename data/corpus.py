""" Functions for common low-level corpus manipulation """

import warnings
from tqdm import tqdm
import os

import pandas as pd
import numpy as np
import nltk

from collections import Counter, defaultdict

def tokenize_posts(posts, overwrite=False):
    """ Add a 'tokens' column to the posts dataframe """

    if not overwrite and 'tokens' in posts:
        warnings.warn("Posts are already tokenized. Skipping tokenization.")
        return posts
    
    tokens = [nltk.tokenize.word_tokenize(text) for text in tqdm(posts['clean_text'], desc="Tokenizing posts.")]
    posts = posts.assign(tokens=tokens)

    n = len(posts)
    posts = posts[posts.tokens.map(len) > 0]
    print("Filtered {} posts with 0-length utterances".format(n - len(posts)))

    return posts

def pos_tag_posts(posts, overwrite=False):
    """ Add a 'pos_tags' column to the posts dataframe """

    if not overwrite and 'pos_tags' in posts:
        warnings.warn("Posts are already PoS tagged. Skipping tagging.")
        return posts
    
    pos = posts.tokens.progress_apply(nltk.pos_tag)
    posts['pos_tags'] = pos.apply(lambda pos_list: list(map(lambda t: t[1], pos_list)))
    
    return posts

def sublists(l, n):
    """ generate all sublists of length n (where order matters) """
    return (l[x:x+n] for x in range(len(l) - n + 1))

def detect_markers(posts, markers, overwrite=False):
    """ Add feature columns for marker counts.

    Markers is i dictionary of lists where the keys are the marker types and
    the values are a list of markers.
    """

    if not 'tokens' in posts:
        raise ValueError("Corpus must be tokenized for marker detection.")


    if not overwrite and all(m in posts for m in markers):
        warnings.warn("All marker columns already exist. Skipping marker detection.")
        return posts

    for m in markers.keys():
        # sort the marker instances by length for efficency
        marker_insts = defaultdict(list)
        for marker_inst in markers[m]:  
            marker_insts[len(marker_inst)].append(marker_inst)
        counts = []
        for i, tokens in enumerate(tqdm(posts['tokens'], desc="Detecting {}.".format(m))):
            c = 0
            for m_len in marker_insts:
                for subseq in sublists(tokens, m_len):
                    if subseq in marker_insts[m_len]:
                        c += 1
            counts.append(c)
        posts[m] = counts

    return posts 


def get_reply_pairs(posts, filter_self_replies=True):
    """ View the posts dataframe as reply pairs """
    pairs = pd.merge(posts, posts, how='inner', left_index=True, right_on='reply_to', suffixes=['_a', '_b'])
    if filter_self_replies:
        pairs = pairs[(pairs.user_a != pairs.user_b)]
    return pairs


def load_network(reply_pairs=None, recreate=False, filename=None):
    try:
        if recreate:
            return create_network(reply_pairs) 
        else: 
            return igraph.Graph.Read_Pickle(filename)
    except TypeError:
        return create_network(reply_pairs) 

def save_network(network, overwrite=False, filename=None):
    if os.path.isfile(filename) and not overwrite:
        warnings.warn("{} already exists. Not overwriting.".format(filename))
    else:
        network.write_pickle(filename)
