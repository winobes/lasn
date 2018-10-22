""" Functions for common low-level corpus manipulation """

import warnings
from tqdm import tqdm
import pandas as pd
import nltk

def tokenize_posts(posts):
    """ Add a 'tokens' column to the posts dataframe """
    tokens = [nltk.tokenize.word_tokenize(text) for text in tqdm(posts['clean_text'], desc="Tokenizing posts.")]
    return posts.assign(tokens=tokens)


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

    feature_columns = {m: [False] * len(posts) for m in markers} # TODO: change to counts for WHAM
    for i, tokens in enumerate(tqdm(posts['tokens']), desc="Detecting markers."):
        for m in markers:
            if any(t.lower() in markers[m] for t in tokens):
                feature_columns[m][i] = True
     
    return posts.assign(**feature_columns)


def get_reply_pairs(posts, filter_self_replies=True):
    """ View the posts dataframe as reply pairs """
    pairs = pd.merge(posts, posts, how='inner', left_index=True, right_on='reply_to', suffixes=['_a', '_b'])
    if filter_self_replies:
        pairs = pairs[(pairs.user_a != pairs.user_b)]
    return pairs
