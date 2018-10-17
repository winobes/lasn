""" Functions for common low-level corpus manipulation """

from tqdm import tqdm
import pandas as pd

def tokenize_corpus(corpus):
    """ Add a 'tokens' column to the corpus dataframe """
    tokens = [nltk.tokenize.word_tokenize(text) for text in tqdm(corpus['clean_text'])]
    return posts.assign(tokens=tokens)


def detect_markers(corpus, markers):
    """ Add feature columns for marker counts.

    Markers is i dictionary of lists where the keys are the marker types and
    the values are a list of markers.
    """

    if not 'tokens' in corpus:
        raise ValueError("Corpus must be tokenized for marker detection.")

    feature_columns = {m: [False] * len(posts) for m in markers} # TODO: change to counts for WHAM
    for i, tokens in enumerate(tqdm(posts['tokens'])):
        for m in markers:
            if any(t.lower() in markers[m] for t in tokens):
                feature_columns[m][i] = True
    posts = posts.assign(**feature_columns)


def get_reply_pairs(posts, filter_self_replies=True):
    """ View the posts dataframe as reply pairs """
    pairs = pd.merge(posts, posts, how='inner', left_index=True, right_on='reply_to', suffixes=['_a', '_b'])
    if filter_self_replies:
        pairs = pairs[(pairs.user_a != pairs.user_b)]
    return pairs
