""" Utility functions for experiment scripts. """

import os

MARKER_TYPES = ['conjunctions', 'articles', 'prepositions', 'adverbs', 'quantifiers', 
    'impersonal_pronouns', 'personal_pronouns', 'auxiliary_verbs']
MARKERS_DIR = os.path.join(os.path.dirname(__file__), "function words/")

def load_markers():
    markers = {}
    for m in MARKER_TYPES:
        with open(os.path.join(MARKERS_DIR, m + '.txt')) as f:
            markers[m] = [line.rstrip('\n').split(' ') for line in f.readlines()] # tokenize multi-word markers
    return markers
