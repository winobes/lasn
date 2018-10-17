""" Utility functions for experiment scripts. """

import os

MARKER_TYPES = ['conjunctions', 'articles', 'prepositions', 'adverbs', 'quantifiers', 
    'impersonal_pronouns', 'personal_pronouns', 'auxiliary_verbs']
MARKERS_DIR = os.path.join(os.path.dirname(__file__), "function words/")

def load_markers():
    markers = {m: [] for m in MARKER_TYPES}
    for m in markers:
        with open(os.path.join(MARKERS_DIR, m + '.txt')) as f:
            markers[m] = [word.rstrip('\n') for word in f.readlines()]
    return markers
