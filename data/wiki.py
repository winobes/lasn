""" Functions for loading the wiki corpus into a pandas dataframe """

import os
import warnings
from tqdm import tqdm
from datetime import datetime

import nltk
import pandas as pd
import numpy as np

from .corpus import load_network as __load_network
from .corpus import save_network as __save_network

DELIM         = " +++$+++ "
CORPUS_DIR    = os.path.join(os.path.dirname(__file__), "wiki/")
CONVO_FILE    = os.path.join(CORPUS_DIR, "wikipedia.talkpages.conversations.txt")
USERS_FILE    = os.path.join(CORPUS_DIR, "wikipedia.talkpages.userinfo.txt")
ADMINS_FILE   = os.path.join(CORPUS_DIR, "wikipedia.talkpages.admins.txt")
POSTS_DF_FILE = os.path.join(CORPUS_DIR, "posts_df.pickle")
USERS_DF_FILE = os.path.join(CORPUS_DIR, "users_df.pickle")
NETWORK_FILE  = os.path.join(CORPUS_DIR, "users_network.pickle")


def load_posts(filename=POSTS_DF_FILE, recreate=False):
    if recreate:
        return create_posts()
    try:
        return pd.read_pickle(filename)
    except FileNotFoundError:
        return create_posts()


def create_posts():

    columns = ['utterance_id', 'user', 'talkpage_user', 'conversation_root', 'reply_to', 
               'timestamp', 'timestamp_unixtime', 'clean_text', 'raw_text']
    posts = {column: [] for column in columns}

    with open(CONVO_FILE) as f:
        for line in tqdm(f.readlines(), desc="Reading conversations file."):
            # parse lines from the conversations file
            if line.startswith("could not match") or line.strip() == "":  # skip blank lines
                continue
            line = line.rstrip('\n').split(DELIM)
            assert(len(line) == len(columns))
            line = {column: value for column, value in zip(columns, line)}
            # convert timestamps to datetime objects
            try:
                line['timestamp'] = datetime.strptime(line['timestamp'], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                line['timestamp'] = None
            for column, value in line.items():
                posts[column].append(value)
                
    posts = pd.DataFrame(data=posts, index=posts['utterance_id'], columns=columns)

    n = len(posts)
    posts = posts[~(posts.user == '')]
    print("Filtered {} posts from blank or missing users".format(n - len(posts)))

    return posts


def save_posts(posts, filename=POSTS_DF_FILE, overwrite=False):
    if os.path.isfile(filename) and not overwrite:
        warnings.warn("{} already exists. Not overwriting.".format(filename))
    else:
        pd.to_pickle(posts, POSTS_DF_FILE)


def create_users(posts=None):
    """ Creates users from users corpus file.
    If `posts` is supplied, 'post_count' is also computed.
    Users from `posts` who are missing from the users file are included.
    """

    columns = ['user', 'edit_count', 'gender', 'numerical_id']
    users = {column: [] for column in columns}
    with open(USERS_FILE) as f:
        for line in tqdm(f.readlines(), desc="Reading users file."):
            line = line.rstrip('\n').split(DELIM)
            assert(len(line) == len(columns))
            line = {column: value for column, value in zip(columns, line)}
            for column, value in line.items():
                users[column].append(value)
    users = pd.DataFrame(data=users, columns=columns)

    # add users from posts file (and post counts), if provided
    if posts is not None: 
        post_counts = posts.assign(post_count=1)[['user', 'post_count']].groupby('user').sum().reset_index()
        users = pd.merge(users, post_counts, on='user', how='outer')

    # add adminship information
    columns = ['user', 'admin_ascension']
    admins = {column: [] for column in columns}
    with open(ADMINS_FILE) as f:
        for line in tqdm(f.readlines(), desc="Reading admins file."):
            line = line.rstrip('\n').split(' ')
            line = ' '.join(line[:-1]), line[-1]
            assert(len(line) == len(columns))
            line = {column: value for column, value in zip(columns, line)}

            # convert timestamps to datetime objects
            try:
                line['admin_ascension'] = datetime.strptime(line['admin_ascension'], "%Y-%m-%d")
            except ValueError:
                line['admin_ascension'] = None
            for column, value in line.items():
                admins[column].append(value)
    admins = pd.DataFrame(admins)
    users = pd.merge(users, admins, on='user', how='left')
    users['admin'] = users['admin_ascension'].notna() # add binary admin category

    users = users.set_index('user')
    return users


def load_users(posts=None, filename=USERS_DF_FILE, recreate=False):
    if recreate:
        return create_users(posts=posts)
    try:
        return pd.read_pickle(filename)
    except FileNotFoundError:
        return create_users(posts=posts)


def save_users(users, filename=USERS_DF_FILE, overwrite=False):
    if os.path.isfile(filename) and not overwrite:
        warnings.warn("{} already exists. Not overwriting.".format(filename))
    else:
        pd.to_pickle(users, USERS_DF_FILE)


def load_network(reply_pairs=None, recreate=False, filename=NETWORK_FILE):
    return __load_network(reply_pairs, recreate, filename) 


def save_network(network, overwrite=False, filename=NETWORK_FILE):
    __save_network(network, overwrite, filename)
