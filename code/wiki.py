"""
Module for crawling the Wikipedia Talk Page Conversations Corpus:
www.cs.cornell.edu/~cristian/Echoes_of_power_files/wikipedia_conversations_corpus_v1.01.zip
To get started, create a corpus object by supplying Corpus with a path to the unzipped corpus.
"""

from corpus import Corpus, User, Post
from datetime import datetime
import nltk
import pickle
from tqdm import tqdm
from util import get_attribute_dict


DELIM = " +++$+++ "
CONVO_FILE  = "wikipedia.talkpages.conversations.txt"
USERS_FILE  = "wikipedia.talkpages.userinfo.txt"
ADMINS_FILE = "wikipedia.talkpages.admins.txt"
PICKLE_FILE = "wiki_corpus.pickle"
CORPUS_DIR  = ("../data/wiki/")

class WikiCorpus(Corpus):
    
    def __init__(self, users, posts, networks):
        """
        use `sample` to limit the number of posts for testing purposes
        """

        user_data_fields = ['edit_count', 'gender', 'admin_ascention', 'admin']
        post_data_fields = ['conversation_root', 'talkpage_user']

        super(WikiCorpus, self).__init__(users, posts, networks,
                user_data_fields, post_data_fields)


    @classmethod
    def from_corpus_files(cls, sample=None, path=CORPUS_DIR):

        from util import get_lines

        print("Opening corpus files...")

        def user_line(line):
            user_id, edit_count, gender, _ = line.split(DELIM)
            data = {'edit_count': edit_count, 'gender': gender, 
                    'admin_ascention': None, 'admin': False}
            return User(user_id, data)

        def post_line(line):
            # ignore UNIX timestamp 
            post_id, author_id, talkpage_user, conversation_root, parent_id, \
                    timestamp, _, clean_text, _ = line.split(DELIM)
            try:
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                timestamp = None
            data = {'conversation_root': conversation_root, 'talkpage_user': talkpage_user}
            return Post(post_id, parent_id, author_id, timestamp, 
                    clean_text, tokens=None, data=data)

        
        print("Loading users...")
        users = {}
        lines = get_lines(path + USERS_FILE)
        for line in tqdm(lines):
            new_user = user_line(line)
            users[new_user.id] = new_user
        lines = get_lines(path + ADMINS_FILE)
        for line in lines:
            s = line.split(" ")
            date = s[-1]
            user = "".join(w + " " for w in s[:-1])[:-1]
            if user in users:
                users[user].data['admin'] = True
                if date == "NA":
                    users[user].data['admin_ascention'] = None
                else:
                    users[user].data['admin_ascention'] = datetime.strptime(date, "%Y-%m-%d")
        # TODO consider pulling users present in conversations file but not in userinfo

        print("Loading posts...")
        posts = {}
        lines = get_lines(path + CONVO_FILE)
        for line in tqdm(lines[:sample]):
            if line.startswith("could not match") or line == "":
                continue
            new_post = post_line(line)
            if new_post.timestamp: # only include utterances with a timestamp
                 posts[new_post.id] = new_post 

        return cls(users, posts, {})


    def to_pickle(self, filename, path=CORPUS_DIR):
    
        users = [get_attribute_dict(user) for user in self.users.values()]
        posts = [get_attribute_dict(post) for post in self.posts.values()]
        networks = self.networks
        user_data_fields = self.user_data_fields
        post_data_fields = self.post_data_fields
        
        with open(path + filename, 'wb') as f:
            pickle.dump((users, posts, networks, user_data_fields, post_data_fields), f)


    @classmethod
    def from_pickle(cls, filename, path=CORPUS_DIR):

        print("Opening pickle...")
        with open(path + filename, 'rb') as f:
            users, posts, networks, user_data_fields, post_data_fields = pickle.load(f)

        print("Loading users...")
        user_args = ['id', 'data']
        users = {user['id']: User(*(user[arg] for arg in user_args)) for user in tqdm(users)}

        print("Loading posts...")
        post_args = ['id', 'parent_id', 'author_id', 'timestamp', 'clean_text', 'tokens', 'data']
        posts = {post['id']: Post(*(post[arg] for arg in post_args)) for post in tqdm(posts)}

        return cls(users, posts, networks, user_data_fields, post_data_fields)


