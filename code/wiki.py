"""
Module for crawling the Wikipedia Talk Page Conversations Corpus:
www.cs.cornell.edu/~cristian/Echoes_of_power_files/wikipedia_conversations_corpus_v1.01.zip
To get started, create a corpus object by supplying Corpus with a path to the unzipped corpus.
"""

import re
from datetime import datetime
import corpus
import nltk
import pickle
from tqdm import tqdm


DELIM = " +++$+++ "
CONVO_FILE  = "wikipedia.talkpages.conversations.txt"
USERS_FILE  = "wikipedia.talkpages.userinfo.txt"
ADMINS_FILE = "wikipedia.talkpages.admins.txt"
PICKLE_FILE = "wiki_corpus.pickle"
CORPUS_DIR  = "../data/wiki/"


class User(corpus.Speaker):

    def __init__(self, id, utts, edit_count, gender, numerical_id,
            admin, admin_ascention):
        self.utts = utts
        self.edit_count = edit_count
        self.gender = gender
        self.numerical_id = numerical_id
        self.admin = admin
        self.admin_ascention = admin_ascention
        super(User, self).__init__(id, utts)

    @classmethod
    def from_userinfo_file(cls, line):
        """
        Used by WikiCorpus.from_corpus_files. Note that utts, admin, and 
        admin_ascention are left to be initialized by the corpus.
        """
        id, edit_count, gender, numerical_id = line.split(DELIM)
        return cls(id, [], edit_count, gender, numerical_id, False, None)



class Post(corpus.Utterance):

    def __init__(self, id, user_id, reply_to_id, reply_to, timestamp,
            text, talkpage_user_id, conversation_root):
        self.talkpage_user_id = talkpage_user_id
        self.conversation_root = conversation_root
        self.reply_to_id = reply_to_id
        tokens = nltk.word_tokenize(text)
        super(Post, self).__init__(id, reply_to, user_id, timestamp,
                text, tokens)

    @classmethod
    def from_conversations_file(cls, line):
        """
        Used by WikiCorpus.from_corpus_files. Note that reply_to is left
        to be initialized by the corpus
        """
        id, user_id, talkpage, root, reply_to_id, timestamp, _, text, _  = line.split(DELIM)
        try:
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            timestamp = None
        return cls(id, user_id, reply_to_id, None, timestamp, text, talkpage, root)



class WikiCorpus(corpus.Corpus):
    """
    utts: a dictionary of utterancess by utt id
    users: a dictionary of users by numerical id 
    """


    def __init__(self, users, posts, networks):
        super(WikiCorpus, self).__init__(users, posts, networks)


    @classmethod
    def from_corpus_files(cls, path=CORPUS_DIR):

        from util import get_lines

        print("Creating Users...")
        users = {}
        lines = get_lines(path + USERS_FILE)
        for line in tqdm(lines):
            new_user = User.from_userinfo_file(line)
            users[new_user.id] = new_user
        lines = get_lines(path + ADMINS_FILE)
        for line in lines:
            s = line.split(" ")
            date = s[-1]
            user = "".join(w + " " for w in s[:-1])[:-1]
            if user in users:
                users[user].admin = True
                if date == "NA":
                    users[user].admin_ascention = None
                else:
                    users[user].admin_ascention = datetime.strptime(date, "%Y-%m-%d")
        # TODO consider pulling users present in conversations file but not in userinfo
        
        print("Loading conversations...")
        posts = {}
        lines = get_lines(path + CONVO_FILE)
        for line in tqdm(lines):
            if line.startswith("could not match") or line == "":
                continue
            new_post = Post.from_conversations_file(line)
            user = new_post.speaker_id
            if new_post.timestamp: # only include utterances with a timestamp
                 posts[new_post.id] = new_post 
                 if user in users:
                    users[user].utts.append(new_post)
        # loop back over posts to connect replies        
        for post in posts.values():
            if post.reply_to_id in posts: # else left as None
                post.reply_to = posts[post.reply_to_id]

        return cls(users, posts, {})


    def save_pickle(self):
        with open(CORPUS_DIR + PICKLE_FILE, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load_pickle(cls):
        with open(CORPUS_DIR + PICKLE_FILE, 'rb') as f:
            corpus = pickle.load(f)
            print('now were here')
            return cls(corpus.users, corpus.utts, corpus.networks)



if __name__ == '__main__':
    corpus = WikiCorpus.from_corpus_files()
    corpus.save_pickle()
