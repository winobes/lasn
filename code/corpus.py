import networkx as nx
from util import get_lines, inside_daterange
from tqdm import tqdm
import csv


FWORDS_DIR = '../data/function words/'
markers = {m: get_lines(FWORDS_DIR + m + '.txt') for m in 
            ['conjunctions', 'articles', 'prepositions', 'adverbs', 'quantifiers', 
            'impersonal pronouns', 'personal pronouns', 'auxiliary verbs']}


class Post(object):

    def __init__(self, id, parent_id, author_id, timestamp, 
            clean_text, raw_text, tokens=None, data=None):
        self.id = id
        self.parent_id = parent_id
        self.author_id = author_id 
        self.timestamp = timestamp
        self.clean_text = clean_text
        self.raw_text = raw_text
        self.data = data
        self.tokens = tokens
        self._author = None
        self._parent = None
        self._corpus = None


    def get_parent(self):
        return self._parent

    def get_author(self):
        return self._author

    def get_tokens(self, sentence_tokenize=False):
        """
        Returns the tokenized sentence. If `sentence_tokenize` is
        True then it returns a list of tokens (one for each sentence
        in the post).
        """
        if sentence_tokenize:
            return self.tokens
        else:
            return [word for sentence in self.tokens for word in sentence]


    def exhibits_marker(self, m):
        return any(f_word in self.get_tokens() for f_word in markers[m])


class User(object):
    
    def __init__(self, id, data=None):
        self.id = id
        self.data = data
        self._posts = []
        self._corpus = None


    def get_posts(self, date_range=None):
        if date_range:
            posts = [post for post in self._posts if inside_daterange(post.timestamp, date_range)]
        else:
            posts = self._posts
        return posts


    def replies_to(self, other, date_range=None):
        reply_pairs = []
        for reply in self.get_posts(date_range):
            parent = reply.get_parent()
            if parent and parent.author_id == other.id:
                reply_pairs.append((parent, reply))
        return reply_pairs


class Corpus(object):

    def __init__(self, users, posts, networks):

        print("Setting up corpus...")

        for user in users.values():
            user._corpus = self

        for post in tqdm(posts.values()):
            post._corpus = self
            if not post.tokens:
                post.tokens = self.tokenize(post)
            if post.author_id in users:
                users[post.author_id]._posts.append(post)
            if post.parent_id in posts:
                post._parent = posts[post.parent_id]
            if post.author_id in users:
                post._author = users[post.author_id]
        
        self.users = users
        self.posts = posts
        self.networks = networks


    @staticmethod
    def tokenize(post):
        from nltk.tokenize import word_tokenize, sent_tokenize
        return [word_tokenize(sent) for sent in sent_tokenize(post.clean_text)]


    def generate_network(self, name, criteria=lambda x: True,
            prune=True, normalize_edge_weights=True):
        """
        Crates a NetworkX network whose edges are based on reply pairs within 
        the range of dates. A start/end date of None means no time boxing.
        
        Edge weights are based on the number of reply pairs between users. 

        Edges are symmetric (undirected network) and anti-reflexive (ignore self-
        loops).

        If `normalize_edge_weights` is True, edge weights are normalized with 
        respect to the largest edge. 

        If `prune` is True, the network is pruned to its largest connected 
        component.
        """

        net = nx.Graph()

        posts = [post for post in self.posts.values() if criteria(post)]

        print("Generating network ...")
        unknown_users = 0
        for post in tqdm(posts): 
            user_A = post.author_id
            parent = post.get_parent() 
            if not parent: # If not a reply, it's the conversation head
                continue

            user_B = parent.author_id 
            if not user_A in self.users or not user_B in self.users:
                unknown_users += 1
                continue
            elif  user_B == user_A:
                continue  # ignore self-talk
            elif net.has_edge(user_A, user_B):
                    net[user_A][user_B]['weight'] += 1
            else:
                net.add_edge(user_A, user_B, weight=1)
        print("There were", unknown_users, "posts to/from unknown users.")
        print("The unpruned network has ", len(net.nodes()), "nodes.")

        if prune:
            print("Pruning network to its largest component...")
            minor_components = list(nx.connected_components(net))[1:]
            disconnected_users = [user for com in minor_components for user in com]
            net.remove_nodes_from(disconnected_users)
            print("\t removed", len(disconnected_users), "users from", 
                    len(minor_components), "disconnected components.")

        if normalize_edge_weights:
            print("Normalizing edge weights...")
            max_weight = max(net[user_A][user_B]['weight'] 
                    for user_A in net for user_B in net[user_A])
            for user_A in net:
                for user_B in net[user_A]:
                    net[user_A][user_B]['weight'] /= max_weight
        
        self.networks[name] = net
        return net


    def export_user_data(self, data_fields, filename):
        header = ['user_id'] + data_fields
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for user in self.users.values():
                data = [user.data[field] if field in user.data else None for field in data_fields]
                writer.writerow([user.id] +  data)


