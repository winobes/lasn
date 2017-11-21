import networkx as nx
from util import get_lines, inside_daterange, div_safe
from tqdm import tqdm
import csv
from collections import defaultdict, Counter
import warnings
import numpy as np


FWORDS_DIR = '../data/function words/'
markers = {m: get_lines(FWORDS_DIR + m + '.txt') for m in 
            ['conjunctions', 'articles', 'prepositions', 'adverbs', 'quantifiers', 
            'impersonal pronouns', 'personal pronouns', 'auxiliary verbs']}


class Post(object):

    def __init__(self, id, parent_id, author_id, timestamp, 
            clean_text, tokens=None, data=None):
        self.id = id
        self.parent_id = parent_id
        self.author_id = author_id 
        self.timestamp = timestamp
        self.clean_text = clean_text
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

    def __init__(self, users, posts, networks, user_data_fields=None, post_data_fields=None):

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
        self.user_data_fields = user_data_fields if user_data_fields else []
        self.post_data_fields = post_data_fields if post_data_fields else []

    @staticmethod
    def tokenize(post):
        from nltk.tokenize import word_tokenize, sent_tokenize
        return [word_tokenize(sent) for sent in sent_tokenize(post.clean_text)]

    def register_user_data(self, field_name, data_dict):
        self.user_data_fields.append(field_name)
        for user in self.users:
            data = data_dict[user] if user in data_dict else None
            self.users[user].data[field_name] = data

    def register_post_data(self, field_name, data_dict):
        self.post_data_fields.append(field_name)
        for post in self.posts:
            data = data_dict[post] if post in data_dict else None
            self.posts[post].data[field_name] = data

    def export_user_data(self, filename, blacklist=None):
        if not blacklist:
            blacklist = []
        data_fields = [f for f in self.user_data_fields if f not in blacklist]
        header = ['user_id'] + data_fields
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for user in self.users.values():
                data = [user.data[field] for field in data_fields]
                writer.writerow([user.id] +  data)

    def export_post_data(self, filename, blacklist=None):
        if not blacklist:
            blacklist = []
        data_fields = [f for f in self.post_data_fields if f not in blacklist]
        header = ['post_id'] + data_fields
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for post in self.posts.values():
                data = [post.data[field] for field in data_fields]
                writer.writerow([post.id] +  data)

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

    def get_coordination(self, reference_group=None, target_group=None):
        """
        Calculates the coordination users give to/receive from users in
        the reference group.
        """

        A = list(reference_group) if reference_group else list(self.users.keys())
        B = list(target_group) if target_group else list(self.users.keys())

        B_to_index = {b: i for i, b in enumerate(B)}
        marker_list = list(markers.keys())

        # coordination given counts - counting over pairs 
        # (u_a, u_b) where a in A said u_a and b replied with u_b
        Am_bm = np.zeros((len(markers), len(B))) # defaultdict(Counter) # count u_a and u_b exhibit m 
        Am_b  = np.zeros((len(markers), len(B))) # defaultdict(Counter) # count u_a exhibits m
        A_bm  = np.zeros((len(markers), len(B))) # defaultdict(Counter) # count u_b exhibits m
        A_b   = np.zeros((len(B),)) # count total number of reply pairs (u_a, u_b)

        # coordination received counts - counting over pairs
        # (u_b, u_a) where b said u_b and a in A replied with u_a
        bm_Am = np.zeros((len(markers), len(B))) # defaultdict(Counter) # count u_b and u_a exhibit m 
        bm_A  = np.zeros((len(markers), len(B))) # defaultdict(Counter) # count u_b exhibits m
        b_Am  = np.zeros((len(markers), len(B))) # defaultdict(Counter) # count u_a exhibits m
        b_A   = np.zeros((len(B),)) # Counter() # count total number of reply pairs (u_b, u_a)

        # count marker occurances & instances of alignment among reply pairs
        print("Gathering marker counts...")
        for reply in tqdm(self.posts.values()):
            reply_user = reply.get_author()
            parent = reply.get_parent()
            parent_user = parent.get_author() if parent else None
            if reply_user and parent_user:
                reply_user = reply_user.id
                parent_user = parent_user.id
            else:
                continue
            # increment coordination given counts
            if parent_user in A and reply_user in B: 
                b_i = B_to_index[reply_user]
                A_b[b_i] += 1
                for m_i, m in enumerate(marker_list):
                    if parent.exhibits_marker(m):
                        Am_b[m_i,b_i] += 1
                        if reply.exhibits_marker(m):
                            Am_bm[m_i,b_i] += 1
                    if reply.exhibits_marker(m):
                        A_bm[m_i,b_i] += 1

            # increment coordination received counts
            if reply_user in A and parent_user in B: 
                b_i = B_to_index[parent_user]
                b_A[b_i] += 1
                for m_i, m in enumerate(marker_list):
                    if parent.exhibits_marker(m):
                        bm_A[m_i,b_i] += 1
                        if reply.exhibits_marker(m):
                            bm_Am[m_i,b_i] += 1
                    if reply.exhibits_marker(m):
                        b_Am[m_i,b_i] += 1

        # calculate the per-marker coordinations
        coord_given = (Am_bm / div_safe(Am_b)) - (A_bm / div_safe(A_b))
        coord_received = (bm_Am / div_safe(bm_A)) - (b_Am / div_safe(b_A))

        # calculate the aggrigate (ignoring undefined markers its undefined for)
        coord_given_agg3 = np.nanmean(coord_given, axis=0)
        coord_received_agg3 = np.nanmean(coord_received, axis=0)

        # convert back to dictionary output
        coord_given = {m: {b: coord_given[m_i,b_i] 
            for b_i, b in enumerate(B)} for m_i, m in enumerate(marker_list)}
        coord_received = {m: {b: coord_received[m_i,b_i] 
            for b_i, b in enumerate(B)} for m_i, m in enumerate(marker_list)}

        for b in tqdm(B):
            coord_given['agg3'] = {b: coord_given_agg3[b_i] for b_i, b in enumerate(B)} 
            coord_received['agg3'] = {b: coord_received_agg3[b_i] for b_i, b in enumerate(B)} 
    
        return coord_given, coord_received

