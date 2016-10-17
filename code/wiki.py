"""
Module for crawling the Wikipedia Talk Page Conversations Corpus:
www.cs.cornell.edu/~cristian/Echoes_of_power_files/wikipedia_conversations_corpus_v1.01.zip
To get started, create a corpus object by supplying Corpus with a path to the unzipped corpus.
"""

import re
from datetime import datetime


def get_lines(path):
    """
    Returns a list of strings with no newline characters.
    Assumes that each line (including the last one) ends in a newline.
    """
    with open(path, 'r') as f:
        return [line[:-1] for line in f.readlines()]

FWORDS_DIR = '../data/function words/'
markers = ['conjunctions', 'articles', 'prepositions', 'adverbs', 'quantifiers', 
           'impersonal pronouns', 'personal pronouns', 'auxiliary verbs']
markers = {m: get_lines(FWORDS_DIR + m + '.txt') for m in markers}

earliest_date = datetime(2006,1,1)
latest_date = datetime(2011,12,31)

class User:

    def __init__(self, corpus, s):
        self.corpus = corpus
        c = s.split(corpus.DELIM)
        self.user = c[0]
        self.edit_count = int(c[1]) 
        self.gender = c[2] 
        self.numerical_id = c[3]
        self.admin = False # initialized by corpus
        self.admin_ascention = None # initialized by corpus
        self.utts = [] # initialized by corpus

class Utt:

    def __init__(self, corpus, s):
        self.corpus = corpus
        c = s.split(corpus.DELIM)
        self.utt_id = c[0]
        self.user_id = c[1]
        self.talkpage_user_id = c[2]
        self.conversation_root = c[3]
        self.reply_to = self.corpus.utts[c[4]] if c[4] in self.corpus.utts else None
        try:
            self.timestamp = datetime.strptime(c[5], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            self.timestamp = None
        # skip the unix time c[6]
        self.clean_text = c[7]
        # skip the raw text c[8]

        # TODO: see if real tokenization makes any difference
        self.word_list = re.sub("[^\w]", " ", self.clean_text).split()

    def exhibits(self, m):
        return any(f_word in self.word_list for f_word in markers[m])

class Corpus:
    """
    utts: a dictionary of utterancess by utt id
    users: a dictionary of users by numerical id 
    """

    DELIM = " +++$+++ "
    CONVO_FILE  = "wikipedia.talkpages.conversations.txt"
    USERS_FILE  = "wikipedia.talkpages.userinfo.txt"
    ADMINS_FILE = "wikipedia.talkpages.admins.txt"
    FWORDS_DIR  = "function words/"

    def __init__(self, path, start_date=earliest_date, end_date=latest_date):

        print("Creating Users...")
        self.users = {}
        lines = get_lines(path + self.USERS_FILE)
        for line in lines:
            new_user = User(self, line)
            self.users[new_user.user] = new_user
        lines = get_lines(path + self.ADMINS_FILE)
        for line in lines:
            s = line.split(" ")
            date = s[-1]
            user = "".join(w + " " for w in s[:-1])[:-1]
            if user in self.users:
                self.users[user].admin = True
                if date == "NA":
                    self.users[user].admin_ascention = None
                else:
                    self.users[user].admin_ascention = datetime.strptime(date, "%Y-%m-%d")
        
        print("Loading conversations...")
        self.utts = {}
        lines = get_lines(path + self.CONVO_FILE)
        for line in lines:
            if line.startswith("could not match"):
                continue
            if line == "": # indicates a new conversation
                continue
            new_utt = Utt(self, line)
            user = new_utt.user_id
            if new_utt.timestamp and start_date <= new_utt.timestamp and new_utt.timestamp <= end_date:
                 self.utts[new_utt.utt_id] = new_utt
                 if user in self.users:
                    self.users[user].utts.append(new_utt)


    def generate_network(self, prune=True, normalize_edge_weights=True):
        """
        Crates a NetworkX network whose edges are based on reply pairs within 
        the range of dates. A start/end date of None means on the left/right.
        Edge weights are based on the number of reply pairs between users. 

        Conversation heads (utterances that are not in reply to anything) give
        weight between the user and the user whose talk page the conversation
        is on.

        Edges are symmetric (undirected network) and anti-reflexive (ignore self-
        loops).

        If `normalize_edge_weights` is True, edge weights are normalized with 
        respect to the largest edge. 

        If `prune` is True, the network is pruned to its largest connected 
        component.
        """

        import networkx as nx

        net = nx.Graph()
        utts = self.utts.values()

        print("Generating network from", len(utts), "utterances...")
        reply_to_unknown_user = 0
        for utt in utts: 
            user_A = utt.user_id
            if utt.reply_to:
                user_B = utt.reply_to.user_id
            else: # If not a reply, it's the conversation head
                user_B = utt.talkpage_user_id
            if user_A in self.users and user_B in self.users and not user_B == user_A:
                if net.has_edge(user_A, user_B):
                    net[user_A][user_B]['weight'] += 1
                else:
                    net.add_edge(user_A, user_B, weight = 1)
            else:
                reply_to_unknown_user += 1
        print("There were", reply_to_unknown_user, "replies to unknown users.")
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

        return net


    def reply_pairs(self, a, b):
        """ 
        Gives utterance pairs where user b is replying to user a.
        """
        reply_pairs = []
        for u2 in self.users[b].utts:
            u1 = u2.reply_to
            if not u1:
                continue
            if u1.user_id == a:
                reply_pairs.append((u1,u2))
        return reply_pairs


    def coordination(self, a, b):
        """
        Calculates the coordination of user b towards user a along each marker as 
        in Danescu-Niculescu-Mizil et. al. 2012, equation 2.

        We remove wholesale repetitions of phrases up to length `ignore_ngram_repeats` 
        to help ensure that only true style matching is being captured. See Danescu-
        Niculescu-Mizil et. al. 2012, footnote 11.
    	"""
        reply_pairs = self.reply_pairs(a, b) 
        coord = {m:None for m in markers}
        coord['agg3'] = None
        if not reply_pairs:
            return coord
        for m in markers:
            m_a = 0; m_b = 0; m_ab = 0
            total = len(reply_pairs)
            for u1,u2 in reply_pairs:
                if u1.exhibits(m):
                    m_a += 1
                    if u2.exhibits(m):
                        m_b  += 1
                        m_ab += 1
                elif u2.exhibits(m):
                    m_b += 1
            if m_a > 0 and m_b > 0 and m_a != total and m_b != total:
                coord[m] = m_ab/m_a - m_b/total
        defined_markers = [m for m in markers if coord[m] is not None]
        if defined_markers:
            coord['agg3'] = sum(coord[m] for m in defined_markers) / len(defined_markers)
        return coord

    def coordination_given(self, a): 
        coords = {b:self.coordination(b, a) for b in self.users}
        agg_defined = [b for b in coords if coords[b]['agg3'] is not None]
        if not agg_defined:
            return None 
        return sum(coords[b]['agg3'] for b in agg_defined) / len(agg_defined)
        
    def coordination_received(self, a):
        coords = {b:self.coordination(a, b) for b in self.users} 
        agg_defined = [b for b in coords if coords[b]['agg3'] is not None]
        if not agg_defined:
            return None 
        return sum(coords[b]['agg3'] for b in agg_defined) / len(agg_defined)
 
def split_list(data, measure, cutoff):
    sorted_data = sorted(data, key=lambda x: measure[x], reverse=True)
    return (sorted_data[0:cutoff], sorted_data[cutoff:])


