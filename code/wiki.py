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

FWORDS_DIR = 'function words/'
markers = ['conjunctions', 'articles', 'prepositions', 'adverbs', 'quantifiers', 
           'impersonal pronouns', 'personal pronouns', 'auxiliary verbs']
markers = {m: get_lines(FWORDS_DIR + m + '.txt') for m in markers}

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
        self.reply_to = None if c[4] in {"initial_post", "-1"} else self.corpus.utts[c[4]]
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

    def __init__(self, path):

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
            self.utts[new_utt.utt_id] = new_utt
            if user in self.users:
                self.users[user].utts.append(new_utt)


    def generate_network(self, start_date=None, end_date=None, 
            prune=True, normalize_edge_weights=True):
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
        net.add_nodes_from(self.users.keys())
        utts = self.utts.values()

        print("Generating network from reply pairs...")
        reply_to_unknown_user = 0
        for utt in self.utts.values():
            user_A = utt.user_id
            if utt.reply_to:
                user_B = utt.reply_to.user_id
            else: # If not a reply, it's the conversation head
                user_B = utt.talkpage_user_id
            if user_B in self.users and not user_B == user_A:
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


    def reply_pairs(self, group_A, group_B):
        """ 
        Gives utterance pairs where users in group_B are replying to utterances
        of users in group_A. 
        """
        reply_pairs = []
        utt_ids = set()
        for utt_b in self.utts.values():
            if utt_b.user_id in group_B:
                if utt_b.reply_to: 
                    utt_a = utts[utt_b.reply_to]
                    if utt_a.user_id in group_A and not utt_a.user_id == utt_b.user_id:
                        reply_pairs.append((utt_a, utt_b))
        return reply_pairs


    def coordination_given(self, group_A, group_B, ignore_ngram_repeats=False):
        """
        Calculates the coordination of group_B towards group_A (iterables of user 
        id's) along each marker as in Danescu-Niculescu-Mizil et. al. 2012, equation 3.

        We remove wholesale repetitions of phrases up to length `ignore_ngram_repeats` 
        to help ensure that only true style matching is being captured. See Danescu-
        Niculescu-Mizil et. al. 2012, footnote 11.

        Note that this is not an aggregate measure; the return value is a dictionary 
        containing, for each user in group_B, a dictionary from marker names to that 
        user's coordination given to group_A along that marker.
        """

        raw = {user : {m :{'A_b': 0, 'Am_b' : 0, 'A_bm' : 0, 'Am_bm' : 0} 
            for m in markers} for user in group_B}
        utts = self.utts.values()
        print("Scanning", len(utts), "utterances.")
        i = 0
        for u2 in utts:
            i += 1
            print("Done", i, end='\r')
            if u2.user_id in group_B:
                u1 = u2.reply_to
                if u1 and u1.user_id in group_A:
                    for m in markers:
                        raw[u2.user_id][m]['A_b'] += 1
                        if u1.exhibits(m):
                            raw[u2.user_id][m]['Am_b'] += 1
                            if u2.exhibits(m):
                                raw[u2.user_id][m]['Am_bm'] += 1
                                raw[u2.user_id][m]['A_bm'] += 1
                        elif u2.exhibits(m):
                            raw[u2.user_id][m]['A_bm'] += 1

        # we might want to do something with raw at some point like only let the marker
        # be defined for users that have more than x in A_b or Am_b.
        coords = {user : {m : raw[user][m]['Am_bm']/raw[user][m]['Am_b'] - 
                              raw[user][m]['A_bm']/raw[user][m]['A_b'] 
                          if raw[user][m]['Am_b'] and raw[user][m]['A_b'] else None 
                          for m in markers} for user in raw}
        print()
        return coords


    def coordination_received(self, group_A, group_B):
        """
        Orthoganal to the measure defined by Danescu-Niculescu-Mizil above...
        This is essentially the same measure as Corpus.coord, but with respect to the
        initiator of the reply pair. So it gives back a dictionary of whose keys are
        the users in group_A and whose values are the coordinations they receive from
        the users in group_B.
        """        
        raw = {user : {m :{'a_B': 0, 'am_B' : 0, 'a_Bm' : 0, 'am_Bm' : 0} 
            for m in markers} for user in group_A}
        utts = self.utts.values()
        print("Scanning", len(utts), "utterances.")
        i = 0
        for u2 in utts:
            i += 1
            print("Done", i, end='\r')
            if u2.user_id in group_B:
                u1 = u2.reply_to
                if u1 and u1.user_id in group_A:
                    for m in markers:
                        raw[u1.user_id][m]['a_B'] += 1
                        if u1.exhibits(m):
                            raw[u1.user_id][m]['am_B'] += 1
                            if u2.exhibits(m):
                                raw[u1.user_id][m]['am_Bm'] += 1
                                raw[u1.user_id][m]['a_Bm'] += 1
                        elif u2.exhibits(m):
                            raw[u1.user_id][m]['a_Bm'] += 1

        # we might want to do something with raw at some point like only let the marker
        # be defined for users that have more than x in A_b or Am_b.
        coords = {user : {m : raw[user][m]['am_Bm']/raw[user][m]['am_B'] - 
                              raw[user][m]['a_Bm']/raw[user][m]['a_B'] 
                        if raw[user][m]['am_B'] and raw[user][m]['a_B'] else None 
                        for m in markers} for user in raw}
        print()
        return coords


def aggregate1(coords):
    """ Only consider individuals for whom all markers are defined """
    aggregate = {user: (sum(coords[user].values())/len(markers) 
        if all(coords[user][m] != None for m in markers) else None) 
        for user in coords}
    return aggregate

def aggregate2(coords):
    """
    Where the a marker is not defined for some individual, use the group
    average for that marker.
    """
    average_per_marker = {}
    for m in markers:
        defined_for_m = {user for user in coords if coords[user][m] != None}
        average_per_marker[m] = sum([coords[user][m] 
            for user in defined_for_m])/len(defined_for_m)

    new_coords = {user : {m :(coords[user][m] 
        if coords[user][m] != None else average_per_marker[m]) for m in markers} 
        for user in coords}

    aggregate = {user : sum(new_coords[user].values())/len(markers) for user in coords}
    return aggregate

def aggregate3(coords):
    """
    Where a marker is not defined, simply leave it out of the user's aggigate
    score. This assumes that the missing marker would have shown the same level
    of coordination as the ones for which we have data.
    """
    aggregate = {}
    for user in coords:
        defined = [coords[user][m] for m in markers if coords[user][m] != None]
        if defined:
            aggregate[user] = sum(defined)/len(defined)
        else: # user doesn't have any markes defined
            aggregate[user] = None
    return aggregate

def average_aggregate(aggs):
    """ Calculates average aggregate coordination ignoring undefined values. """
    values = [agg for agg in aggs.values() if agg != None]
    return sum(values)/len(values)

def split_list(data, measure, cutoff):
    sorted_data = sorted(data, key=lambda x: measure[x], reverse=True)
    return (sorted_data[0:cutoff], sorted_data[cutoff:])


