""" Functions for common low-level corpus manipulation """

import warnings
from tqdm import tqdm

import pandas as pd
import nltk
import igraph

from collections import Counter, defaultdict

def tokenize_posts(posts, overwrite=False):
    """ Add a 'tokens' column to the posts dataframe """

    if not overwrite and 'tokens' in posts:
        warnings.warn("Posts are already tokenized. Skipping tokenization.")
        return posts
    
    tokens = [nltk.tokenize.word_tokenize(text) for text in tqdm(posts['clean_text'], desc="Tokenizing posts.")]
    posts = posts.assign(tokens=tokens)

    n = len(posts)
    posts = posts[posts.tokens.map(len) > 0]
    print("Filtered {} posts with 0-length utterances".format(n - len(posts)))

    return posts

def pos_tag_posts(posts, overwrite=False):
    """ Add a 'pos_tags' column to the posts dataframe """

    if not overwrite and 'pos_tags' in posts:
        warnings.warn("Posts are already PoS tagged. Skipping tagging.")
        return posts
    
    pos = posts.tokens.progress_apply(nltk.pos_tag)
    posts['pos_tags'] = pos.apply(lambda pos_list: list(map(lambda t: t[1], pos_list)))
    
    return posts

def sublists(l, n):
    """ generate all sublists of length n (where order matters) """
    return (l[x:x+n] for x in range(len(l) - n + 1))

def detect_markers(posts, markers, overwrite=False):
    """ Add feature columns for marker counts.

    Markers is i dictionary of lists where the keys are the marker types and
    the values are a list of markers.
    """

    if not 'tokens' in posts:
        raise ValueError("Corpus must be tokenized for marker detection.")


    if not overwrite and all(m in posts for m in markers):
        warnings.warn("All marker columns already exist. Skipping marker detection.")
        return posts

    for m in markers.keys():
        # sort the marker instances by length for efficency
        marker_insts = defaultdict(list)
        for marker_inst in markers[m]:  
            marker_insts[len(marker_inst)].append(marker_inst)
        counts = []
        for i, tokens in enumerate(tqdm(posts['tokens'], desc="Detecting {}.".format(m))):
            c = 0
            for m_len in marker_insts:
                for subseq in sublists(tokens, m_len):
                    if subseq in marker_insts[m_len]:
                        c += 1
            counts.append(c)
        posts[m] = counts

    return posts 


def get_reply_pairs(posts, filter_self_replies=True):
    """ View the posts dataframe as reply pairs """
    pairs = pd.merge(posts, posts, how='inner', left_index=True, right_on='reply_to', suffixes=['_a', '_b'])
    if filter_self_replies:
        pairs = pairs[(pairs.user_a != pairs.user_b)]
    return pairs


def create_network(reply_pairs):
    """ Creates an undirected (symmetric), weighted social network based on reply pair counts. """

    directed_edges = Counter(reply_pairs[['user_a', 'user_b']].itertuples(index=False, name=None))

    # for the undirected edge weights, sum the directed communication between each pair
    undirected_edges = Counter()
    for (a,b), count in directed_edges.items():
        if (b,a) in undirected_edges:
            undirected_edges[(b,a)] += count
        else:
            undirected_edges[(a,b)] = count
            
    edges = list(undirected_edges.keys())
    edge_weights = [undirected_edges[e] for e in edges]

    # include both ends of communication in the vertex list
    vertices = set()
    for (a, b), count in undirected_edges.items():
        vertices.add(a)
        vertices.add(b)
    vertices = list(vertices)

    g = igraph.Graph()
    for v in vertices:
        g.add_vertex(name=v)
        
    g.add_edges(edges)
    g.es['weight'] = edge_weights

    return g

def compute_centrality(users, network, overwrite=False, normalize=False):
    """ Adds Eigenvector centrality to the users list.
    Note: users not in the network will have NaN cenrality .
    """ 

    if not overwrite and 'centrality' in users:
        warnings.warn("Eigenvector centrality has already been computed. Skipping.")
        return users

    eigen_list, eigen_value = network.evcent(directed=False, scale=normalize, weights='weight', return_eigenvalue=True)
    vertices = [v['name'] for v in network.vs]
    centrality = pd.DataFrame(data = eigen_list, index=vertices, columns=['centrality'])
    print("Computed eigenvector centrality for {} users. Eigenvalue was {}.".format(len(eigen_list), eigen_value))

    return users.join(centrality, how='left')

