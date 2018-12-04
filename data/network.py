import igraph
import numpy as np
import warnings

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


def compute_centrality(users, network, log_normalize=False):
    """ Adds Eigenvector centrality to the users list.
    Note: users not in the network will have NaN cenrality .
    """ 

    if 'centrality' in users:
        warnings.warn("Eigenvector centrality has already been computed. Skipping.")
        return users

    eigen_list, eigenvalue = network.evcent(directed=False, scale=False, weights='weight', return_eigenvalue=True)
    print("Computed eigenvector centrality for {} users. Eigenvalue was {}.".format(len(eigen_list), eigenvalue))
    vertices = [v['name'] for v in network.vs]
    c = pd.Series(eigen_list, index=vertices)
    c = c.replace({0: np.nan}) # igraph sets disconnected nodes centrality to 0 but NaN is better for us.

    if c.min() <= 0 or eigenvalue <= 0:
        ValueError("Min centrality was {} and eigenvalue was {}... could not find eigenvalue?".format(c.min(), eigenvalue))

    if log_normalize:
        from math import log
        print("Un-normalized range: [{},{}]".format(c.min(),c.max()))
        c = c[c.notna()]
        assert not any(c==0) 
        c = c.apply(lambda x: log(x))
        c_min, c_max = c.min(), c.max()  # do 0 centrality (disconnected users) for norming purposes
        c = c.apply(lambda x: (x - c_min) / (c_max - c_min))
        print("Log-normalized range: [{},{}]".format(c.min(),c.max()))

    users['centrality'] = c
    return users


