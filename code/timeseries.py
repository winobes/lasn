import wiki
import networkx as nx
import csv
import sys
from datetime import datetime
from collections import defaultdict


if sys.argv[1] == 'extract':

    data = {}

    for year in range(2006, 2012):

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        print("Gathering", year, "data.")
        corpus = wiki.Corpus('../data/corpus/', start_date, end_date)
        network = corpus.generate_network()

        print("Calculating Egienvector centrality")
        eigenvector = nx.eigenvector_centrality_numpy(network)

        print("Calculating coordination given/received.")
        coordination_given = {}
        coordination_received = {}
        for i, user in enumerate(network.nodes()):
            print('user', i, 'of', len(network.nodes()), end='\r')
            coordination_given[user] = corpus.coordination_given(user)
            coordination_received[user] = corpus.coordination_received(user)

        data[year] = {user: [eigenvector[user], coordination_given[user], coordination_received[user]]
            for user in network.nodes()}

        print()


    for year in range(2006,2012):
        with open('../data/' + str(year) + '_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['user', 'e_central', 'coord_given', 'coord_received'])
            writer.writerows([[user] + data[year][user] for user in data[year]])



if sys.argv[1] == 'reformat':
    data = defaultdict(dict)
    for year in range(2006,2012):
        with open('../data/' + str(year) + '_data.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row['user']][year] = dict(row)
    for user in data:
        for year in data[user]:
            next_year = year + 1
            if next_year in data[user]:
                e_central_chg = float(data[user][next_year]['e_central']) - float(data[user][year]['e_central'])
                data[user][year]['e_central_chg'] = e_central_chg

    data_flat = []
    for user in data:
        for year in data[user]:
            row = data[user][year]
            if 'e_central_chg' in row:
                data_flat.append([user, year, row['coord_given'], row['coord_received'], row['e_central'], row['e_central_chg']])
    with open('../data/chg_e_central_y2y.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['user', 'year', 'coord_given', 'coord_received', 'e_central', 'e_central_chg'])
        writer.writerows(data_flat)



                



