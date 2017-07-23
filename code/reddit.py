import json
import bz2
import random
import csv
import os
import pickle
from collections import Counter
from datetime import datetime
import networkx as nx

year_from = 2010
year_thru = 2012

path = "../data/reddit/"
comment_counts_filename = path + "comment_counts_{:04}.json"  # format by year
chosen_subreddits_filename = path + "subreddits.csv"
subreddit_comments_filename = path + "subreddits/{}_{:04}-{:02}.json"
zip_filename = path + "RC_{:04}-{:02}.bz2"  # format by year, month
subreddit_pickle_filename = path + "subreddits/{}_{:04}to{:04}.pickle"



def create_comment_counts(year, month):
    
    try:
        with open(comment_counts_filename.format(year), 'r') as f:
            comment_counts = json.load(f)
    except FileNotFoundError:
        comment_counts = {}

    month_counts = Counter()
    filename = zip_filename.format(year,month)
    print("Counting comments in {}".format(filename))
    with bz2.open(filename) as zipfile:
        for line in zipfile:
            line = line.decode('utf-8')
            comment = json.loads(line)
            month_counts[comment['subreddit']] += 1

    comment_counts["{}-{:02}".format(year, month)] = month_counts 
    with open(comment_counts_filename.format(year), 'w') as f:
        json.dump(comment_counts, f)


def load_comment_counts():
    comment_counts = {}
    for year in range(year_from, year_thru + 1):
        with open(comment_counts_filename.format(year)) as f:
            comment_counts_year = json.load(f)
            comment_counts[year] = {int(month[-2:]): comment_counts_year[month] for month in comment_counts_year}
    return comment_counts


def choose_random_subreddits(n_subreddits, min_count_per_month, blacklist=None):
    comment_counts = load_comment_counts()
    initial_month = True
    for year in comment_counts:
        for month in comment_counts[year]:
            counts = comment_counts[year][month]
            if initial_month:
                subreddits = {subreddit for subreddit in counts if counts[subreddit] >= min_count_per_month}
                initial_month = False
            else:
                for subreddit in counts:
                    if counts[subreddit] < min_count_per_month:
                        subreddits.discard(subreddit)
    if blacklist:
        # TODO load blacklist
        for subreddit in blacklist:
            subreddits.discard(subreddit)

    print("choosing from {} total subreddits".format(len(subreddits)))
    subreddits = random.sample(subreddits, n_subreddits)
    with open(chosen_subreddits_filename, 'w') as f:
        writer = csv.writer(f)
        print(len(subreddits))
        writer.writerows([[s] for s in subreddits])


def extract_chosen_subreddits(year):

    with open(chosen_subreddits_filename, 'r') as f:
        reader = csv.reader(f)
        subreddits = [line[0] for line in reader]

    for month in range(1, 13):
        comments = {subreddit: [] for subreddit in subreddits}
        filename = zip_filename.format(year,month)
        print('Searching {} for comments from {} selected subreddits.'.format(filename, len(subreddits)))
        with bz2.open(filename) as zipfile:
            for line in zipfile:
                line = line.decode('utf-8')
                comment = json.loads(line)
                subreddit = comment['subreddit']
                if subreddit in subreddits:
                    comments[subreddit].append(comment)
        for subreddit in subreddits:
            filename = '{}_{:04}-{:02}.json'.format(subreddit, year, month)
            with open('reddit/subreddits/' + filename, 'w') as f:
                json.dump(comments[subreddit], f)


class Corpus:

    def __init__(self, subreddits):
        self.subreddits = subreddits

    @classmethod
    def load_json(cls, subreddits_file):
        subreddits = []
        with open(subreddits_file) as f:
            reader = csv.reader(f)
            subreddits_list = [line[0] for line in reader]
        for subreddit in subreddits_list:
            subreddits.append(Subreddit(subreddit))
        return cls(subreddits)

    @classmethod
    def load_pickle(cls, subreddits_file):
        subreddits = []
        with open(subreddits_file) as f:
            reader = csv.reader(f)
            subreddits_list = [line[0] for line in reader]
        for subreddit in subreddits_list:
            subreddits.append(Subreddit.from_pickle(subreddit))
        return cls(subreddits)
        
    def save_pickle(self):
        for subreddit in self.subreddits:
            subreddit.save_pickle()


class User:

    def __init__(self, subreddit, comments, id):
        self.comments = comments
        self.id = id
        self.subreddit = subreddit


class Subreddit:

    def __init__(self, subreddit_name):
        comments = []
        for year in range(year_from, year_thru):
            for month in range(1, 13):
                with open(subreddit_comments_filename.format(subreddit_name, year, month)) as f:
                    comments += json.load(f)
        comments = {comment['id']: Comment(comment) for comment in comments}
        
        # create users
        users = {}
        for comment in comments.values():
            if comment.author:
                if comment.author in users:
                    users[comment.author].comments.append(comment)
                else:
                    users[comment.author] = User(self, [comment], comment.author)

        # create network
        missing_comment_links = 0
        network = nx.Graph()
        for comment in comments.values():
            if not comment.author:
                missing_comment_links += 1
                continue
            user_a = users[comment.author]
            if comment.reply_to in comments and comments[comment.reply_to].author in users:
                user_b = users[comments[comment.reply_to].author]
                if network.has_edge(user_a, user_b):
                    network[user_a][user_b]['weight'] += 1
                else:
                    network.add_edge(user_a, user_b, weight=1)
            else:
                missing_comment_links += 1
                continue

        self.name = subreddit_name
        self.comments = comments
        self.users = users
        self.network = network
        self.missing_comment_links = missing_comment_links


    def save_pickle(self):
        with open(subreddit_pickle_filename.format(self.name, year_from, year_thru), 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load_pickle(cls, subreddit_name):
        with open(subreddit_pickle_filename.format(self.name, year_from, year_thru), 'rb') as f:
            subreddit = pickle.load(f)
            return subreddit
         
        
class Comment:

    def __init__(self, comment_dict):
        self.id = comment_dict['id']
        self.score = comment_dict['score']
        self.down_votes = comment_dict['downs']
        self.up_votes = comment_dict['ups']
        self.edited = comment_dict['edited']
        self.timestamp = datetime.fromtimestamp(float(comment_dict['created_utc']))
        self.author = comment_dict['author'] if comment_dict['author'] != '[deleted]' else None
        self.reply_to = comment_dict['parent_id']



if __name__ == '__main__':
    import sys
    
    if sys.argv[1] == 'count-comments':
        year = int(sys.argv[2])
        for month in range(1, 13):
            try:
                create_comment_counts(year,month)
            except FileNotFoundError:
                print('No source file for month {}'.format(month))

    elif sys.argv[1] == 'choose-subreddits':
        n_subreddits = int(sys.argv[2])
        min_count_per_month = int(sys.argv[3])
        blacklist = None
        subreddits = choose_random_subreddits(n_subreddits, min_count_per_month, blacklist)

    elif sys.argv[1] == 'extract-subreddits':
        year = int(sys.argv[2])
        extract_chosen_subreddits(year)

    elif sys.argv[1] == 'create-corpus':
        subreddits_file = sys.argv[2] 
        subreddits_file = 'reddit/' + subreddits_file
        corpus = Corpus.load_json(subreddits_file)
        corpus.save_pickle()

