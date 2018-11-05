#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from data import wiki
from data import corpus
import wham

import pystan
import pandas as pd
import pickle
from tqdm import tqdm


# In[ ]:


markers = wham.load_markers()

posts = wiki.load_posts()
posts = corpus.tokenize_posts(posts)
posts = corpus.detect_markers(posts, markers, overwrite=False)
wiki.save_posts(posts, overwrite=False)


# In[ ]:


pairs = corpus.get_reply_pairs(posts)
users = wiki.create_users(posts=posts)

network = corpus.create_network(pairs)
users = corpus.compute_centrality(users, network, normalize=True, overwrite=True)


# In[ ]:


users['admin'] = users['admin'].fillna(False)


# In[ ]:


threshold = users.centrality.mean() + users.centrality.std()
users['highly_central'] = (users['centrality'] > threshold)
users.highly_central.value_counts()


# In[ ]:


markers = wham.load_markers()
reply_pairs = pairs.join(users, on='user_a')
subpops_column = 'admin'
default = False
sample=500

reply_pairs['len_b'] = reply_pairs.tokens_b.apply(len)

# tortured reshaping
for m in markers:
    reply_pairs[m] = list(zip(reply_pairs[m+'_a'], reply_pairs[m+'_b']))
id_vars = ['utterance_id_b', 'user_a', 'user_b', subpops_column, 'len_b']
df = pd.melt(reply_pairs, id_vars=id_vars, value_vars=markers, var_name='marker')
df = df.join(pd.DataFrame(df['value'].tolist(), columns=['count_a', 'count_b']))

df['bool_a'] = df.count_a > 0
df['bool_b'] = df.count_b > 0

df['n_base'] = df.len_b.where(~df.bool_a, 0)
df['n_align'] = df.len_b.where(df.bool_a, 0)
df['c_base'] = df.count_b.where(~df.bool_a & df.bool_b, 0)
df['c_align'] = df.count_b.where(df.bool_a & df.bool_b, 0)

# change the marker & subpop labels to indices Stan will like
marker_idx = {m:i+1 for i,m in enumerate(markers)}
df['marker'] = df['marker'].apply(lambda x: marker_idx[x])
subpops = df[subpops_column].unique()
subpop_idx = {s:i+1 for i,s in enumerate(subpops)}
df[subpops_column] = df[subpops_column].apply(lambda x: subpop_idx[x])


# In[ ]:


if sample:
    df = df.sample(sample)
print(len(df))

data = {
    "NumMarkers": len(markers),
    "NumSubPops": len(subpops),
    "NumObservations": len(df),
    "SpeakerSubPop": df[subpops_column].values,
    "MarkerType": df.marker.values,
    "NumUtterancesAB": df.n_align.values,
    "NumUtterancesNotAB": df.n_base.values,
    "CountsAB": df.c_align.values,
    "CountsNotAB": df.c_base.values,
    "StdDev": .25
}

## Compile the Stan model
sm = pystan.StanModel(file='wham/alignment.cauchy.stan', verbose=True)

## Sample // fit the model to the data
import time
t_start = time.time()
fit = sm.sampling(data=data, iter=200, chains=2)
t_end = time.time()


# In[ ]:


print("Sampling time was {} hours.".format((t_end - t_start) / 3600))
print(fit.stansummary(pars=['eta_pop', 'eta_ab_pop', 'eta_subpop', 'eta_ab_subpop']))
print()
print('markers')
for i, m in marker_idx.items():
    print("{}: {}".format(m, i))
print()
print(subpops_column)
for i, m in subpop_idx.items():
    print("{}: {}".format(m, i))

with open('stan_model.pickle', 'wb') as f:
    pickle.dump(sm, f)
with open('stan_fit.pickle', 'wb') as f:
    pickle.dump(fit, f)


# In[ ]:




