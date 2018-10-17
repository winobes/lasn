#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from data import wiki
from data import corpus

from experiments.util import load_markers
import pandas as pd
import pystan


# In[ ]:


posts = wiki.load_posts()
pairs = corpus.get_reply_pairs(posts)


# ## Format the input data for Stan

# In[ ]:



markers = load_markers()

# merge the marker usage columns for the reply pair
for m in markers:
    pairs[m] = list(zip(pairs[m+'_a'], pairs[m+'_b']))

# reshape
df = pd.melt(pairs, id_vars = ['user_a', 'user_b', 'utterance_id_b'], value_vars=markers, var_name='marker')

# change the marker labels to indices Stan will like
marker_idx = {m:i+1 for i,m in enumerate(markers)}
df['marker'] = df['marker'].apply(lambda x: marker_idx[x])

# reshape again
df = df.pivot_table(index=['user_a', 'user_b', 'marker'], columns='value', aggfunc='size', fill_value=0)
df = df.reset_index()

# df = df.sample(500)
print(len(df))


# In[ ]:


data = {
    "NumMarkers": len(markers),
    "NumObservations": len(df),
    "MarkerType": df.marker.values,
    "NumUtterancesAB": (df[(True, True)] + df[(True, False)]).values,
    "NumUtterancesNotAB": (df[(False, True)] + df[(False, False)]).values,
    "CountsAB": df[(True, True)].values,
    "CountsNotAB": df[(False, True)].values,
    "StdDev": .25
}


# ### Compile the Stan model

# In[ ]:


sm = pystan.StanModel(file='experiments/alignment.cauchy.nosubpop.stan', verbose=True)


# ### Fit the Stan model to the data
# save the paramteer `eta_ab_pop`

# In[ ]:


import time

start = time.time()
fit = sm.sampling(data=data, iter=200, pars=['eta_ab_pop'], chains=4)
end = time.time()
print(end - start)


# In[ ]:


print(fit.stansummary(probs=[.95]))
print()
for i, m in marker_idx.items():
    print("{}: {}".format(m, i))


# In[ ]:


import pickle
with open('stan_model.pickle', 'wb') as f:
    pickle.dump(sm, f)
with open('stan_fit.pickle', 'wb') as f:
    pickle.dump(fit, f)


# In[ ]:





# In[ ]:




