#!/usr/bin/env python
# coding: utf-8


from data import wiki
from data import corpus
import wham

import pystan
import pandas as pd
import pickle
from tqdm import tqdm

sample=1000
df = pandas.read_pickle('wham_df_2018-11-20.pickle')

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

