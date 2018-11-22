""" Replicating WHAM alignment (Doyle, 2016) on the Wiki corpus (nosubpop) """

from data import wiki
from data import corpus
import alignment 

posts = wiki.load_posts()
posts = corpus.tokenize_posts(posts)

markers = alignment.load_markers()
posts = corpus.detect_markers(posts, markers)

print("getting reply pairs")
pairs = corpus.get_reply_pairs(posts)

print("fitting model")
model, fit = alignment.fit_wham_nosubpops(pairs, markers, sample=500)

print(fit.stansummary())
print()
for i, m in marker_idx.items():
    print("{}: {}".format(m, i))

import pickle
with open('stan_model.pickle', 'wb') as f:
    pickle.dump(sm, f)
with open('stan_fit.pickle', 'wb') as f:
    pickle.dump(fit, f)

