import pandas as pd
import pystan

def fit_wham_nosubpops(reply_pairs, markers, sample=None):
    
    # merge the marker usage columns for the reply pair
    for m in markers:
        reply_pairs[m] = list(zip(reply_pairs[m+'_a'], reply_pairs[m+'_b']))

    # reshape
    df = pd.melt(reply_pairs, id_vars = ['user_a', 'user_b', 'utterance_id_b'], value_vars=markers, var_name='marker')

    # change the marker labels to indices Stan will like
    marker_idx = {m:i+1 for i,m in enumerate(markers)}
    df['marker'] = df['marker'].apply(lambda x: marker_idx[x])

    # reshape again
    df = df.pivot_table(index=['user_a', 'user_b', 'marker'], columns='value', aggfunc='size', fill_value=0)
    df = df.reset_index()

    if sample:
        df = df.sample(sample)
    print(len(df))

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

    ## Compile the Stan model
    sm = pystan.StanModel(file='alignment.cauchy.nosubpop.stan', verbose=True)

    ## Sample // fit the model to the data
    import time
    start = time.time()
    fit = sm.sampling(data=data, iter=200, pars=['eta_ab_pop'], chains=2)
    end = time.time()
    print(end - start)


def fit_wham(reply_pairs, subpops_column, markers, sample=None):
    pass

