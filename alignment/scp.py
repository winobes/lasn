import numpy as np
import pandas as pd
from .util import load_markers

markers = load_markers()
markers_list = list(markers)
# convienance lists of column names for speakers a/b
markers_a = [m+'_a' for m in markers]
markers_b = [m+'_b' for m in markers]

def scp(df, groupA):
    """ SCP coordination as defined by Echoes of Power (DNM 2014).
    df     is a dataframe with columns: groupA+'_a', 'user_a', 'user_b',
           and marker+'_a', marker+'_b' for each marker.
    groupA is a the column distinguishing the sets of users to measure
           coordination *towards* (i.e. everyone with the same groupA value
           is treated as the same "target"). 
    """
    
    groupA = groupA+'_a'
   
    # binarize marker presence
    ma = (df[markers_a] > 0).rename(dict(zip(markers_a, markers_list)), axis=1)
    mb = (df[markers_b] > 0).rename(dict(zip(markers_b, markers_list)), axis=1)
    mab = ma & mb

    ma[groupA], ma['user_b'] = df[groupA], df['user_b']
    mb[groupA], mb['user_b'] = df[groupA], df['user_b']
    mab[groupA], mab['user_b'] = df[groupA], df['user_b']
    
    a = ma.groupby([groupA, 'user_b']).sum()
    b = mb.groupby([groupA, 'user_b']).sum()
    ab = mab.groupby([groupA, 'user_b']).sum()
    total = df.groupby([groupA, 'user_b']).size()
    total = pd.concat([total] * len(markers), keys=markers_list, axis=1) # reshape to match others

    p_base = b.div(total)
    p_cond = ab.div(a)
    p_cond = p_cond.replace([np.inf, -np.inf], np.nan)
    
    scp = (p_cond - p_base)
    
    agg1 = scp.dropna().mean(axis=1)
    agg2 = scp.fillna(scp.mean(level=0)).mean(axis=1)
    agg3 = scp.transpose().fillna(scp.mean(axis=1)).transpose().mean(axis=1)
    scp['agg1'], scp['agg2'], scp['agg3'] = agg1, agg2, agg3
    
    return scp

def wscp(posts, df, groupA):
    
    def base_prob_m(m):
        base_m_count = posts.groupby('user')[m].sum()
        base_word_count = posts.groupby('user')['length'].sum()
        base_prob = base_m_count / base_word_count
        return base_prob

    def cond_prob_m(m):
        cond_m_count = df[ma_bin[m]].groupby([groupA+'_a', 'user_b'])[m+'_b'].sum()
        cond_word_count = df[ma_bin[m]].groupby([groupA+'_a', 'user_b'])['length_b'].sum()
        cond_prob = cond_m_count / cond_word_count
        return cond_prob

    ma_bin = (df[markers_a] > 0).rename(dict(zip(markers_a, markers_list)), axis=1)
    base_prob = pd.DataFrame({m: base_prob_m(m) for m in markers})
    cond_prob = pd.DataFrame({m: cond_prob_m(m) for m in markers})
    
    wscp = cond_prob.subtract(base_prob, level=1)
    
    agg1 = wscp.dropna().mean(axis=1)
    agg2 = wscp.fillna(wscp.mean(level=0)).mean(axis=1)
    agg3 = wscp.transpose().fillna(wscp.mean(axis=1)).transpose().mean(axis=1)
    wscp['agg1'], wscp['agg2'], wscp['agg3'] = agg1, agg2, agg3
    
    return wscp
