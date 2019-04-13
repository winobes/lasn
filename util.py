import scipy
import numpy as np
import math

def binary_comparison_stats(data, diff):
    """ data: series of floats
        diff: series of booleans marking the comparison groups
    """
    d1, d2 = data[diff], data[~diff]
    mean1, mean2 = d1.mean(), d2.mean()
    std1, std2 = d1.std(), d2.std()
    n1, n2 = len(d1), len(d2)
    t_stat, p_value = scipy.stats.ttest_ind(d1, d2, nan_policy='omit')
    pooled_std = math.sqrt((((n1 - 1) * std1**2) + ((n2 - 1) * std2**2))/(n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    print('|Condition |   True   |   False   |')
    print('|---------------------------------|')
    print('|        N | {:>8} | {:>8}  |'.format(n1,n2))
    print('|     mean | {: 8.2f} | {: 8.2f}  |'.format(mean1,mean2))
    print('|   stddev | {: 8.2f} | {: 8.2f}  |'.format(std1,std2))
    print('|---------------------------------|')
    print("t-statistic: {: 4.2f} p-value: {: 4.3f}".format(t_stat, p_value))
    print("Cohen's D: {: 4.2f}".format(cohens_d))
    print()
