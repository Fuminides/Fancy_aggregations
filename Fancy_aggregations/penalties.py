# -*- coding: utf-8 -*-
"""
File containing different penalty functions to aggregate data.

Bustince, H., Beliakov, G., Dimuro, G. P., Bedregal, B., & Mesiar, R. (2017). 
On the definition of penalty functions in data aggregation. Fuzzy Sets and Systems, 323, 1-18.

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np

# =============================================================================
# ~ PENALTY
# =============================================================================

def penalty_aggregation(X, agg_functions):
    '''

    :param X:
    :param agg_functions:
    :return:
    '''
    min_distances = np.Inf
    res = -1

    for ag_f in agg_functions:
        aux = ag_f(X)
        distance = np.sum(np.abs(X - aux))

        if distance < min_distances:
            res = aux
            min_distances = distance

    return res