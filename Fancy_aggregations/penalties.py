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
# ~ Cost functions
# =============================================================================
# They all sould hold the interface: (real, yhat, axis) as inputs.
# 
def _cuadratic_cost(real, yhat, axis):
	return np.sum((real - yhat) * (real - yhat), axis=axis, keepdims=False)

def _huber_cost(real, yhat, axis, M=0.3):
	r2_cost = _cuadratic_cost(real, yhat, axis)
	root_cost = np.sqrt(r2_cost)
	outlier_detected = root_cost > M

	outlier_costs = 2 * M * root_cost - M * M

	return root_cost * (1 - outlier_detected) + outlier_costs * outlier_detected

def _random_cost(real, yhat, axis):
    return np.sum(np.abs(0.5 - yhat), axis=axis, keepdims=False)

def _class_cost(real, yhat, axis):
    return np.sum(1 - np.abs(0.5 - yhat), axis=axis, keepdims=False)

def _optimistic_cost(real, yhat, axis):
	return np.sum(1 - yhat, axis=axis, keepdims=False)

def _realistic_optimistic_cost(real, yhat, axis):
	return np.sum(np.max(real, axis=axis, keepdims=True) - yhat, axis=axis, keepdims=False)

def _pesimitic_cost(real, yhat, axis):
    return np.sum(yhat, axis=axis, keepdims=False)

def _realistic_pesimistic_cost(real, yhat, axis):
    return np.sum(yhat - np.min(real, axis=axis, keepdims=True), axis=axis, keepdims=False)


cost_functions = [_cuadratic_cost, _huber_cost, _optimistic_cost, _realistic_optimistic_cost, _pesimitic_cost, _realistic_pesimistic_cost, _class_cost, _random_cost]
# =============================================================================
# ~ PENALTY
# =============================================================================
def penalty_aggregation(X, agg_functions, axis=0, keepdims=False, cost=_cuadratic_cost):
    '''

    :param X:
    :param agg_functions:
    :return:
    '''
    agg_matrix = []
    agg_distances_shape =  [len(agg_functions)] + list(agg_functions[0](X, axis=axis, keepdims=False).shape)
    agg_distances = np.zeros(agg_distances_shape)


    for ix, ag_f in enumerate(agg_functions):
        aux = ag_f(X, axis=axis, keepdims=True)
        distances = cost(X, aux, axis)
        aux = ag_f(X, axis=axis, keepdims=False)

        agg_matrix.append(aux)
        agg_distances[ix] = distances

    agg_choose = np.argmin(agg_distances, axis=0)
    res = np.zeros(agg_choose.shape)
    for index, x in np.ndenumerate(agg_choose):
        res[index] = agg_matrix[x][index]

    if keepdims:
        res =  res= np.expand_dims(res, axis=axis)

    return res