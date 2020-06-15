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
	return np.mean((real - yhat)**2, axis=axis, keepdims=False)

def _anti_cuadratic_cost(real, yhat, axis):
    return np.mean(1 - (real - yhat)**2, axis=axis, keepdims=False)

def _huber_cost(real, yhat, axis, M=0.3):
	r2_cost = _cuadratic_cost(real, yhat, axis)
	outlier_detected = r2_cost > M

	outlier_costs = 2 * M * r2_cost - M * M

	return r2_cost * (1 - outlier_detected) + outlier_costs * outlier_detected

def _random_cost(real, yhat, axis):
    return np.mean((0.5 - yhat)**2, axis=axis, keepdims=False)

def _optimistic_cost(real, yhat, axis):
	return np.mean((1 - yhat)**2, axis=axis, keepdims=False)

def _realistic_optimistic_cost(real, yhat, axis):
	return np.mean((np.max(real, axis=axis, keepdims=True) - yhat)**2, axis=axis, keepdims=False)

def _pessimistic_cost(real, yhat, axis):
    return np.mean(yhat**2, axis=axis, keepdims=False)

def _realistic_pesimistic_cost(real, yhat, axis):
    return np.mean((yhat - np.min(real, axis=axis, keepdims=True))**2, axis=axis, keepdims=False)

def _convex_comb(f1, f2, alpha0=0.5):
    return lambda real, yhat, axis, alpha=alpha0: f1(real, yhat, axis) * alpha + f2(real, yhat, axis) * (1 - alpha)

def _convex_quasi_comb(f1, f2, alpha0=0.5):
    return lambda real, yhat, axis, alpha=alpha0: np.minimum((f1(real, yhat, axis) * alpha + f2(real, yhat, axis) * (1 - alpha))/(1 - alpha),1)

def _func_base_cost(agg):
    return lambda real, yhat, axis: np.abs(agg(real, axis=axis) - yhat)

base_cost_functions = [_cuadratic_cost, _realistic_optimistic_cost, _random_cost, _anti_cuadratic_cost, _huber_cost, _realistic_pesimistic_cost]

cost_functions = [_convex_comb(_cuadratic_cost, _realistic_optimistic_cost),
 _convex_comb(_huber_cost, _realistic_optimistic_cost), 
 _convex_quasi_comb(_anti_cuadratic_cost, _optimistic_cost),
 _convex_quasi_comb(_huber_cost, _anti_cuadratic_cost)]

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

def penalty_optimization(X, agg_functions, axis=0, keepdims=False, cost=_cuadratic_cost):
    '''
    EXPERIMENTAL: instead of computing the penalty function using aggregation functions,
    it uses an optimization algorithm to reduce the cost. (More costly)
    :param X:
    :param agg_functions:
    :return:
    '''
    from scipy.optimize import basinhopping
    minimizer_kwargs = {"method":"L-BFGS-B"}

    init_pop = np.random.normal(0.5, 0.25, X.shape[np.arange(len(X.shape))!=axis])
    function_alpha = lambda yhat: cost(X, yhat, axis=axis)
    res = basinhopping(function_alpha, x0=init_pop, minimizer_kwargs=minimizer_kwargs, niter=100)

    if keepdims:
        res =  res= np.expand_dims(res, axis=axis)

    return res.x

