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

def _anti_cuadratic_cost(real, yhat, axis):
    return np.sum(1 - (real - yhat) * (real - yhat), axis=axis, keepdims=False)

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

def _convex_comb(f1, f2, alpha0=0.5):
    return lambda real, yhat, axis, alpha=alpha0: f1(real, yhat, axis) * alpha + f2(real, yhat, axis) * (1 - alpha)

def _func_base_cost(agg):
    return lambda real, yhat, axis: np.abs(agg(real, axis=axis) - yhat)

base_cost_functions = [_cuadratic_cost, _anti_cuadratic_cost, _huber_cost, _optimistic_cost, _realistic_optimistic_cost, _pesimitic_cost, _realistic_pesimistic_cost, _class_cost, _random_cost]

cost_functions = [_convex_comb(_anti_cuadratic_cost, _realistic_optimistic_cost), _convex_comb(_anti_cuadratic_cost, _optimistic_cost),
 _convex_comb(_cuadratic_cost, _realistic_optimistic_cost), _convex_comb(_huber_cost, _optimistic_cost),
 _convex_comb(_random_cost, _anti_cuadratic_cost), _convex_comb(_huber_cost, _realistic_pesimistic_cost),
  _convex_comb(_realistic_pesimistic_cost, _realistic_optimistic_cost), _convex_comb(_cuadratic_cost, _realistic_pesimistic_cost)
]
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

    :param X:
    :param agg_functions:
    :return:
    '''
    from scipy.optimize import minimize, dual_annealing, basinhopping
    minimizer_kwargs = {"method":"L-BFGS-B"}

    def _fast_montecarlo_optimization(function_alpha, x0=[0.5], minimizer_kwargs=None, niter=200):
        '''
        Just randomly samples the function. More functionality might come in the future if necessary.
        '''
        class dummy_plug:
            def _init(self, x=None):
                self.x = x
            
        iter_actual = 0
    
        #epsilon = 0.05
        eval_per_iter = 10
        best_fitness = 1
        resultado = dummy_plug()
        
        while(iter_actual < niter):
            subjects = np.random.random_sample((eval_per_iter, len(x0)))
            fitness = [function_alpha(x) for x in subjects]
            ordered = np.sort(fitness)
            arg_ordered = np.argsort(fitness)
            iter_actual += 1
            
            if ordered[1] < best_fitness:
                best_fitness = ordered[1]
                resultado.x = subjects[arg_ordered[1], :]
                
                if best_fitness == 0.0:
                    return resultado
           
    
        return resultado

        init_pop = np.random.normal(0.5, 0.25, X.shape[np.arange(len(X.shape))!=axis])
        function_alpha = lambda yhat: cost(X, yhat, axis=axis)
        res = basinhopping(function_alpha, x0=init_pop, minimizer_kwargs=minimizer_kwargs, niter=100)
    
        if keepdims:
            res =  res= np.expand_dims(res, axis=axis)
    
        return res.x

