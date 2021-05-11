# -*- coding: utf-8 -*-
"""
File containing different functions to work with intervaluate data or generate intervals.

Expression taken from:
A. Jurio, M. Pagola, R. Mesiar, G. Beliakov and H. Bustince, "Image Magnification Using Interval Information," 
in IEEE Transactions on Image Processing, vol. 20, no. 11, pp. 3112-3123, Nov. 2011.
doi: 10.1109/TIP.2011.2158227
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5782984&isnumber=6045652

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np

from . import implications as _imp
from . import integrals as _int
from Fancy_aggregations import intervals as _iv

def k_alpha_operator(a, alpha_order):
    return a[1] * alpha_order + (1-alpha_order)*a[0]
    
def intervaluate(x, y, implication_operator=_imp.reichenbach_implication):
    """Returns an array with the interval composed using x, y and an implication operador.
        The new dimension is appended at the end of the array."""
    res = np.zeros((list(x.shape) + [2]))
    res[..., 0] = 1 - implication_operator(x, y)
    res[..., 1] = 1 - implication_operator(x, y) + y
    return res

def admissible_k_alpha_order(interval, interval2, alpha=0.5, beta=0.1):
    k_alpha_operator = lambda a, alpha_order: a[1] * alpha_order + (1-alpha_order)*a[0]
    
    if k_alpha_operator(interval, alpha) > k_alpha_operator(interval2, alpha):
        return interval
    elif k_alpha_operator(interval, alpha) < k_alpha_operator(interval2, alpha):
        return interval2
    else:
        if k_alpha_operator(interval, beta) > k_alpha_operator(interval2, beta):
            return interval
        else:
            return interval2
    
def arg_admissible_k_alpha_order(interval, interval2, alpha=0.5, beta=0.1):    
    if k_alpha_operator(interval, alpha) > k_alpha_operator(interval2, alpha):
        return 0
    elif k_alpha_operator(interval, alpha) < k_alpha_operator(interval2, alpha):
        return 1
    else:
        if k_alpha_operator(interval, beta) > k_alpha_operator(interval2, beta):
            return 0
        else:
            return 1
            

def admissible_intervalued_array_sort(X, axis=0, keepdims=False, alpha_order=0.5, beta_order=0.1):
    if axis != len(X.shape)-2:
        X = np.swapaxes(X, axis, len(X.shape)-2)
        
    #Iterate over all but the last 2 dimensions
    idx = np.ndindex(X.shape[:-2])
    
    res = np.zeros(X.shape)
    for index in idx:
        res[index] = sorted(X[index], key=lambda a: (k_alpha_operator(a, alpha_order), k_alpha_operator(a, beta_order)))
    
    if axis != len(X.shape)-2:
        res = np.swapaxes(res, axis, len(X.shape)-2)

    return res

def admissible_intervalued_array_argsort(X, axis=0, alpha_order=0.5, beta_order=0.1):
    if axis != len(X.shape)-2:
            X = np.swapaxes(X, axis, len(X.shape)-2)
            
    #Iterate over all but the last 2 dimensions
    idx = np.ndindex(X.shape[:-2])
    
    res = np.zeros(list(X.shape[:-2]) + [1])
    for index in idx:
        res[index] = sorted(range(X[index].shape[0]), key=lambda a: (k_alpha_operator(X[index][a,:], alpha_order), k_alpha_operator(X[index][a,:], beta_order)))[-1]
    
    if axis != len(X.shape)-2:
        res = np.swapaxes(res, axis, len(X.shape)-2)

    return np.squeeze(res)
    
def sugeno_general(X, alpha_order, beta_order, tnorm=np.minimum, tconorm=np.max, axis=0, keepdims=True):
    # sort interval according to alpha and beta
    features, samples, clases, interl_dim = X.shape
    #increasing_iv = sorted(X, key=lambda a: (_iv.k_alpha_operator(a, alpha_order), _iv.k_alpha_operator(a, beta_order)))
    sorted_X = admissible_intervalued_array_sort(X, axis=axis, keepdims=False, alpha_order=alpha_order, beta_order=beta_order)
    medida = _int.generate_cardinality(X.shape[axis], 2)
    oneshapes = np.ones(len(X.shape)-1, dtype=np.int32)
    oneshapes[axis] = len(medida)
    medida = np.array(medida).reshape(oneshapes)

    if tnorm == np.prod:
    	iv1 = sorted_X[:, :, :, 0] * medida
    	iv2 = sorted_X[:, :, :, 1] * medida

    else:
        iv1 = tnorm(sorted_X[:, :, :, 0], medida)
        iv2 = tnorm(sorted_X[:, :, :, 1], medida)

    sorted_X[:, :, :, 0] = iv1
    sorted_X[:, :, :, 1] = iv2

    if tconorm == np.max:
        sorted_X = admissible_intervalued_array_sort(sorted_X, axis=axis, keepdims=False, alpha_order=alpha_order, beta_order=beta_order)
        res = sorted_X[-1, :, :, :]

    if not keepdims:
        res = np.squeeze(res)

    return res

def sugeno(X, alpha_order, beta_order, axis=0, keepdims=True):
    return sugeno_general(X, alpha_order, beta_order, tnorm=np.minimum, tconorm=np.max, axis=axis, keepdims=keepdims)
    