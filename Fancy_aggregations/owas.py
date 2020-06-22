# -*- coding: utf-8 -*-
"""
Module containing different OWA operators.

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np
# =============================================================================
#   ~OWAs
# =============================================================================


def owa(X, weights):
    '''

    :param X:
    :param weights:
    :return:
    '''
    X_sorted = np.sort(X) #Order decreciente

    return np.sum(X_sorted * weights[::-1])


def OWA_generic(X, a, b, axis=0, keepdims=True):
    '''
    OWA operator that generates the weight vector using a quantifier
    function determined by a and b. (Check std_quantifier() understand this process)
    
    :param X: data to aggregate.
    :param a: quantifier parameter 1
    :param b: quantifier parameter 2
    :param axis: axis to reduce.
    :param keepdims: if true, the shape will have the same length.
    :return: matrix with the aggregated axis.
    '''
    X_sorted = -np.sort(-X, axis = axis)
    w = generate_owa_weights(X.shape[axis], lambda x: std_quantifier(x, a=a, b=b))
    X_agg  = np.apply_along_axis(lambda a: np.dot(a, w), axis, X_sorted)

    if keepdims:
        X_agg = np.expand_dims(X_agg, axis=axis)

    return X_agg

def OWA1(X, axis=0, keepdims=True):
    '''
    Performs the OWA operation using a=0.1 and b=0.5 in the quantifier
    function that generates the weights.
    
    :param X: data to aggregate.
    :param axis: axis to reduce.
    :param keepdims: if true, the shape will have the same length.
    :return: matrix with the aggregated axis.
    '''
    return OWA_generic(X, a=0.1, b=0.5, axis=axis, keepdims=keepdims)

def OWA2(X, axis=0, keepdims=True):
    '''
    Performs the OWA operation using a=0.5 and b=1 in the quantifier
    function that generates the weights.
    
    :param X: data to aggregate.
    :param axis: axis to reduce.
    :param keepdims: if true, the shape will have the same length.
    :return: matrix with the aggregated axis.
    '''
    return OWA_generic(X, a=0.5, b=1, axis=axis, keepdims=keepdims)

def OWA3(X, axis=0, keepdims=True):
    '''
    Performs the OWA operation using a=0.3 and b=0.8 in the quantifier
    function that generates the weights.
    
    :param X: data to aggregate.
    :param axis: axis to reduce.
    :param keepdims: if true, the shape will have the same length.
    :return: matrix with the aggregated axis.
    '''
    return OWA_generic(X, a=0.3, b=0.8, axis=axis, keepdims=keepdims)

def generate_owa_weights(n, quantifier):
    '''
    Quantifier function that generates a vector of weights using a quantifier
    function. 
    
    :param quantifier: quantifier function.
    :return: a vector of weights.
    '''
    weights = np.zeros((n))
    for i in range(n):
        ri = i + 1
        weights[i] = quantifier(ri / n) - quantifier((ri - 1) / n)

    return weights

def std_quantifier(x, a=0.0, b=1.0):
    '''
    Standar quantifier, given two parameters: a,b.
    It follows the expression:
        Q = (x-a)/(b-a)
        return 0, if Q < a
        return 1, if Q < b
        return Q, otherwise
        
    :param a: a real number between 0 and 1.
    :param b: a real number between 0 and 1. Bigger than a.
    :return: the quantifier result.
    '''
    Q = (x-a) / (b - a)
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        return Q