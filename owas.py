# -*- coding: utf-8 -*-
"""
File containing different OWA operators.

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
    X_sorted = -np.sort(-X, axis = axis)
    w = generate_owa_weights(X.shape[axis], lambda x: std_quantifier(x, a=0.1, b=0.5))
    X_agg  = np.apply_along_axis(lambda a: np.dot(a, w), axis, X_sorted)

    if keepdims:
        X_agg = np.expand_dims(X_agg, axis=axis)

    return X_agg

def OWA1(X, axis=0, keepdims=True):
    return OWA_generic(X, a=0.1, b=0.5, axis=axis, keepdims=keepdims)

def OWA2(X, axis=0, keepdims=True):
    return OWA_generic(X, a=0.5, b=1, axis=axis, keepdims=keepdims)

def OWA3(X, axis=0, keepdims=True):
    return OWA_generic(X, a=0.3, b=0.8, axis=axis, keepdims=keepdims)

def generate_owa_weights(n, quantifier):
    '''

    :param quantifier:
    :return:
    '''
    weights = np.zeros((n))
    for i in range(n):
        ri = i + 1
        weights[i] = quantifier(ri / n) - quantifier((ri - 1) / n)

    return weights

def std_quantifier(x, a=0.0, b=1.0):
    '''

    :param a:
    :param b:
    :return:
    '''
    Q = x - a / (b - a)
    if Q < a:
        return 0
    elif Q > b:
        return 1
    else:
        return Q