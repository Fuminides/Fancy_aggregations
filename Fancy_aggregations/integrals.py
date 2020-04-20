# -*- coding: utf-8 -*-
"""
File containing different discrete integrals to aggregate data.


For the CF12 look:
Graçaliz Pereira Dimuro, Giancarlo Lucca, Benjamín Bedregal, Radko Mesiar, José Antonio Sanz, Chin-Teng Lin, Humberto Bustince,
Generalized CF1F2-integrals: From Choquet-like aggregation to ordered directionally monotone functions,
Fuzzy Sets and Systems,
Volume 378,
2020,
Pages 44-67,
ISSN 0165-0114,
https://doi.org/10.1016/j.fss.2019.01.009.
(http://www.sciencedirect.com/science/article/pii/S0165011418305451)

For the Sugeno generalization:
Ko, L. W., Lu, Y. C., Bustince, H., Chang, Y. C., Chang, Y., Ferandez, J., ... & Lin, C. T. (2019). 
Multimodal Fuzzy Fusion for Enhancing the Motor-Imagery-Based Brain Computer Interface. IEEE Computational Intelligence Magazine, 14(1), 96-106.

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np
from . import tnorms
# =============================================================================
# ~ MEASURES
# =============================================================================
def _differentiation_1_distance(X):
    #Perform differentiation for each consecuent point in the X dataset (time series)
    return np.append(X[0], X[1:] - X[0:-1])


def generate_cardinality(N, p = 2):
    '''
    Generate the cardinality measure for a N-sized vector.
    '''
    return [(x/ N)**p for x in np.arange(N, 0, -1)]


def generate_cardinality_matrix(N, matrix_shape, p = 2):
    '''
    Generate the cardinality measure for a N-sized vector, and returns it in a matrix shape.
    Use this if you cannot broadcast generate_cardinality() correctly.
    N and matrix_shape must be coherent (matrix_shape[0] == N)
    '''
    res = np.zeros(matrix_shape)
    dif_elements = [(x/ N)**p for x in np.arange(N, 0, -1)]

    for ix, elements in enumerate(dif_elements ):
        res[ix,...] = dif_elements[ix]

    return res


# =============================================================================
# ~ INTEGRALS
# =============================================================================
def choquet_integral_symmetric(X, measure=None, axis=0, keepdims=True):
    '''
    Aggregates a numpy array alongise an axis using the choquet integral.
    
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])

    X_sorted = np.sort(X, axis = axis)

    X_differenced = np.apply_along_axis(_differentiation_1_distance, axis, X_sorted)
    X_agg  = np.apply_along_axis(lambda a: np.dot(a, measure), axis, X_differenced)

    if keepdims:
        X_agg = np.expand_dims(X_agg, axis=axis)

    return X_agg

def choquet_integral_CF(X, measure=None, axis=0, tnorm=tnorms.hamacher_tnorm, keepdims=True):
    '''
    Aggregates a numpy array alongise an axis using the choquet integral.

    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])

    X_sorted = np.sort(X, axis = axis)

    X_differenced = np.apply_along_axis(_differentiation_1_distance, axis, X_sorted)
    X_agg  = np.sum(np.apply_along_axis(lambda a: tnorm(a, measure), axis, X_differenced), axis=axis)

    if keepdims:
        X_agg = np.expand_dims(X_agg, axis=axis)

    return X_agg


def choquet_integral_symmetric_cf12(X, measure=None, axis=0, f1=np.minimum.reduce, f2=np.minimum.reduce, keepdims=False):
    '''
    Aggregates data using a generalization of the Choquet integral.
    
    All hail Giancarlo.
    
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])
    X_sorted = np.sort(X, axis = axis)
    F_1 = lambda a, b: f1(a[1:],b[1:])
    F_2 = lambda a, b: f2(a[0:-1],b[1:])
    F12 = lambda a, b: np.sum(np.append(f1(a[0], b[0]), F_1(a, b) - F_2(a, b)))

    X_agg = np.apply_along_axis(F12, axis, X_sorted, measure)

    if keepdims:
        X_agg = np.expand_dims(X_agg, axis=axis)

    return X_agg


def sugeno_fuzzy_integral(X, measure=None, axis = 0, keepdims=True):
    '''
    Aggregates data using a generalization of the Choquet integral.
    
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])

    return sugeno_fuzzy_integral_generalized(X, measure, axis, np.minimum, np.amax, keepdims)


def sugeno_fuzzy_integral_generalized(X, measure, axis = 0, f1 = np.minimum, f2 = np.amax, keepdims=True):
    '''
    Aggregates data using a generalization of the Choquet integral.
    
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    X_sorted = np.sort(X, axis = axis)
    return f2(f1(np.take(X_sorted, np.arange(0, X_sorted.shape[axis]), axis), measure), axis=axis, keepdims=keepdims)