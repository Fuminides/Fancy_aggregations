# -*- coding: utf-8 -*-
"""
File containing different functions to aggregate data using Moderate Deviations. The expressions have been obtained from the following paper:

A.H. Altalhi, J.I. Forcén, M. Pagola, E. Barrenechea, H. Bustince, Zdenko Takáč,
Moderate deviation and restricted equivalence functions for measuring similarity between data,
Information Sciences,
Volume 501,
2019,
Pages 19-29,
ISSN 0020-0255,
https://doi.org/10.1016/j.ins.2019.05.078.
(http://www.sciencedirect.com/science/article/pii/S0020025519305031)

Please, cite accordingly.

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np

# =============================================================================
# ~ MODERATE DEVIATIONS
# =============================================================================
def custom_distance(x, y, Mp, Mn, R1, R2):
    '''

    :param R1:
    :param R2:
    :return:
    '''
    if x <= y:
        return Mp - Mp*R1(x, y)
    else:
        return Mn*R2(x,y) - Mn

def custom_distance_morphs(x, y, Mp, Mn, F1, F2, T1, T2):
    '''
    TODO, and will probably stay like that for long.
    :param x:
    :param y:
    :param Mp:
    :param Mn:
    :param F1:
    :param F2:
    :param T1:
    :param T2:
    :return:
    '''
    pass

def distance_f1(x, y, Mp, Mn):
    '''

    :return:
    '''
    if x <= y:
        return Mp*(y - x)*(y - x)
    else:
        return Mn*(y*y - x*x)

def distance_f2(x, y, Mp, Mn):
    '''

    :return:
    '''
    if x <= y:
        return Mp*(y - x)
    else:
        return Mn*(y - x)

def cut_point(D, x_sigma, Mp, Mn):
    k = -1

    for ix, element in enumerate(x_sigma):
        if ix < len(x_sigma) - 1:
            con1 = np.sum([D(x_sigma[i], element, Mp, Mn) for i in range(len(x_sigma))]) <= 0
            cond2 = np.sum([D(x_sigma[i], x_sigma[ix + 1], Mp, Mn) for i in range(len(x_sigma))]) >= 0

            if con1 and cond2:
                k = ix
    return k

def moderate_deviation_f(X, D=distance_f2, Mp=1, Mn=1, axis=0):
    '''


    '''
    n = len(X)
    x_sigma = np.sort(X, axis=0)
    k = cut_point(D, x_sigma, Mp, Mn)

    f = (Mp * np.sum(x_sigma[0:k+1]) + Mn*np.sum(x_sigma[k+1:])) / (k*Mp + (n - k)*Mn)

    return f

def moderate_deviation_eq(X, D=distance_f1, Mp=1, Mn=1):
    '''

    '''
    n = len(X)
    x_sigma = np.sort(X)
    k = cut_point(D, x_sigma, Mp ,Mn)

    a = (k+1)*Mp + (n - k-1)*Mn
    b = -2*Mp*np.sum(x_sigma[0:k+1])
    x_sigma_squared = np.power(x_sigma, 2)
    c = Mp*np.sum(x_sigma_squared[0:k+1]) - Mn*np.sum(x_sigma_squared[k+1:])

    sqr_term = np.sqrt(b*b - 4*a*c)
    y1 = (-b + sqr_term) / (2*a)
    y2 = (-b - sqr_term) / (2*a)

    return y1, y2

def md_aggregation(X, axis=0, keepdims=True, md_function=moderate_deviation_f, Mp=1, Mn=10):
    '''
    Designed to use the md functions using the same interface as the rest of the numpy aggregation functions.
    IT ONLY WORKS IN 3 DIMENSIONAL ARRAY (features, samples, classes)
    :param X:
    :param axis:
    :param keepdims:
    :param md_function:
    :return:
    '''
    if axis != 0:
        X = np.transpose(X, (0, axis))

    clasificadores, muestras, clases = X.shape
    if keepdims:
        result = np.zeros([1] +list(X.shape[1:]))
    else:
        result = np.zeros(X.shape[1:])

    for m in range(muestras):
        #print(md_function(X[:, m, 0], Mp=1, Mn=10))
        if keepdims:
            for clase in range(clases):
                result[0, m, clase] = md_function(X[:, m, clase], Mp=1, Mn=10)
        else:
            for clase in range(clases):
                result[m, clase] = md_function(X[:, m, clase], Mp=1, Mn=10)

    if axis != 0:
        X = np.transpose(X, (0, axis))


    return result

def multichannel_md(X, weights, epsilon=0.005, axis=0, keepdims=False):
    '''
    Multichannel moderate-deviation based agg fucntion.

    More details in: 
    Martin Papčo, Iosu Rodríguez-Martínez, Javier Fumanal-Idocin, Abdulrahman H. Altalhi, Humberto Bustince,
	A fusion method for multi-valued data, Information Fusion, Volume 71, 2021, Pages 1-10, ISSN 1566-2535,
	https://doi.org/10.1016/j.inffus.2021.01.001.

    '''
    reduce_size = X.shape[axis]
    aux = weights * X
    return np.sum(aux * (X + epsilon), axis=axis, keepdims=keepdims) / np.sum(reduce_size * weights * epsilon) + np.sum(aux, axis=axis, keepdims=keepdims)


