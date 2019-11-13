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

def moderate_deviation_f(X, D=distance_f2, Mp=1, Mn=1):
    '''
    Expression () in the following paper.

    
    '''
    n = len(X)
    x_sigma = np.sort(X)
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