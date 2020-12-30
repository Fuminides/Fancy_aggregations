# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:30:43 2020

@author: javi-
"""

import numpy as np

import sklearn.linear_model

from . import penalties as pn
from . import binary_parser as bp

# =============================================================================
# MPA ALPHA FORWARDS
# =============================================================================
def mpa_aggregation(logits, agg1, agg2, alpha, keepdims=False):
    n_2 = len(logits)
    n_1, samples, clases = logits[0].shape
    
    res = np.zeros((n_2, samples, clases))
    
    for ix, logit in enumerate(logits):
        res[ix, :, :] = agg1(logit, axis=0, keepdims=False, alpha=alpha[ix])
    
    return agg2(res, axis=0, keepdims=keepdims, alpha=alpha[-1])

def logistic_alpha_forward(X, cost_convex, clf):
    '''
    X shape: (bandas, samples, clases)
    out shape: (samples, clases)
    '''

    reformed_X = np.swapaxes(X, 0, 1)
    reformed_X = reformed_X.reshape((reformed_X.shape[0], reformed_X.shape[1]*reformed_X.shape[2]))

    alphas = clf.predict(reformed_X)
    result = np.zeros((X.shape[1], X.shape[2]))

    for sample in range(X.shape[1]):
        alpha_cost = lambda real, yhat, axis: cost_convex(real, yhat, axis, alphas[sample])

        result[sample] = pn.penalty_aggregation(X[:, sample, :], [bp.parse(x) for x in bp.classic_aggs], axis=0, keepdims=False, cost=alpha_cost)

    return result

def multimodal_alpha_forward(X, cost, cost2, alpha, agg_set=bp.classic_aggs):
    '''
    X shape: list of n arrays (bandas, samples, clases)
    clfs: list of alphas.
    out shape: (samples, clases)
    '''
    david_played_and_it_pleased_the_lord = [bp.parse(x) for x in agg_set]
    agg_phase_1 = lambda X0, alpha, keepdims=False, axis=0: pn.penalty_aggregation(X0, david_played_and_it_pleased_the_lord, axis=axis, keepdims=keepdims, cost=lambda real, yhat, axis: cost(real, yhat, axis, alpha=alpha))
    agg_phase_2 = lambda X0, alpha, keepdims=False, axis=0: pn.penalty_aggregation(X0, david_played_and_it_pleased_the_lord, axis=axis, keepdims=keepdims, cost=lambda real, yhat, axis: cost2(real, yhat, axis, alpha=alpha))
    
    return mpa_aggregation(X, agg_phase_1, agg_phase_2, alpha, keepdims=False)

# =============================================================================
# LEARN ALPHA - VARIABILITY MAX ALGORITHM UNI-MODAL
# =============================================================================
def generate_real_alpha(X, y, aggs, cost, opt=1):
    a = None
    b = None

    for alpha in np.arange(0.01, 1.01, 0.1):
        alpha_cost = lambda X, yhat, axis: cost(X, yhat, axis, alpha)
        pagg = pn.penalty_aggregation(X, [bp.parse(x) for x in aggs], axis=0, keepdims=False, cost=alpha_cost)

        if np.argmax(pagg) == y:
            if a is None:
                a = alpha
            else:
                b = alpha

    if a is None:
        a = 0.5
    if b is None:
        b = 0.5

    d1 = np.abs(a - 0.5)
    d2 = np.abs(b - 0.5)



    if opt == 1:
        if d1 <= d2:
            return a
        else:
            return b

    elif opt == 2:
        return (a + b) / 2
    
def generate_train_data_alpha(logits, labels, aggs=bp.classic_aggs, cost=pn.cost_functions[0], opt=1):
    '''
    Generates and return the alpha targets for a series of data and their labels,
    using the specified aggregations and cost functions in a MPA.
    '''
    bands, samples, classes = logits.shape

    y = np.zeros((samples,))

    for sample in range(samples):
        y[sample] = generate_real_alpha(logits[:,sample,:], labels[sample], aggs, cost, opt=opt)

    return y

def learn_model(X, y, cost, aggs=bp.classic_aggs, opt=1):
    '''
    X shape: list of n arrays (bandas, samples, clases)
    out shape: (samples, clases)

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    X_reshaped = np.swapaxes(X, 0, 1)
    X_reshaped = X_reshaped.reshape((X_reshaped.shape[0], X_reshaped.shape[1]*X_reshaped.shape[2]))
    y_alpha = generate_train_data_alpha(X, y, aggs=aggs, cost=cost, opt=opt)
    clf = sklearn.linear_model.LinearRegression().fit(X_reshaped, y_alpha)
    
    return clf

# =============================================================================
# MULTIMODAL ALPHA OPTIMIZATION - LEAST SQAURES VARAIBLITY + ACC ALGORITHM
# =============================================================================
def eval_alpha(alpha_v, y_hat, y):
    '''
    

    Returns
    -------
    None.

    '''
    alpha_score = np.mean(np.minimum(alpha_v, 1 - alpha_v))
    acc_score = np.mean(np.equal(y_hat, y))
    
    return (alpha_score + acc_score) / 2


def eval_conf(X, alpha, y, agg1, agg2):
    '''
    Computes the mpa agg for X, and returns the optimization score.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    agg1 : TYPE
        DESCRIPTION.
    agg2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    y_hat = np.argmax(mpa_aggregation(X, agg1, agg2, alpha), axis=1)
    return eval_alpha(alpha, y_hat, y)

def gen_all_good_alpha_mff(X, y, costs, aggs=bp.choquet_family + bp.sugeno_family + bp.overlap, opt=1, four_class=False):
    '''
    Learn the logistic regression for the whole set of datasets.
    '''
    from scipy.optimize import least_squares

    agg_phases = [lambda X0, alpha, keepdims=False, axis=0: pn.penalty_aggregation(X0, aggs, axis=axis, keepdims=keepdims, cost=lambda real, yhat, axis: costs[ix](real, yhat, axis, alpha=alpha)) for ix in range(costs)]
    
    
    optimize_lambda = lambda alpha: -eval_conf(X, alpha, y, agg_phases) #Remember we are minimizng
    x0_alpha = np.array([0.5] * len(X) + [0.5]) #WIP: chek the size of each phase.
    
    res_1 = least_squares(optimize_lambda, x0_alpha, bounds=[0.0001, 0.9999])
    

    return res_1.x