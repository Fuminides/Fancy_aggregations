# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:46:38 2020

@author: javi-
"""
import numpy as np

from . import intervals as _iv

alpha_order = 0.5
beta_order = 0.1

def _d2(x, y, Mp, Mn):
    if x <= y:
        return Mp * (y**2 - x**2)
    else:
        return Mn * (y**2 - x**2)

def md_2(intervalued_logits, Mp=10, Mn=20, deviation_function=_d2, alpha_order=0.5, beta_order=0.1):  
    # sort interval according to alpha and beta
    n = intervalued_logits.shape[0]
    increasing_iv = np.array(sorted(intervalued_logits, key=lambda a: (_iv.k_alpha_operator(a, alpha_order), _iv.k_alpha_operator(a, beta_order))))
    
    min_width = min(intervalued_logits[:,1] - intervalued_logits[:,0])
        # calculate switch point
    k_alphas = [iv[1] * alpha_order + (1-alpha_order)*iv[0] for iv in increasing_iv]
    
    for k in range(n-1, 0, -1):
        sumlesser = 0
        for k_alpha in k_alphas:
            sumlesser = sumlesser + deviation_function(k_alpha, k_alphas[k-1], Mp, Mn)
        if sumlesser <= 0:
            break
        
    # calculate Kalpha(Y) (solve equation)
    skalphas = [k_alpha**2 for k_alpha in k_alphas]
    k_alpha_y = np.sqrt((Mp * sum(skalphas[:k]) + Mn * sum(skalphas[k:]))/(k * Mp + (n - k) * Mn))
    # obtain interval from kalphaY and width
    return np.array([k_alpha_y - alpha_order * min_width, k_alpha_y - alpha_order * min_width + min_width])


def _d5(x, y, Mp, Mn):
    if x <= y:
        return Mp * (y - x) ** 2
    else:
        return Mn * (y**2 - x**2)
    
def md_5(intervalued_logits, Mp=10, Mn=20, deviation_function=_d5, alpha_order=0.5, beta_order=0.1):
    
    # sort interval according to alpha and beta
    n = intervalued_logits.shape[0]
    increasing_iv = sorted(intervalued_logits, key=lambda a: (_iv.k_alpha_operator(a, alpha_order), _iv.k_alpha_operator(a, beta_order)))
    
    min_width = min(intervalued_logits[:,1] - intervalued_logits[:,0])
    # calculate switch point
    k_alphas = [iv[1] * alpha_order + (1-alpha_order)*iv[0] for iv in increasing_iv]
    
    for k in range(n-1, 0, -1):
        sumlesser = 0
        for k_alpha in k_alphas:
            sumlesser = sumlesser + deviation_function(k_alpha, k_alphas[k-1], Mp, Mn)
        if sumlesser <= 0:
            break
        
    # calculate Kalpha(Y) (solve equation)
    a = k * Mp + (n - k) * Mn
    b = -2 * Mp * sum(k_alphas[:k])
    skalphas = [k_alpha**2 for k_alpha in k_alphas]
    c = Mp * sum(skalphas[:k]) - Mn * sum(skalphas[k:])
    if a == 0:
        sol = -c / b
        k_alpha_y = sol
    elif b == 0:
        sol = np.sqrt(-c / a)
        if k_alphas[k-1] <= sol < k_alphas[k]:
            k_alpha_y = sol
        else:
            k_alpha_y = -sol
    else:
        sol = (-b + np.sqrt(b**2 - 4*a*c)) / (2 * a)
        if k_alphas[k-1] <= sol < k_alphas[k]:
            k_alpha_y = sol
        else:
            k_alpha_y = (-b - np.sqrt(b**2 - 4*a*c)) / (2 * a)
    
    return np.array([k_alpha_y - alpha_order * min_width, k_alpha_y - alpha_order * min_width + min_width])

def md_interval_aggregation(X, axis=0, keepdims=False, md=md_5, alpha_order=0.5, beta_order=0.1, Mp=10, Mn=20):
    '''
    Performs the md intervalued aggregation.
    
    :argument X: intervaluated data to aggregate (last dimension = 2)
    :argument axis: axis to reduce.
    :argument keepdims: keep the reduced dimension or not.
    :argument md: moderate dedviation function to aggregate.
    '''
    if X.shape[-1] != 2:
        print('The input data has not intervalued shape. Last dimension must be 2.')
    else:
        if axis != len(X.shape)-2:
            X = np.swapaxes(X, axis, len(X.shape)-2)
        
        
        res = np.zeros(list(X.shape[:-2]) + [1, 2])
            
        #Iterate over all but the last 2 dimensions
        idx = np.ndindex(X.shape[:-2])
        
        for index in idx:
            res[index] = md(X[index], Mp=Mp, Mn=Mn, alpha_order=alpha_order, beta_order=beta_order)
            
        if axis != len(X.shape)-2:
            res = np.swapaxes(res, axis, len(res.shape)-2)
            
        if not keepdims:
            res = np.squeeze(res)
        
        return res