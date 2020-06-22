# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:37:31 2020

@author: javi-
"""
import numpy as np

from . import intervals as _iv
from . import owas as _owa

def iowa(X,axis=0, keepdims=False, w=None, alpha_order=0.5, beta_order=0.1):
     if X.shape[-1] != 2:
        print('The input data has not intervalued shape. Last dimension must be 2.')
     else:
        if axis != len(X.shape)-2:
            X = np.swapaxes(X, axis, len(X.shape)-2)
            
            res = np.zeros(list(X.shape[:-2]) + [1, 2])
            
        #Iterate over all but the last 2 dimensions
        idx = np.ndindex(X.shape[:-2])
        
        for index in idx:
            ordered_X = sorted(X[index], key=lambda a: (_iv.k_alpha_operator(a, alpha_order), _iv.k_alpha_operator(a, beta_order)))
            res[index] = np.sum(np.multiply(ordered_X, w.reshape(len(w),1)), axis=0)
            
        if axis != len(X.shape)-2:
            res = np.swapaxes(res, axis, len(res.shape)-2)
            
        if not keepdims:
            res = np.squeeze(res)
        
        return res
    
def std_iowa(X, axis=0, keepdims=False, alpha_order=0.5, beta_order=0.1, a=0.1, b=0.5):
    w = _owa.generate_owa_weights(X.shape[axis], lambda x: _owa.std_quantifier(x, a=a, b=b))
    return iowa(X, axis=axis, keepdims=keepdims, alpha_order=alpha_order, beta_order=beta_order, w=w)

def iowa1(X, axis=0, keepdims=False, alpha_order=0.5, beta_order=0.1):
    return std_iowa(X, axis, keepdims, alpha_order, beta_order, 0.1, 0.5)

def iowa2(X, axis=0, keepdims=False, alpha_order=0.5, beta_order=0.1):
    return std_iowa(X, axis, keepdims, alpha_order, beta_order, 0.5, 1)

def iowa3(X, axis=0, keepdims=False, alpha_order=0.5, beta_order=0.1):
    return std_iowa(X, axis, keepdims, alpha_order, beta_order, 0.3, 0.8)