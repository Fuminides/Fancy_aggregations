# -*- coding: utf-8 -*-
"""
File containing different t-norms and t-conorms operators.
Note: unless the contrary is specified, all functions are calculated pairwise between x and y.

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np

# =============================================================================
# ~ T - NORMS
# =============================================================================
def prod(x, y=None, axis=0, keepdims=False):
    if y is None:
        return np.prod(x, axis=axis, keepdims=keepdims)
    else:
        return x * y
    
def hamacher_tnorm(x, y=None, axis=0, keepdims=False):
    '''
    :return: Hamacher t-norm for a pair-wise or alongside a vector if only one is specified. 
    '''
    if y is None:
        return v_tnorm(x, hamacher_tnorm, axis, keepdims)

    zero_pairs = np.logical_and(np.equal(x, 0), np.equal(y, 0))
    x = np.array(x)
    y = np.array(y)

    non_zero_terms = 1 - zero_pairs
    non_zero_terms = non_zero_terms.astype(bool)

    result = np.zeros(x.shape)
    result[non_zero_terms] = np.divide(np.multiply(x[non_zero_terms], y[non_zero_terms]),(x[non_zero_terms] + y[non_zero_terms] - x[non_zero_terms] * y[non_zero_terms]))

    return result


def lukasiewicz_tnorm(x, y=None, axis=0, keepdims=False):
    '''
    :return: Lukasiewicz t-norm for a pair-wise or alongside a vector if only one is specified. 
    '''
    if y is None:
        return v_tnorm(x, lukasiewicz_tnorm, axis, keepdims)
    return np.maximum(0, x + y - 1)

def luka_tnorm(x, y=None, axis=0, keepdims=False):
    '''
    :return: Lukasiewicz t-norm for a pair-wise or alongside a vector if only one is specified.
    '''
    return lukasiewicz_tnorm(x, y, axis, keepdims)


def drastic_tnorm(x, y=None, axis=0, keepdims=False):
    '''
    :return: Drastic t-norm for a pair-wise or alongside a vector if only one is specified. 
    '''
    if y is None:
        return v_tnorm(x, drastic_tnorm, axis, keepdims)
    buenos_x = np.multiply(x, np.equal(x, 0))
    buenos_y = np.multiply(y, np.equal(y, 0))
    malos = np.zeros(x.shape)
    return buenos_x + buenos_y + malos


def nilpotent_tnorm(x, y=None, axis=0, keepdims=False):
    '''
    :return: Idelpotent t-norm for a pair-wise or alongside a vector if only one is specified. 
    '''
    if y is None:
        return v_tnorm(x, nilpotent_tnorm, axis, keepdims)
    terminos1 = (x + y) > 1
    return np.minimum(x, y) * terminos1

# =============================================================================
# ~ T - CONORMS
# =============================================================================
def complementary_t_c_norm(x, y=None, tnorm=luka_tnorm, axis=0, keepdims=False):
    #Returns the tcnorm value for the specified tnorm.
    if y is None:
        tconorma = lambda x, y: 1 - tnorm(1 - x, 1 - y)
        return v_tnorm(x, tconorma, axis, keepdims)

    return 1 - tnorm(1 - x, 1 - y)

def probabilistc_sum(x, y=None, axis=0, keepdims=False):
    if y is None:
        return v_tnorm(x, probabilistc_sum, axis, keepdims)

    return x + y - x * y

def bounded_sum(x, y=None, axis=0, keepdims=False):
    return complementary_t_c_norm(x, y, lukasiewicz_tnorm, axis, keepdims)

def drastic_tcnorm(x, y=None, axis=0, keepdims=False):
    return complementary_t_c_norm(x, y, drastic_tnorm, axis, keepdims)

def nilpotent_maximum(x, y=None, axis=0, keepdims=False):
    return complementary_t_c_norm(x, y, nilpotent_tnorm, axis, keepdims)

def einstein_sum(x, y=None, axis=0, keepdims=False):
    return complementary_t_c_norm(x, y, hamacher_tnorm, axis, keepdims)

def v_tnorm(x, tnorm=None, axis=0, keepdims=False):
    """Calculates the given tnorm alongside the vector X"""
    def tnorm_1D(X, tnorm):
        tam = len(X)
        for ix, elem in enumerate(X):
            if ix == 0:
                acum_norm = tnorm(elem, X[ix+1])
            elif ix < tam - 1:
                acum_norm = tnorm(acum_norm, X[ix+1])

        return acum_norm
    original_shape = x.shape

    if original_shape[axis] == 1:
        if not keepdims:
            return np.mean(x, axis=axis) #We must delete that dimension
        else:
            return x
            
    res = np.apply_along_axis(tnorm_1D, axis, x, tnorm)

    if keepdims:
        res =  res= np.expand_dims(res, axis=axis)


    return res

def fv_tnorm(tnorm, axis0=0):
    """Returns a vectorized tnorm given a pairwise tnorm."""
    return lambda x, axis=axis0: v_tnorm(x, tnorm, axis=axis)