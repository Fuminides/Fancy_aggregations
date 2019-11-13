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
def hamacher_tnorm(x, y=None):
    '''
    :return: Hamacher t-norm for a pair-wise or alongside a vector if only one is specified. 
    '''
    if y is None:
        return v_tnorm(x, hamacher_tnorm)

    zero_pairs = np.logical_and(np.equal(x, 0), np.equal(y, 0))
    non_zero_terms = [aux for aux in range(len(x)) if not zero_pairs[aux]]

    try:
        result = np.zeros(x.shape)
    except AttributeError:
        result = np.zeros(len(x))
        x = np.array(x)
        y = np.array(y)

    result[non_zero_terms] = np.divide(np.multiply(x[non_zero_terms], y[non_zero_terms]),(x[non_zero_terms] + y[non_zero_terms] - x[non_zero_terms] * y[non_zero_terms]))

    return result


def lukasiewicz_tnorm(x, y):
    '''
    :return: Lukasiewicz t-norm for a pair-wise or alongside a vector if only one is specified. 
    '''
    return np.maximum(0, x + y - 1)


def drastic_tnorm(x, y):
    '''
    :return: Drastic t-norm for a pair-wise or alongside a vector if only one is specified. 
    '''
    buenos_x = np.multiply(x, np.equal(x, 0))
    buenos_y = np.multiply(y, np.equal(y, 0))
    malos = np.zeros(x.shape)
    return buenos_x + buenos_y + malos


def nilpotent_tnorm(x, y):
    '''
    :return: Idelpotent t-norm for a pair-wise or alongside a vector if only one is specified. 
    '''
    terminos1 = (x + y) > 1
    return np.minimum(x, y) * terminos1


# =============================================================================
# ~ T - CONORMS
# =============================================================================
def complementary_t_c_norm(x, y, tnorm):
    #Returns the tcnorm value for the specified tnorm.
    return 1 - tnorm(1 -x, 1- y)


def probabilistc_sum(x, y):
    return x + y - x * y


def bounded_sum(x, y):
    return complementary_t_c_norm(x, y, lukasiewicz_tnorm)


def drastic_tcnorm(x, y):
    return complementary_t_c_norm(x, y, drastic_tnorm)


def nilpotent_maximum(x, y):
    return complementary_t_c_norm(x, y, nilpotent_tnorm)


def einstein_sum(x, y):
    return complementary_t_c_norm(x, y, hamacher_tnorm)


def v_tnorm(X, tnorm):
    """Calculates the given tnorm alongside the vector X"""
    tam = len(X)
    for ix, elem in enumerate(X):
        if ix == 0:
            acum_norm = tnorm(elem, X[ix+1])
        elif ix < tam - 1:
            acum_norm = tnorm(acum_norm, X[ix+1])

    return acum_norm

def fv_tnorm(tnorm):
    """Returns a vectorized tnorm given a pairwise tnorm."""
    return lambda x: v_tnorm(a, tnorm)