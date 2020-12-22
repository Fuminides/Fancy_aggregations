# -*- coding: utf-8 -*-
"""
File containing the "averaging t-norms".


@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np

def _v_isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    """
    returns True if a is close in value to b. False otherwise
    :param a: one of the values to be tested
    :param b: the other value to be tested
    :param rel_tol=1e-9: The relative tolerance -- the amount of error
                         allowed, relative to the absolute value of the
                         larger input values.
    :param abs_tol=0.0: The minimum absolute tolerance level -- useful
                        for comparisons to zero.
    NOTES:
    -inf, inf and NaN behave similarly to the IEEE 754 Standard. That
    is, NaN is not close to anything, even itself. inf and -inf are
    only close to themselves.
    The function can be used with any type that supports comparison,
    substratcion and multiplication, including Decimal, Fraction, and
    Complex
    Complex values are compared based on their absolute value.
    See PEP-0485 for a detailed description
    """

    if np.mean(np.equal(a, b)) == 1:  # short-circuit exact equality
        return np.ones(a.shape)

    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError('error tolerances must be non-negative')

    diff = abs(b - a)
    c1 = diff <= abs(rel_tol * b)
    c2 = diff <= abs(rel_tol * a)
    c3 = diff <= abs_tol

    return np.maximum(np.maximum(c1, c2), c3)

def _my_is_close(a, b):
  '''
  Computes isclose() from a naive perspective, and applying the math isclose() but in a vectorized way.
  
  '''   
  a = np.squeeze(a)
  b = np.squeeze(b)

  abs_dif = np.abs(a - b)
  c1 = np.less(abs_dif, 0.05)
  c2 = _v_isclose(a, b)

  return np.maximum(c1, c2)

  
def averaging_operator_min_delta_1d(x, tnorm=np.min, purge_0=True):
    if purge_0:
        x = x[~_my_is_close(x, np.zeros(len(x)))]

        if len(x) == 0:
                return 0.0

    min_x = np.min(x)  
    aux = np.zeros(len(x)) + min_x
    x_1_n = x[~np.array(_my_is_close(x, aux), dtype=bool)]

            
    if len(x_1_n) == 0:
        t_norm_res = 0.0

    elif len(x_1_n) == 1:
        t_norm_res = x_1_n - min_x
    else:
        t_norm_res = tnorm(x_1_n - min_x)

    return min_x + t_norm_res

def averaging_operator_max_delta_1d(x, tnorm, purge_0=True):
    if purge_0:
        x = x[~_my_is_close(x, np.zeros(len(x)))]

        if len(x) == 0:
                return 0.0

    min_x = np.max(x)  
    aux = np.zeros(len(x)) + min_x
    x_1_n = x[~np.array(_my_is_close(x, aux), dtype=bool)]

    if len(x_1_n) == 0:
        t_norm_res = 0.0

    elif len(x_1_n) == 1:
        t_norm_res = x_1_n - min_x
    else:
        t_norm_res = tnorm(x_1_n - min_x)

    return np.max(x) - t_norm_res

def averaging_operator_min_delta(x, axis, keepdims, tnorm=np.min, purge_0=True):
    res = np.apply_along_axis(averaging_operator_min_delta_1d, axis, x, tnorm=tnorm, purge_0=purge_0)

    if keepdims:
        res = np.expand_dims(res, axis=axis)

    return res

def averaging_operator_max_delta(x, axis, keepdims, tnorm=np.min, purge_0=True):
    res = np.apply_along_axis(averaging_operator_max_delta_1d, axis, x, tnorm=tnorm, purge_0=purge_0)

    if keepdims:
        res = np.expand_dims(res, axis=axis)

    return res

