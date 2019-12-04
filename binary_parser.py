# -*- coding: utf-8 -*-
"""

Created on 04/12/2019

@author: Javier Fumanal Idocin
"""
import numpy as np
import integrals
import moderate_deviations
import tnorms

def parse(agg_name, axis = 1, keepdims=True):
    agg_minuscula = agg_name.lower()
    if agg_minuscula == 'mean':
        return lambda a: np.mean(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'median':
        return lambda a: np.median(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'min':
        return lambda a: np.min(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'max':
       return lambda a: np.max(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'md':
        return lambda a: moderate_deviations.md_aggregation(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'sugeno':
        return lambda a: integrals.sugeno_fuzzy_integral(a, integrals.generate_cardinality_matrix(a.shape[axis], a.shape), axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'shamacher':
        return lambda a: integrals.sugeno_fuzzy_integral_generalized(a, integrals.generate_cardinality_matrix(a.shape[axis], a.shape),axis=axis, keepdims=keepdims, f1=tnorms.hamacher_tnorm)
    elif agg_minuscula == 'choquet':
        return lambda a: integrals.choquet_integral_symmetric(a, integrals.generate_cardinality(a.shape[axis]), axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'cf12':
        return lambda a: integrals.choquet_integral_symmetric_cf12(a, integrals.generate_cardinality(a.shape[0]), axis=axis, keepdims=keepdims, f1=np.minimum, f2=np.minimum)
    elif agg_minuscula == 'lucrezia':
        return lambda a: integrals.choquet_integral_symmetric_cf12(a, integrals.generate_cardinality(a.shape[0]), axis=axis, keepdims=keepdims, f1=np.minimum, f2=np.minimum)
    else:
        raise KeyError