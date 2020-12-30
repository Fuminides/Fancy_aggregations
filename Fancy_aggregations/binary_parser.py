# -*- coding: utf-8 -*-
"""

Created on 04/12/2019

@author: Javier Fumanal Idocin
"""
import numpy as np

from . import integrals
from . import moderate_deviations
from . import tnorms
from . import owas
from . import overlaps

supported_functions = ['mean', 'median', 'min', 'max', 'sugeno', 'shamacher', 'choquet', 'cfminmin', 'cf12', 'cf', 'owa1', 'owa2', 'owa3', 'geomean', 'sinoverlap', 'hmean',
                'hamacher', 'luka', 'drastic', 'nilpotent', 'probabilistic_sum', 'bounded_sum', 'drastic_tcnorm', 'nilpotent_maximum', 'einstein_sum']


classic_aggs = ['mean', 'median', 'min', 'max']
owa = ['owa1', 'owa2', 'owa3']
fuzzy_integral = ['sugeno', 'choquet']
choquet_family = ['choquet', 'cf', 'cf12', 'cfminmin']
sugeno_family = ['sugeno', 'shamacher', 'fhamacher']
tnorm = ['min', 'prod', 'hamacher', ' luka', 'drastic', 'nilpotent']
tcnorm = ['probabilistic_sum', 'bounded_sum', 'drastic_tcnorm', 'nilpotent_maximum', 'einstein_sum']
overlap = ['geomean', 'sinoverlap', 'hmean']


def parse(agg_name, axis_f = 0, keepdims_f=True):
    agg_minuscula = agg_name.lower()
    if agg_minuscula == 'mean':
        return lambda a, axis=axis_f, keepdims=keepdims_f: np.mean(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'median':
        return lambda a, axis=axis_f, keepdims=keepdims_f: np.median(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'min':
        return lambda a, axis=axis_f, keepdims=keepdims_f: np.min(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'max':
        return lambda a, axis=axis_f, keepdims=keepdims_f: np.max(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'md':
        return lambda a, axis=axis_f, keepdims=keepdims_f: moderate_deviations.md_aggregation(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'sugeno':
        return lambda a, axis=axis_f, keepdims=keepdims_f: integrals.sugeno_fuzzy_integral(a, integrals.generate_cardinality_matrix(a.shape[axis], a.shape), axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'shamacher':
        return lambda a, axis=axis_f, keepdims=keepdims_f: integrals.sugeno_fuzzy_integral_generalized(a, integrals.generate_cardinality_matrix(a.shape[axis], a.shape), axis=axis, keepdims=keepdims, f1 = tnorms.hamacher_tnorm, f2 = np.amax)
    elif agg_minuscula == 'pre_hamacher':
        my_pre = lambda x, y: x * np.abs(2*y - 1)
        return lambda a, axis=axis_f, keepdims=keepdims_f: integrals.sugeno_fuzzy_integral_generalized(a, integrals.generate_cardinality_matrix(a.shape[axis], a.shape), axis=axis, keepdims=keepdims, f1 = my_pre, f2 = np.amax)    
    elif agg_minuscula == 'fhamacher':
        return lambda a, axis=axis_f, keepdims=keepdims_f: integrals.sugeno_fuzzy_integral_generalized(a, integrals.generate_cardinality_matrix(a.shape[axis], a.shape), axis=axis, keepdims=keepdims, f1 = tnorms.prod, f2 = np.sum)
    elif agg_minuscula == 'choquet':
        return lambda a, axis=axis_f, keepdims=keepdims_f: integrals.choquet_integral_symmetric(a, integrals.generate_cardinality(a.shape[axis]), axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'cfminmin':
        return lambda a, axis=axis_f, keepdims=keepdims_f: integrals.choquet_integral_symmetric_cf12(a, integrals.generate_cardinality(a.shape[axis]), axis=axis, keepdims=keepdims, f1=np.minimum, f2=np.minimum)
    elif agg_minuscula == 'cf12':
        return lambda a, axis=axis_f, keepdims=keepdims_f: integrals.choquet_integral_symmetric_cf12(a, integrals.generate_cardinality(a.shape[axis]), axis=axis, keepdims=keepdims, f1=lambda a,b, axis=axis, keepdims=keepdims: np.sqrt(a* b), f2=tnorms.lukasiewicz_tnorm)
    elif agg_minuscula == 'owa1':
        return lambda a, axis=axis_f, keepdims=keepdims_f: owas.OWA1(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'owa2':
        return lambda a, axis=axis_f, keepdims=keepdims_f: owas.OWA2(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'owa3':
        return lambda a, axis=axis_f, keepdims=keepdims_f: owas.OWA3(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'cf':
        return lambda a, axis=axis_f, keepdims=keepdims_f: integrals.choquet_integral_CF(a, integrals.generate_cardinality(a.shape[axis]), axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'geomean':
        return lambda a, axis=axis_f, keepdims=keepdims_f: overlaps.geo_mean(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'sinoverlap':
        return lambda a, axis=axis_f, keepdims=keepdims_f: overlaps.sin_overlap(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'hmean':
        return lambda a, axis=axis_f, keepdims=keepdims_f: overlaps.harmonic_mean(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'hamacher':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.hamacher_tnorm(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'luka':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.lukasiewicz_tnorm(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'drastic':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.drastic_tnorm(a, axis=axis, keepdims=keepdims)           
    elif agg_minuscula == 'nilpotent':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.nilpotent_tnorm(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'probabilistic_sum':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.probabilistc_sum(a, axis=axis, keepdims=keepdims)    
    elif agg_minuscula == 'bounded_sum':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.bounded_sum(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'drastic_tcnorm':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.drastic_tcnorm(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'nilpotent_maximum':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.nilpotent_maximum(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'einstein_sum':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.einstein_sum(a, axis=axis, keepdims=keepdims)
    elif agg_minuscula == 'prod':
        return lambda a, axis=axis_f, keepdims=keepdims_f: tnorms.prod(a, axis=axis, keepdims=keepdims)
    
    elif agg_minuscula == 'mode':
        from scipy.stats import mode
        if keepdims_f:
            print('Warning: mode does not keep dimension')
        return lambda a, axis=axis_f, keepdims=keepdims_f: mode(a, axis=axis)
    else:
        raise KeyError(agg_name)