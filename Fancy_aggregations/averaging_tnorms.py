# -*- coding: utf-8 -*-
"""
File containing the "averaging t-norms".


@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np

def _my_is_close(a, b):
  abs_dif = np.abs(a - b)
  return np.less(abs_dif, 0.05)
  
def averaging_operator_min_delta_1d(x, tnorm=np.min, purge_0=True):
	if purge_0:
		x = np.array([y for y in x if not math.isclose(y, 0, abs_tol = 10e-10)])

		if len(x) == 0:
				return 0.0

	min_x = np.min(x)  
 
	x_1_n = x[~_my_is_close(x, min_x)]

	if len(x_1_n) == 0:
		t_norm_res = 0.0
	elif len(x_1_n) == 1:
		t_norm_res = x_1_n - min_x
	else:
		t_norm_res = tnorm(x_1_n - min_x)

	return min_x + t_norm_res

def averaging_operator_min_delta(x, axis, keepdims, tnorm=np.min, purge_0=True):
	res = np.apply_along_axis(averaging_operator_min_delta_1d, axis, x, tnorm=tnorm, purge_0=purge_0)

	if keepdims:
		res = np.expand_dims(res, axis=axis)

	return res

def averaging_operator_max_delta_1d(x, tnorm):
	
	arg_max_x = np.argmax(x)  
	max_x = x[arg_max_x]
	x_1_n = x[np.arange(len(x))!=arg_max_x]

	return max_x - tnorm(x_1_n)