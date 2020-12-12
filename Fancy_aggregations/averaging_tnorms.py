# -*- coding: utf-8 -*-
"""
File containing the averaging t-norms.

Note: unless the contrary is specified, all functions are calculated pairwise between x and y.

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np

def averaging_operator_min_delta_1d(x, tnorm, purge_0=True):
	
	if purge_0:
		x = np.array([y for y in x if y > 0])

	min_x = np.min(x)  
	
	x_1_n = x[x!=min_x]

	if len(x_1_n) == 0:
		t_norm_res = 0.0
	else:
		t_norm_res = tnorm(x_1_n)

	return min_x + t_norm_res


def averaging_operator_max_delta_1d(x, tnorm):
	
	arg_max_x = np.argmax(x)  
	max_x = x[arg_max_x]
	x_1_n = x[np.arange(len(x))!=arg_max_x]

	return max_x - tnorm(x_1_n)
