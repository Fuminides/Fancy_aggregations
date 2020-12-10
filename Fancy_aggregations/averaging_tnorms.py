# -*- coding: utf-8 -*-
"""
File containing the averaging t-norms.

Note: unless the contrary is specified, all functions are calculated pairwise between x and y.

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np

def averaging_operator_min_delta_1d(x, tnorm):
	
	arg_min_x = np.argmin(x)  
	min_x = x[arg_min_x]
	x_1_n = x[np.arange(len(x))!=arg_min_x]

	return min_x + tnorm(x_1_n)


def averaging_operator_max_delta_1d(x, tnorm):
	
	arg_max_x = np.argmax(x)  
	max_x = x[arg_max_x]
	x_1_n = x[np.arange(len(x))!=arg_max_x]

	return max_x - tnorm(x_1_n)
