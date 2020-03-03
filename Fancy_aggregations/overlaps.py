import numpy as np

def geo_mean(X, X2=None, axis=0, keepdims=True):
	'''
	Returns the geometrical mean of a vector, or element wise if two matrixes specified.
	'''
	if X2 is None:
		exponente = 1/X.shape[axis]

		return np.power(np.prod(X, axis=axis, keepdims=keepdims), exponente)

	else:
		producto = X * X2

		return np.power(producto, 1/len(producto))

def sin_overlap(X, axis=0, keepdims=True):
	'''
    Returns the sin overlap.
	'''
	exponent = 1 / (2 * X.shape[axis])
	return np.sin(np.pi/2 * np.power(np.prod(X, axis=axis, keepdims=keepdims), exponent))

def harmonic_mean(X, axis=0, keepdims=True):
	'''
	Returns the harmonic mean. Checks for nan and keep dims. Requires scipy.
	'''
	import scipy.stats

	base_result = scipy.stats.hmean(X, axis=axis)
	pruned = np.nan_to_num(base_result)

	if keepdims:
		pruned = np.expand_dims(pruned, axis=axis)

	return pruned