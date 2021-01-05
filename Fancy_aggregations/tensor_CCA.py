import torch
import numpy as np

def _differentiation_1_distance(X):
    #Perform differentiation for each consecuent point in the X dataset (time series)
    filter = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1, bias=False)
	kernel = np.array([-1.0, 1.0])
	kernel = torch.from_numpy(kernel).view(1,1,2)
	filter.weight.data = kernel
	filter.weight.requires_grad = False

    return filter(X)

def generate_cardinality(N, p = 2):
    '''
    Generate the cardinality measure for a N-sized vector.
    '''
    return (torch.arange(N, 0, -1)/ N)**p

def generate_cardinality_matrix(N, matrix_shape, p = 2):
    '''
    Generate the cardinality measure for a N-sized vector, and returns it in a matrix shape.
    Use this if you cannot broadcast generate_cardinality() correctly.
    N and matrix_shape must be coherent (matrix_shape[0] == N)
    '''
    res = torch.zeros(matrix_shape)
    dif_elements = [(x/ N)**p for x in torch.arange(N, 0, -1)]

    for ix, elements in enumerate(dif_elements):
        res[ix,...] = dif_elements[ix]

    return res

#ALL TORCH SUGENO IMPL ARE DIRECT TRANSLATIONS FROM THE NUMPY ONES
def torch_sugeno(X, measure=None, axis = 0, f1 = torch.minimum, f2 = torch.amax, keepdims=True):
    '''
    Aggregates data using a generalization of the Choquet integral.
    
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])

    X_sorted, indices = torch.sort(X, axis = axis)
    return f2(f1(torch.take(X_sorted, torch.arange(0, X_sorted.shape[axis]), axis), measure), axis=axis, keepdims=keepdims)

def choquet_integral_symmetric(X, measure=None, axis=0, keepdims=True):
    '''
    Aggregates a numpy array alongise an axis using the choquet integral.
    
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis]) #Uses an implementation trick not valid for generallizations
        measure_twin = torch.cat(measure[1:], torch.tensor(0))
        measure_diff = measure - measure_twin

    X_sorted = torch.sort(X, axis = axis)

    
    X_agg = torch.sum(X * measure, dim=axis, keepdims=keepdims)

    return X_agg

def choquet_integral_CF(X, measure=None, axis=0, tnorm=tnorms.hamacher_tnorm, keepdims=True):
    '''
    Aggregates a numpy array alongise an axis using the choquet integral.

    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])

    X_sorted = np.sort(X, axis = axis)

    X_differenced = np.apply_along_axis(_differentiation_1_distance, axis, X_sorted)
    X_agg  = np.sum(np.apply_along_axis(lambda a: tnorm(a, measure), axis, X_differenced), axis=axis)

    if keepdims:
        X_agg = np.expand_dims(X_agg, axis=axis)

    return X_agg


def choquet_integral_symmetric_cf12(X, measure=None, axis=0, f1=np.minimum.reduce, f2=np.minimum.reduce, keepdims=False):
    '''
    Aggregates data using a generalization of the Choquet integral.
    
    All hail Giancarlo.
    
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])
    X_sorted = np.sort(X, axis = axis)
    F_1 = lambda a, b: f1(a[1:],b[1:])
    F_2 = lambda a, b: f2(a[0:-1],b[1:])
    F12 = lambda a, b: np.sum(np.append(f1(a[0], b[0]), F_1(a, b) - F_2(a, b)))

    X_agg = np.apply_along_axis(F12, axis, X_sorted, measure)

    if keepdims:
        X_agg = np.expand_dims(X_agg, axis=axis)

    return X_agg

class CCA_unimodal(torch.nn.Module):
  def __init__(self, agg1, agg2):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CCA_unimodal, self).__init__()

        #HARDCODED AGGS
        self.agg1 = agg1
        self.agg2 = agg2

        self.alpha = torch.tensor(0.5, requires_grad=True)

        self.myparameters = nn.Parameter(self.alpha)

        self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        #HARDCODED FORWARD
        #Phase 1
        c1 = self.agg1(x, 0 , False)
        c2 = self.agg2(x, 0 , False)

        c_f = c1 * self.alpha + c2 * (1 - self.alpha)

        logits = self.softmax(c_f)
        
        return logits

class CCA_multimodal(torch.nn.Module):
  def __init__(self, alfa_shape_s1, s1_agg1, s2_agg2, s2_agg1, s2_agg2):
        """
        alfa_shape_1 should be n_classifiers2

        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CCA_multimodal, self).__init__()

        #HARDCODED AGGS
        self.s1_agg1 = s1_agg1
        self.s1_agg2 = s1_agg2

        self.s2_agg1 = s2_agg1
        self.s2_agg2 = s2_agg2

        self.alpha1 = torch.nn.Parameter(torch.tensor(torch.from_numpy(np.ones(alfa_shape_s1)* 0.5), requires_grad=True))
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))

        self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
        """
        x shape should be:
        n_classifiers1 x n_classifiers2 x samples x clases
        """

        #HARDCODED FORWARD
        #Phase 1
        c1 = self.s1_agg1(x, 0 , False)
        c2 = self.s1_agg2(x, 0 , False)

        c_f = c1 * self.alpha1 + c2 * (1 - self.alpha1)

        c_f1 = self.s2_agg1(c_f, 0 , False)
        c_f2 = self.s2_agg2(c_f, 0 , False)

        c_f2 = c_f1 * self.alpha2 + c_f2 * (1 - self.alpha2)

        logits = self.softmax(c_f2)
        
        return logits

#Helpers. 
def ready_CCA_unimodal(x, ag1, ag2):
	clasi, samples, clases = x.shape
	net_ag = CCA_unimodal(ag1, ag2)

	return net_ag

def ready_CCA_multimodal(x, ag1, ag2, ag3, ag4):
	clasi1, clasi2, samples, clases = x.shape
	net_ag = CCA_multimodal(clasi2, ag1, ag2, ag3, ag4)

	return net_ag