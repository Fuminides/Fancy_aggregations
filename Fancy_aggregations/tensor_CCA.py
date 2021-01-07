import torch
import numpy as np

import torch.nn.functional as F

def _differentiation_1_distance(X):
    #Perform differentiation for each consecuent point in the X dataset (time series)
    #Only for axis=0
    X = X.permute(2, 1, 0)
    aux =  X - F.pad(X, (1, 0))[:, :, :-1]
    return aux.permute(2,1,0)

def diff(X):
    '''
    Only works for two shapes:
    shape 1: experts, samples, classes
    shape 2: experts1, experts2, samples, classes
    '''
    x_len = len(X.shape)
    if x_len == 3:
        return _differentiation_1_distance(X)
    elif x_len == 4:
        exp1, exp2, samples, clases = X.shape
        X = X.reshape((exp1, exp2 * samples, clases))
        aux = _differentiation_1_distance(X)
        return aux.reshape((exp1, exp2, samples, clases))
# =============================================================================
# TNORMS
# =============================================================================
def hamacher_product(x, y):
    return x*y / (x + y - x*y + 0.00000001) 

# =============================================================================
# TCNORMS
# =============================================================================
def torch_max(x, axis=0, keepdims=False):
    v, i = torch.max(x, dim=axis, keepdims=False)
    
    return v
# =============================================================================
# INTEGRALS
# =============================================================================
def torch_mean(x, axis=0, keepdims=False):
    v = torch.mean(x, dim=axis, keepdims=False)
    
    return v
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
def torch_sugeno(X, measure=None, axis = 0, f1 = torch.minimum, f2 = torch.amax, keepdims=False):
    '''
    Aggregates data using a generalization of the Choquet integral.
    
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])
        new_shape = [1] * len(X.shape)
        new_shape[axis] = len(measure)
        measure = torch.reshape(measure, new_shape)

    X_sorted, indices = torch.sort(X, dim=axis)
    return f2(f1(X_sorted, measure), axis=axis, keepdims=keepdims)

def torch_choquet(X, measure=None, axis=0, keepdims=True):
    '''
    Aggregates a numpy array alongise an axis using the choquet integral.
    
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis]) #Uses an implementation trick not valid for generallizations
        measure_twin = torch.cat((measure[1:], torch.tensor([0])))
        measure = measure - measure_twin
        new_shape = [1] * len(X.shape)
        new_shape[axis] = len(measure)
        measure = torch.reshape(measure, new_shape)


    X_sorted, indices = torch.sort(X, axis = axis)

    X_agg = torch.sum(X_sorted * measure, dim=axis, keepdims=keepdims)

    return X_agg

def torch_CF(X, measure=None, axis=0, tnorm=hamacher_product, keepdims=False):
    '''
    Aggregates a numpy array alongise an axis using the choquet integral.

    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])
        new_shape = [1] * len(X.shape)
        new_shape[axis] = len(measure)
        measure = torch.reshape(measure, new_shape)

    X_sorted, indices = torch.sort(X, axis = axis)

    assert axis == 0 #Not implemented for other axis
    X_differenced = diff(X_sorted)
    X_agg  = torch.sum(tnorm(X_differenced, measure), dim=axis, keepdims=keepdims)

    return X_agg


def torch_CF1F2(X, measure=None, axis=0, f1=torch.minimum, f2=torch.minimum, keepdims=False):
    '''
    Aggregates data using a generalization of the Choquet integral.
       
    :param X: Data to aggregate.
    :param measure: Vector containing the measure numeric values (Symmetric!)
    :param axis: Axis alongside to aggregate.
    '''
    if measure is None:
        measure = generate_cardinality(X.shape[axis])
        new_shape = [1] * len(X.shape)
        new_shape[axis] = len(measure)
        measure = torch.reshape(measure, new_shape)
        
    X1_sorted, indices = torch.sort(X, axis = axis)
    X2 = diff(X1_sorted)
    X2_sorted = X1_sorted - X2
    
    
    F_1 = f1(X1_sorted, measure)
    F_2 = f2(X2_sorted, measure)
    

    X_agg = torch.sum(F_1 - F_2, dim=axis, keepdims=keepdims)

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

        self.myparameters = torch.nn.Parameter(self.alpha)

        self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        #HARDCODED FORWARD
        #Phase 1
        c1 = self.agg1(x, axis=0, keepdims=False)
        c2 = self.agg2(x, axis=0, keepdims=False)

        c_f = c1 * self.alpha + c2 * (1 - self.alpha)

        logits = self.softmax(c_f)
        
        return logits

class CCA_multimodal(torch.nn.Module):
  def __init__(self, alfa_shape_s1, s1_agg1, s1_agg2, s2_agg1, s2_agg2):
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

        self.alpha1 = torch.nn.Parameter(torch.rand(alfa_shape_s1), requires_grad=True)
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))

        self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
        """
        x shape should be:
        n_classifiers1 x n_classifiers2 x samples x clases
        """

        #HARDCODED FORWARD
        #Phase 1
        c1 = self.s1_agg1(x, axis=0 , keepdims=False)
        c2 = self.s1_agg2(x, axis=0 , keepdims=False)

        c_f = c1 * self.alpha1 + c2 * (1 - self.alpha1)

        c_f1 = self.s2_agg1(c_f, axis=0 , keepdims=False)
        c_f2 = self.s2_agg2(c_f, axis=0 , keepdims=False)

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