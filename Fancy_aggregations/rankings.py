# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:56:33 2019

Paper title: Aggregation of individual rankings thorugh fusion functions: critism and optimality analysis
Journal: Transactions on Fuzzy Systems
Authors: H. bustince et al. 
(Under review)

@author: javier.fumanal.idocin
"""
import numpy as np

def fusion_preorder_preferences(rankings):
    '''
    Returns a ranking fusion based on each expert preferences.
    '''
    preferences, experts = rankings.shape
    result = []
    
    #1st preference
    result.append(set(rankings[:, 1]))
    #Define the rest
    for i in range(1, preferences):
        result.append(set(rankings[:, 1]) - set([i for i in result]))
        
    return result

def fusion_preorder_scores(rankings):
    '''
    Returns a ranking fusion based on each expert scores using the R1 function
    in the paper.
    '''
    
    preferences, experts = rankings.shape
    
    return preferences.max(1).argsort()[::-1]

def quantify_ordinal_r1(A):
    '''
    Given a set of preferences for a group of experts, it returns a quantification.
    '''
    m, n = A.shape
    B = np.zeros(n)
    for j in range(n):
        B[j] = m*n -  np.sum([theta_function(A[i,j], A[i,:]) for i in range(m)])
            
    return B
    
    
def quantify_ordinal_r2(A):
    '''
    Given a set of preferences for a group expert, it returns a quantification.
    '''
    m, n = A.shape
    B = np.zeros(n)
    for j in range(n):
        #B[j] = #TODO
        pass
            
    return B

def theta_function(a, A):
    l = np.argmax(a == A)
    return 1 + l
    
        
    
    
    
    