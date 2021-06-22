#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:58:57 2021

@author: javier
"""

import numpy as np
def abs_v(x, y):
    return np.abs(x - y)

def quadratic(x, y):
    return (x-y) * (x-y)

def square(x, y):
    return np.sqrt(abs_v)

def squarewise(x, y):
    return np.abs(np.sqrt(x) - np.sqrt(y))

def abs_square(x, y):
    return np.abs(x*x - y*y)

def root_square(x, y):
    return (np.sqrt(x) - np.sqrt(y))**2