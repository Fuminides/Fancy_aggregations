# -*- coding: utf-8 -*-
"""
File containing different OWA operators.

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import numpy as np
# =============================================================================
#   ~OWAs
# =============================================================================


def owa(X, weights):
    '''

    :param X:
    :param weights:
    :return:
    '''
    X_sorted = np.sort(X) #Order decreciente

    return np.sum(X_sorted * weights[::-1])