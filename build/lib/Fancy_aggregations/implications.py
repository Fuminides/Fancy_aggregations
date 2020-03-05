# -*- coding: utf-8 -*-
"""
File containing different implication operators.

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""
def kleese_dienes_implication(x, y):
    '''
    Performs pairwise the kleese_dienes implication.
    '''
    return max(1 - x, y)


def reinchenbag_implication(x, y):
    """Performs pairwise the reinchenbag implication."""
    return 1 - x + x * y


def luka_implication(x, y):
    """Performs pairwise the Lukasiewicz implication."""
    return min(1, 1 - x + y)

def lukasiewicz_implication(x, y):
    """Performs pairwise the Lukasiewicz implication."""
    return luka_implication(x, y)