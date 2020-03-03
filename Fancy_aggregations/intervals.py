# -*- coding: utf-8 -*-
"""
File containing different functions to work with intervaluate data or generate intervals.

Expression taken from:
A. Jurio, M. Pagola, R. Mesiar, G. Beliakov and H. Bustince, "Image Magnification Using Interval Information," 
in IEEE Transactions on Image Processing, vol. 20, no. 11, pp. 3112-3123, Nov. 2011.
doi: 10.1109/TIP.2011.2158227
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5782984&isnumber=6045652

@author: Javier Fumanal Idocin (UPNA).

To suggest changes or submit new code please use the github page.
"""

import implications as _imp

def intervaluate(x, y, implication_operator=_imp.reinchenbag_implication):
    """Returns a tuple with the interval composed using x, y and an implication operador."""
    return (implication_operator(x, y), implication_operator(x, y) + y)