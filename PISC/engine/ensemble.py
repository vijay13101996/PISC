"""
This is mainly an auxiliary module to help
with defining and testing the ensembles
"""

import numpy as np

# More details to be added when theta constrained ensembles are to be created

class Ensemble(object):
    
    def __init__(self, beta, ndim=3,theta=None):
        self.beta = 1.0*beta
        self.ndim = ndim
        self.theta=theta
        
