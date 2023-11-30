import numpy as np

# More details to be added when theta constrained ensembles are to be created

class Ensemble(object):
    """
    This is mainly an auxiliary class to help
    with defining and testing the ensembles
    """

    def __init__(self, beta=1, ndim=2,theta=None,E=None,qlist=None):
        self.beta = 1.0*beta
        self.ndim = ndim
        self.theta=theta
        self.E = E
        self.qlist = qlist

