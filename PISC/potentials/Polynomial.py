import numpy as np
from PISC.potentials.base import PES

class polynomial(PES):
    def __init__(self, coeff, q0=0.0):
        self.coeff = coeff
        self.q0 = q0

    def bind(self,ens,rp): 
        super(polynomial,self).bind(ens,rp)

    def potential(self, q):
        return np.polyval(self.coeff, q-self.q0)

    def dpotential(self, q):
        return np.polyval(np.polyder(self.coeff), q-self.q0)

    def ddpotential(self, q):
        return np.polyval(np.polyder(self.coeff,2),q-self.q0)
