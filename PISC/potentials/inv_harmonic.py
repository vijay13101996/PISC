import numpy as np
from PISC.potentials.base import PES

class InvHarmonic(PES):
    """
    Inverse harmonic oscillator potential. 
    """
    def __init__(self, m, omega):
        self.m = m
        self.omega = omega

    def bind(self,ens,rp,pes_fort=False,transf_fort=False):
        super(InvHarmonic,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)

    def potential(self, q):
        q = q[:,0,:]
        return -0.5*self.m*self.omega**2*(q**2)

    def dpotential(self, q):
        return -self.m*self.omega**2*q

    def ddpotential(self, q):
        return -self.m*self.omega**2*np.ones_like(q)
