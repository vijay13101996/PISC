import numpy as np
from PISC.potentials.base import PES

class anharmonic(PES):
    def __init__(self,m,omega,k):
        super(anharmonic).__init__()
        self.m = m
        self.omega = omega
        self.k = k
        
    def bind(self,ens,rp):
        super(anharmonic,self).bind(ens,rp)
    
    def potential(self,q):
        return 0.5*q*self.m*self.omega**2*(1 - np.exp(-q*self.k))/self.k

    def dpotential(self,q):
        return 0.5*q*self.m*self.omega**2*np.exp(-q*self.k) + 0.5*self.m*self.omega**2*(1 - np.exp(-q*self.k))/self.k
    
    def ddpotential(self,q):
        return self.m*self.omega**2*(-0.5*q*self.k + 1.0)*np.exp(-q*self.k)

    
