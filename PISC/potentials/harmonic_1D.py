import numpy as np
from PISC.potentials.base import PES

class harmonic(PES):
    def __init__(self,m,omega):
        super(harmonic).__init__()
        self.omega = omega
        self.m = m
        
    def bind(self,ens,rp,pes_fort=False,transf_fort=False):
        super(harmonic,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)

    def potential(self,q):
        q = q[:,0,:]
        return (self.m*self.omega**2/2.0)*q**2

    def dpotential(self,q):
        return (self.m*self.omega**2)*q

    def ddpotential(self,q):
        return self.m*self.omega**2*np.ones_like(q)
   
    def potential_func(self,q):
        return (self.m*self.omega**2/2.0)*q**2
