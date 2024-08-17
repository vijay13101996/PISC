import numpy as np
from PISC.potentials.base import PES

class quartic(PES):
    def __init__(self,a):
        super(quartic).__init__()   
        self.a = a
        
    def bind(self,ens,rp,pes_fort=False,transf_fort=False):
        super(quartic,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
        
    def potential(self,q):
        q = q[:,0,:]
        return 0.25*self.a*q**4 
    
    def dpotential(self,q):
        return self.a*q**3

    def ddpotential(self,q):
        return 3*self.a*q**2

    def potential_func(self,q):
        return 0.25*self.a*q**4
