import numpy as np
from PISC.potentials.base import PES

class td_dw(PES):
    def __init__(self,lamda0,g,f):
        self.lamda0 = lamda0
        self.g = g
        self.f = f

    def bind(self,ens,rp,pes_fort=False,transf_fort=False):
        super(td_dw,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
    
    def potential(self,q,t=None):
        if(t is None):
            t = self.motion.t
        q = q[:,0,:]
        lamda = self.f(self.lamda0,t)
        #print('t',t)
        return -(lamda**2/4.0)*q**2 + self.g*q**4 + lamda**4/(64*self.g)  

    def dpotential(self,q,t=None):
        if(t is None):
            t = self.motion.t
        lamda = self.f(self.lamda0,t)
        return -(lamda**2/2.0)*q + 4*self.g*q**3

    def ddpotential(self,q,t=None):
        if(t is None):
            t = self.motion.t
        lamda = self.f(self.lamda0,t)
        return -(lamda**2/2.0) + 12*self.g*q**2
    
    def scalar_potential(self,q,t=None):
        if(t is None):
            t = self.motion.t
        lamda = self.f(self.lamda0,t)
        print("lamda: ",lamda)
        return -(lamda**2/4.0)*q**2 + self.g*q**4 + lamda**4/(64*self.g)
