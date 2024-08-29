import numpy as np 
from PISC.potentials.base import PES 
 
class mildly_anharmonic(PES): 
    def __init__(self,m,a,b,w=1.0,n=4): 
        super(mildly_anharmonic).__init__()  
        self.a = a 
        self.b = b 
        self.m = m
        self.w = w
        self.n = n
         
    def bind(self,ens,rp): 
        super(mildly_anharmonic,self).bind(ens,rp) 
         
    def potential(self,q): 
        return 0.5*self.m*self.w**2*q**2 + self.a*q**3 + self.b*q**self.n 
     
    def dpotential(self,q): 
        return self.m*self.w**2*q + 3*self.a*q**2 + self.n*self.b*q**(self.n-1)
 
    def ddpotential(self,q):
        return self.m*self.w**2 + 6*self.a*q + self.n*(self.n-1)*self.b*q**(self.n-2)
