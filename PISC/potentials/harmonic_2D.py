import numpy as np
from PISC.potentials.base import PES

class Harmonic(PES):
        def __init__(self,omega,m=1.0):
            super(Harmonic).__init__()
            self.omega = omega
            self.m = m
                
        def bind(self,ens,rp):
            super(Harmonic,self).bind(ens,rp)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 

        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]       
            return np.transpose([(self.m*self.omega**2)*x ,(self.m*self.omega**2)*y ],axes=[1,0,2])

        def ddpotential(self,q):
            x = q[:,0]
            y = q[:,1]
            return np.transpose([[(self.m*self.omega**2)*np.ones_like(x), np.zeros_like(x)],[np.ones_like(x), (self.m*self.omega**2)*np.ones_like(x)]],axes=[2,0,1,3]) 
 
        def potential_xy(self,x,y):
            return np.array((self.m*self.omega**2/2.0)*(x**2+y**2)) 
