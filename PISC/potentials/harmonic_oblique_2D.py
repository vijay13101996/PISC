import numpy as np
from PISC.potentials.base import PES

class Harmonic_oblique(PES):
		def __init__(self,T,m,omega1,omega2):
			super(Harmonic_oblique).__init__()
			self.T = T
			self.m = m
			self.omega1 = omega1
			self.omega2 = omega2
			self.T11 = T[0,0]
			self.T12 = T[0,1]
			self.T21 = T[1,0]
			self.T22 = T[1,1]
				
		def bind(self,ens,rp):
			super(Harmonic_oblique,self).bind(ens,rp)
 
		def potential(self,q):
			x = q[:,0] 
			y = q[:,1]
			return self.potential_xy(x,y) 

		def dpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			xi	= self.T11*x + self.T12*y
			eta = self.T21*x + self.T22*y
			g1 = self.T11*self.m*self.omega1**2*xi + self.T21*self.m*self.omega2**2*eta		 
			g2 = self.T12*self.m*self.omega1**2*xi + self.T22*self.m*self.omega2**2*eta
			return np.transpose([g1,g2],axes=[1,0,2])

		def ddpotential(self,q):
			x = q[:,0]
			y = q[:,1]
			ones = np.ones_like(x)
			h11 = self.T11**2*self.m*self.omega1**2 + self.T21**2*self.m*self.omega2**2
			h12 = self.T11*self.T12*self.m*self.omega1**2 + self.T21*self.T22*self.m*self.omega2**2
			h22 = self.T12**2*self.m*self.omega1**2 + self.T22**2*self.m*self.omega2**2
			return np.transpose([[h11*ones, h12*ones],[h12*ones, h22*ones]],axes=[2,0,1,3]) 
 
		def potential_xy(self,x,y):
			xi	= self.T11*x + self.T12*y
			eta = self.T21*x + self.T22*y
			return np.array(0.5*self.m*self.omega1**2*xi**2 + 0.5*self.m*self.omega2**2*eta**2)
