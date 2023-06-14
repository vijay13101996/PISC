import numpy as np
from PISC.potentials.base import PES

## This potential is obtained by modifying (restricting to one bath mode)
## Tanimura's system-bath potential in 
## The Journal of Physical Chemistry A 115.16 (2011): 4009-4022.

class Tanimura_SB(PES):
		def __init__(self,D,alpha,m,mb,wb,VLL,VSL,cb):
			super(Tanimura_SB).__init__()
			self.D = D
			self.alpha = alpha
			self.m = m
			self.mb = mb
			self.wb = wb
			self.VLL = VLL
			self.VSL = VSL
			self.cb = cb
				
		def bind(self,ens,rp):
			super(Tanimura_SB,self).bind(ens,rp)
 
		def potential(self,q):
			s = q[:,0] 
			b = q[:,1]
			return self.potential_xy(s,b) 

		def dpotential(self,q):
			s = q[:,0] 
			b = q[:,1]	
			Vs= 2*self.D*self.alpha*(1 - np.exp(-s*self.alpha))*np.exp(-s*self.alpha)\
				- 1.0*self.cb*(b - self.cb*(0.5*s**2*self.VSL + s*self.VLL)/(self.mb*self.wb**2))*(1.0*s*self.VSL + self.VLL)
			Vb= 0.5*self.mb*self.wb**2*(2*b - 2*self.cb*(0.5*s**2*self.VSL + s*self.VLL)/(self.mb*self.wb**2))
			return np.transpose([Vs,Vb],axes=[1,0,2])
	
		def ddpotential(self,q):
			s = q[:,0]
			b = q[:,1]
			Vss= -2*self.D*self.alpha**2*(1 - np.exp(-s*self.alpha))*np.exp(-s*self.alpha) +\
					 2*self.D*self.alpha**2*np.exp(-2*s*self.alpha) - 1.0*self.VSL*self.cb*(b - s*self.cb*(0.5*s*self.VSL + self.VLL)/(self.mb*self.wb**2)) \
						+ 1.0*self.cb**2*(1.0*s*self.VSL + self.VLL)**2/(self.mb*self.wb**2)
			Vsb= -1.0*self.cb*(1.0*s*self.VSL + self.VLL)
			Vbb= 1.0*self.mb*self.wb**2*np.ones_like(s)		
			return np.transpose([[Vss, Vsb],[Vsb, Vbb]],axes=[2,0,1,3]) 
	
		def potential_xy(self,s,b):
			Vs = self.D*(1-np.exp(-self.alpha*s))**2
			Vc = self.VLL*s + 0.5*self.VSL*s**2
			Vsb = 0.5*self.mb*self.wb**2*(b - self.cb*Vc/(self.mb*self.wb**2))**2
			return Vs + Vsb
