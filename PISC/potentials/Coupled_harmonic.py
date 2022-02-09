import numpy as np
from PISC.potentials.base import PES

class coupled_harmonic(PES):
		def __init__(self,omega,g0):
			super(coupled_harmonic).__init__()
			self.omega = omega
			self.g0 = g0
				
		def bind(self,ens,rp):
			super(coupled_harmonic,self).bind(ens,rp)
 
		def potential(self,q):
			x = q[:,0] 
			y = q[:,1]
			return self.potential_xy(x,y) 

		def dpotential(self,q):
			x = q[:,0] 
			y = q[:,1]		 
			return np.transpose([(self.omega**2/2.0)*x + 2*self.g0*x*y**2,(self.omega**2/2.0)*y + 2*self.g0*y*x**2],axes=[1,0,2])
			#return np.transpose([(self.omega**2/2.0)*x + self.g0*y,(self.omega**2/2.0)*y + self.g0*x],axes=[1,0,2])

		def ddpotential(self,q):
			x = q[:,0]
			y = q[:,1]
			return np.transpose([[(self.omega**2/2.0) + 2*self.g0*y**2, 4*self.g0*x*y],[4*self.g0*x*y, (self.omega**2/2.0) + 2*self.g0*x**2]],axes=[2,0,1,3]) 
			#return np.transpose([[(self.omega**2/2.0)*np.ones_like(x), self.g0*np.ones_like(x)],[self.g0*np.ones_like(x), \
			#		(self.omega**2/2.0)*np.ones_like(x)]],axes=[2,0,1,3]) 

		def potential_xy(self,x,y):
			return np.array((self.omega**2/4.0)*(x**2+y**2) + self.g0*x**2*y**2)
			#return np.array((self.omega**2/4.0)*(x**2+y**2) + self.g0*x*y) 
