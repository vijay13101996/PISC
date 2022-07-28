import numpy as np
from PISC.potentials.base import PES

class morse(PES):
	def __init__(self,D,alpha):
		super(morse).__init__()
		self.alpha = alpha
		self.D = D
		
	def bind(self,ens,rp):
		super(morse,self).bind(ens,rp)
	
	def potential(self,q):
		eaq = np.exp(-self.alpha*q)
		return self.D*(1-eaq)**2	
	
	def dpotential(self,q):
		return 2*self.D*self.alpha*(1 - np.exp(-self.alpha*q))*np.exp(-self.alpha*q)

	def ddpotential(self,q):
		return self.alpha**2*(-2*self.D*(1 - np.exp(-self.alpha*q))*np.exp(-self.alpha*q) + 2*self.D*np.exp(-2*self.alpha*q))

