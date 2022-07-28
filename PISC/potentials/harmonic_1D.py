import numpy as np
from PISC.potentials.base import PES

class harmonic(PES):
	def __init__(self,omega):
		super(harmonic).__init__()
		self.omega = omega
		
	def bind(self,ens,rp):
		super(harmonic,self).bind(ens,rp)
	
	def potential(self,q):
		return  (self.omega**2/2.0)*q**2
	
	def dpotential(self,q):
		return (self.omega**2)*q

	def ddpotential(self,q):
		return np.ones(q.shape)*self.omega**2
