import numpy as np
from PISC.potentials.base import PES

class trunc_harmonic(PES):
	def __init__(self,lamda,g,wall):
		super(trunc_harmonic).__init__()
		self.lamda = lamda
		self.g = g
		self.trunc = self.lamda/np.sqrt(8*self.g)#np.sqrt(4.0*Vb/self.lamda**2) 
		self.wall = wall
		
	def bind(self,ens,rp):
		super(trunc_harmonic,self).bind(ens,rp)
	
	def potential(self,q):
		if(abs(q) <= self.trunc):
			return -(self.lamda**2/4.0)*q**2 + self.g*q**4 + self.lamda**4/(64*self.g)
		if(abs(q) >= self.trunc and abs(q) < self.wall):
			return 0.0
		else:
			return 1e4 

	def dpotential(self,q):
		return -(self.lamda**2/2.0)*q + 4*self.g*q**3

	def ddpotential(self,q):
		return -(self.lamda**2/2.0) + 12*self.g*q**2
