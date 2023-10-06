import numpy as np
from PISC.potentials.base import PES

class asym_double_well(PES):
	def __init__(self,lamda,g,k):
		super(asym_double_well).__init__()
		self.lamda = lamda
		self.g = g
		self.k = k
		
	def bind(self,ens,rp):
		super(asym_double_well,self).bind(ens,rp)
	
	def potential(self,q):
		return -(self.lamda**2/4.0)*q**2 + self.g*q**4 + self.k*q**3 + self.lamda**4/(64*self.g)	

	def dpotential(self,q):
		return -(self.lamda**2/2.0)*q + 3*self.k*q**2 + 4*self.g*q**3

	def ddpotential(self,q):
		return -(self.lamda**2/2.0) + 6*self.k*q + 12*self.g*q**2

	
