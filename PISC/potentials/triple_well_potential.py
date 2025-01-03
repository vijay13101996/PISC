import numpy as np
from PISC.potentials.base import PES

class triple_well(PES):
	def __init__(self,lamda,g):
		super(triple_well).__init__()
		self.lamda = lamda
		self.g = g
		
	def bind(self,ens,rp):
		super(triple_well,self).bind(ens,rp)
	
	def potential(self,q):
		return 0.1875*(self.g*q**3 - self.lamda*q)**2

	def dpotential(self,q):
		return 0.1875*(6*self.g*q**2 - 2*self.lamda)*(self.g*q**3 - self.lamda*q)

	def ddpotential(self,q):
		return 2.25*self.g*q**2*(self.g*q**2 - self.lamda) + 0.375*(3*self.g*q**2 - self.lamda)**2

	
