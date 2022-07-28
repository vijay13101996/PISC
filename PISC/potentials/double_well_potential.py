import numpy as np
from PISC.potentials.base import PES

class double_well(PES):
	def __init__(self,lamda,g):
		super(double_well).__init__()
		self.lamda = lamda
		self.g = g
		
	def bind(self,ens,rp):
		super(double_well,self).bind(ens,rp)
	
	def potential(self,q):
		return -(self.lamda**2/4.0)*q**2 + self.g*q**4 + self.lamda**4/(64*self.g)	

	def dpotential(self,q):
		return -(self.lamda**2/2.0)*q + 4*self.g*q**3

	def ddpotential(self,q):
		return -(self.lamda**2/2.0) + 12*self.g*q**2

	
