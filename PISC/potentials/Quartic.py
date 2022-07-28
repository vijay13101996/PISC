import numpy as np
from PISC.potentials.base import PES

class quartic(PES):
	def __init__(self,a):
		super(quartic).__init__()	
		self.a = a
		
	def bind(self,ens,rp):
		super(quartic,self).bind(ens,rp)
		
	def potential(self,q):
		return 0.25*self.a*q**4 
	
	def dpotential(self,q):
		return self.a*q**3

	def ddpotential(self,q):
		return 3*self.a*q**2
