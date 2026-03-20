import numpy as np
from PISC.potentials.base import PES

class mildly_anharmonic(PES):
	def __init__(self,m,a,b):
		super(mildly_anharmonic).__init__()	
		self.a = a
		self.b = b
		self.m = m
		
	def bind(self,ens,rp):
		super(mildly_anharmonic,self).bind(ens,rp)
		
	def potential(self,q):
		return 0.5*self.m*q**2 + self.a*q**3 + self.b*q**4 
	
	def dpotential(self,q):
		return self.m*q + 3*self.a*q**2 + 4*self.b*q**3

	def ddpotential(self,q):
		return self.m + 6*self.a*q + 12*self.b*q**2 
