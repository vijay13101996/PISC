import numpy as np
from PISC.potentials.base import PES

class razavy(PES):
	def __init__(self,D,k,xi):
		super(razavy).__init__()
		self.D = D#1.0
		self.k = np.sqrt(0.16/(D*xi))#k
		self.xi = xi
		print('lambda, Vb',self.D*self.k**2, self.xi**2)
	
	def bind(self,ens,rp):
		super(razavy,self).bind(ens,rp)
	
	def potential(self,q):
		return (self.D*(np.cosh(self.k*q)-1) - self.xi)**2 
		
	def dpotential(self,q):
		return -2*self.D*self.k*np.tanh(self.k*q)/(np.cosh(self.k*q))**2 + 4*self.w*q**3 # + 2*self.w*q

	def ddpotential(self,q):
		return -2*self.D*self.k**2*(np.cosh(2*self.k*q)-2)/(np.cosh(self.k*q))**4 + + 12*self.w*q**2 
