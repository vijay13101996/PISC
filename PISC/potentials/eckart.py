import numpy as np
from PISC.potentials.base import PES

class eckart(PES):
	def __init__(self,D,k,w):
		super(eckart).__init__()
		self.D = D#1.0
		self.k = np.sqrt(0.16/D)#k#0.4
		self.w = w#0.001
		print('lambda',self.D*self.k**2)
	
	def bind(self,ens,rp):
		super(eckart,self).bind(ens,rp)
	
	def potential(self,q):
		return self.D/(np.cosh(self.k*q))**2 - 10*(1- 1/(1+np.exp(-(q-self.w)*4))) + 10*(1- 1/(1+np.exp(-(q+self.w)*4))) + 10.0 #+ self.w*q**4
		
	def dpotential(self,q):
		return -2*self.D*self.k*np.tanh(self.k*q)/(np.cosh(self.k*q))**2 + 4*self.w*q**3 # + 2*self.w*q

	def ddpotential(self,q):
		return -2*self.D*self.k**2*(np.cosh(2*self.k*q)-2)/(np.cosh(self.k*q))**4 + + 12*self.w*q**2 # + 2*self.w	
