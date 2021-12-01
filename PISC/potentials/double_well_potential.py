import numpy as np
from PISC.potentials.base import PES

class double_well(PES):
	def __init__(self,lamda,g):
		super(double_well).__init__()
		self.lamda = lamda
		self.g = g
		
	def bind(self,ens,rp):
		super(double_well,self).bind(ens,rp)
		#self.ddpot_cart=self.ddpotmat*(-self.lamda**2/2.0)   	
	
	def compute_potential(self):
		self.pot = -(self.lamda**2/4.0)*self.rp.qcart**2 + self.g*self.rp.qcart**4 + self.lamda**4/(64*self.g)
		return self.pot
	
	def compute_force(self):
		self.dpot_cart = -(self.lamda**2/2.0)*self.rp.qcart + 4*self.g*self.rp.qcart**3
		return self.dpot_cart

	def compute_hessian(self):
		self.ddpot_cart = self.ddpotmat*(-self.lamda**2/2.0) + 12*self.g*(self.rp.qcart**2)[:,:,None,:,np.newaxis]*self.ddpotmat
		return self.ddpot_cart

	def ret_force(self,q):
		return -(self.lamda**2/2.0)*q + 4*self.g*q**3

	def ret_pot(self,q):
		return -(self.lamda**2/4.0)*q**2 + self.g*q**4 + self.lamda**4/(64*self.g)
