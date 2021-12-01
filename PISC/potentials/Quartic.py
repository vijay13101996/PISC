import numpy as np
from PISC.potentials.base import PES

class Quartic(PES):
	def __init__(self):
		super(Quartic).__init__()	
		
	def bind(self,ens,rp):
		super(Quartic,self).bind(ens,rp)
		#self.ddpot_cart=self.ddpotmat*(-self.lamda**2/2.0)   	
	
	def compute_potential(self):
		self.pot = 0.25*self.rp.qcart**4 
		return self.pot
	
	def compute_force(self):
		self.dpot_cart = self.rp.qcart**3
		return self.dpot_cart

	def compute_hessian(self):
		self.ddpot_cart = 3*(self.rp.qcart**2)[:,:,None,:,np.newaxis]*self.ddpotmat
		return self.ddpot_cart
