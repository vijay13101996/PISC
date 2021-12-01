import numpy as np
from PISC.potentials.base import PES

class harmonic(PES):
	def __init__(self,omega):
		super(harmonic).__init__()
		self.omega = omega
		
	def bind(self,ens,rp):
		super(harmonic,self).bind(ens,rp)
		self.ddpot_cart=self.ddpotmat*(self.rp.m3*self.omega**2)[:,:,None,:,None]
	 	
	def compute_potential(self):
		self.pot = (self.omega**2/2.0)*self.rp.m3*self.rp.qcart**2 
		return self.pot
	
	def compute_force(self):
		self.dpot_cart = (self.rp.m3*self.omega**2)*self.rp.qcart 
		return self.dpot_cart

	def compute_hessian(self):
		return self.ddpot_cart
