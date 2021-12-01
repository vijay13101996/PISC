import numpy as np
from PISC.potentials.base import PES

class Harmonic(PES):
	def __init__(self,omega):
		self.omega = omega
		
	def bind(self,ens,rp):
		super(Harmonic,self).bind(ens,rp)
		for d in range(self.ndim):
				self.ddpot_cart[:,d,d] = np.eye(self.rp.nmodes,self.rp.nmodes)

		self.ddpot_cart*=(self.rp.m3*self.omega**2)[:,:,None,:,None]	
	
	def compute_potential(self):
		self.pot = 0.5*self.rp.m3*self.omega**2*self.rp.qcart**2
		return self.pot
	
	def compute_force(self):
		self.dpot_cart = self.rp.m3*self.omega**2*self.rp.qcart
		return self.dpot_cart

	def compute_hessian(self):
		return self.ddpot_cart	
