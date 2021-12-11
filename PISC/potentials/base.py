import numpy as np

"""
This is the base class for all PES definitions. Each PES
defined in this directory needs to have a dpot function and 
a ddpot function, and provision to convert from cartesian to 
Matsubara coordinates. 
"""

class PES(object):
	def __init__():
		self.pot = None	
		self.dpot_cart = None
		self.dpot = None
		self.ddpot_cart = None
		self.ddpot = None

	def bind(self,ens,rp):
		self.ens = ens
		self.rp = rp
		self.ndim = ens.ndim
		self.nsys = rp.nsys
		self.nmtrans = rp.nmtrans

		self.pot = np.zeros((self.nsys, self.rp.nbeads))	
		self.dpot_cart = np.zeros_like(rp.qcart)
		self.dpot = np.zeros_like(rp.q)
		self.ddpot_cart = np.zeros_like(rp.ddpot_cart)
		self.ddpot = np.zeros_like(rp.ddpot)
   
		self.ddpotmat = np.zeros((self.nsys,self.ndim,self.ndim,self.rp.nmodes,self.rp.nmodes))
			
		for d in range(self.ndim):
			self.ddpotmat[:,d,d] = np.eye(self.rp.nmodes,self.rp.nmodes)
		
	def compute_potential(self):
		raise NotImplementedError("Force function not defined")

	def compute_force(self):
		raise NotImplementedError("Force function not defined")

	def compute_hessian(self):
		raise NotImplementedError("Hessian function not defined")

	def update(self):
		self.compute_potential()
		self.compute_force()
		self.compute_hessian()
		self.dpot = self.nmtrans.cart2mats(self.dpot_cart)
		self.ddpot = self.nmtrans.cart2mats_hessian(self.ddpot_cart)

		# pes update needs to change for MF Matsubara and Matsubara	
