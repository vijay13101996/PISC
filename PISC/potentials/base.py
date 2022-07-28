import numpy as np
import PISC
from PISC.utils.misc import pairwise_swap

"""
This is the base class for all PES definitions. Each PES
defined in this directory needs to have a dpot function and 
a ddpot function, and provision to convert from cartesian to 
Matsubara coordinates. 
"""

# Store all the Matsubara potentials with a 'Matsubara' tag 

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
			
		for d1 in range(self.ndim):
			for d2 in range(self.ndim):
				self.ddpotmat[:,d1,d2] = np.eye(self.rp.nmodes,self.rp.nmodes)
		
	def potential(self,q):
		raise NotImplementedError("Potential function not defined")

	def dpotential(self,q):
		raise NotImplementedError("Force function not defined")

	def ddpotential(self,q):
		raise NotImplementedError("Hessian function not defined")
	
	def compute_potential(self):
		self.pot = self.potential(self.rp.qcart)
		return self.potential
	
	def compute_force(self): # Work out a standard way to do this for higher dimensions
		self.dpot_cart = self.dpotential(self.rp.qcart)
		return self.dpot_cart

	def compute_hessian(self):
		if(self.ndim==1):
			self.ddpot_cart = self.ddpotmat*self.ddpotential(self.rp.qcart)[:,:,None,:,np.newaxis]
		else:
			self.ddpot_cart = self.ddpotmat*self.ddpotential(self.rp.qcart)[...,np.newaxis]
		return self.ddpot_cart

	def compute_mats_potential(self):
		self.pot = self.potential(self.rp.mats_beads())
		return self.pot

	def compute_mats_force(self):
		self.dpot_cart = self.dpotential(self.rp.mats_beads())
		return self.dpot_cart

	def compute_mats_hessian(self):
		self.ddpot_cart = self.ddpotmat*self.ddpotential(self.rp.mats_beads())[:,:,None,:,np.newaxis]
		return self.dpot_cart

	def centrifugal_term(self):
		Rsq = np.sum(self.rp.matsfreqs**2*pairwise_swap(self.rp.q[...,:self.rp.nmats],self.rp.nmats)**2,axis=2)[:,None]
		const = self.ens.theta**2/self.rp.m
		dpot = (-const/Rsq**2)*pairwise_swap(self.rp.matsfreqs[...,:self.rp.nmats],self.rp.nmats)**2*self.rp.q[...,:self.rp.nmats]
		return dpot 

	def update(self):
		if(self.rp.mode=='rp'):
			self.compute_potential()
			self.compute_force()
			self.compute_hessian()
			self.dpot = self.nmtrans.cart2mats(self.dpot_cart)
			self.ddpot = self.nmtrans.cart2mats_hessian(self.ddpot_cart)	
		elif(self.rp.mode=='rp/mats'):
			self.compute_mats_potential()
			self.compute_mats_force()
			self.compute_mats_hessian()
			self.dpot[...,:self.rp.nmats] = (self.rp.nbeads)*self.nmtrans.cart2mats(self.dpot_cart)[...,:self.rp.nmats]
			self.ddpot[...,:self.rp.nmats,:self.rp.nmats] = (self.rp.nbeads)*self.nmtrans.cart2mats_hessian(self.ddpot_cart)[...,:self.rp.nmats,:self.rp.nmats]
		elif(self.rp.mode=='mats'):
			self.dpot = self.dpotential()
			self.ddpot = self.ddpotential()
			self.dpot_cart = self.nmtrans.mats2cart(self.dpot)
			self.ddpot_cart = self.nmtrans.mats2cart_hessian(self.ddpot)	
