"""
This module defines classes for implementing various 
thermostats into the path-integral dynamics code
"""
import numpy as np

# Write the code for the theta-constrained thermostat here at some point. 

class Thermostat(object):
	def bind(self,rp,motion,rng,ens):
		self.rp = rp
		self.motion = motion
		self.rng = rng
		self.ens = ens
		self.dt = self.motion.dt
		
class PILE_L(Thermostat):
	def __init__(self,tau0, pile_lambda):
		self.tau0 = tau0
		self.pile_lambda = pile_lambda		

	def bind(self,rp,motion,rng,ens):
		super(PILE_L,self).bind(rp,motion,rng,ens)
		self.get_propa_coeffs()

	def get_propa_coeffs(self):
		self.friction = 2.0*self.pile_lambda*np.abs(self.rp.dynfreqs)
		self.friction[0] = 1.0/self.tau0
		self.c1 = np.exp(-self.dt*self.friction/2.0)
		self.c2 = np.sqrt(self.rp.nbeads*(1.0 - self.c1**2)/self.ens.beta)

	def thalfstep(self,pmats):
		p = np.reshape(self.rp.p, (-1, self.rp.ndim, self.rp.nmodes))
		sm3 = np.reshape(self.rp.sqdynm3, p.shape)
		sp = p/sm3
		sp *= self.c1
		sp += self.c2*self.rng.normal(size=sp.shape)
		if pmats is not None:
			if pmats.all():
				p[:] = sp*sm3
			else:
				p[...,self.rp.nmats:] = sp[...,self.rp.nmats:]*sm3[...,self.rp.nmats:]
		#print('p',self.rp.p,p.shape)
