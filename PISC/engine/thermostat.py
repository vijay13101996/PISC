"""
This module defines classes for implementing various 
thermostats into the path-integral dynamics code
"""


import numpy as np

# Write the code for the theta-constrained thermostat here at some point. 

class Thermostat(object):
	def bind(self,rp,propa,motion,prng,ens):
		self.rp = rp
		self.propa = propa
		self.motion = motion
		self.prng = prng
		self.ens = ens
		self.dt = self.motion.dt
		
class PILE_L(Thermostat):
	def __init__(self,tau0, pile_lambda):
		self.tau0 = tau0
		self.pile_lamda = pile_lambda		

	def bind(self,rp,propa,motion,prng,ens):
		super(PILE_L,self).bind(rp,propa,motion,prng,ens)
		self.get_propa_coeffs()

	def get_propa_coeffs(self):
		self.friction = 2.0*self.pile_lambda*np.abs(self.rp.dynfreqs)
        self.friction[0] = 1.0/self.tau0
        self.c1 = np.exp(-self.dt*self.friction)
        self.c2 = np.sqrt((1.0 - self.c1**2)/self.ens.beta)

	def thalfstep(self,pmats):
		p = np.reshape(self.rp.p, (-1, self.rp.dim, self.rp.nmodes))
        sm3 = np.reshape(self.rp.sdynm3, p.shape)
        sp = p/sm3
        self.ethermo += 0.5*np.sum(sp**2, axis=(-1,-2))
        sp *= self.c1
        sp += self.c2*self.prng.normal(size=sp.shape)
        if pmats.all():
            p[:] = sp*sm3
        else:
            p[...,self.rp.nmats:] = sp[...,self.rp.nmats]*sm3[...,self.rp.nmats:]
