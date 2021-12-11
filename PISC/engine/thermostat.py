"""
This module defines classes for implementing various 
thermostats into the path-integral dynamics code
"""
import numpy as np
import PISC
from PISC.utils.lapack import Gram_Schmidt
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
	
	def thalfstep(self,pmats): #Consider dropping pmats
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

class Constrain_theta(Thermostat):
	def __init__(self):
		print('init')

	def bind(self,rp,motion,rng,ens):
		super(PILE_L,self).bind(rp,motion,rng,ens)	

	def centrifugal_term(self):
		Rsq = np.sum(self.rp.matsfreqs**2*self.rp.q[...,::-1]**2,axis=2)[:,None]
		const = self.ens.theta**2/self.rp.m
		dpot = (-const/Rsq**2)*self.rp.matsfreqs[::-1]**2*self.rp.q  ### Check the force term here. There is some unseemly behaviour
		return dpot 

	def theta_constrained_randomize(self):
		# Scaling factor for Pi_0 
		R = np.sum(self.rp.matsfreqs**2*self.rp.q[...,::-1]**2,axis=2)**0.5 #Take care of frequency ordering

		# First column of the unitary transformation matrix
		T0 = self.rp.matsfreqs*self.rp.q[...,::-1]/R   

		M = self.rp.nmats
		
		T = rng.rand(self.rp.nsys,M,M)
		T[...,0] = T0[:,0,:]

		# Obtain the unitary transformation matrix
		T = Gram_Schmidt(T)

		N = self.rp.nsys
		Pi = self.rng.normal(0,(self.rp.m/self.ens.beta)**0.5,(N,M))
		
		# Setting Pi_0 to theta/R, so that the resultant distribution has the required theta
		Pi[:,0] = self.ens.theta/R[:,0]
	   
		# Solving for the momentum distribution
		self.rp.p[:,0,:] =np.einsum('ijk,ik->ij', T,Pi)
