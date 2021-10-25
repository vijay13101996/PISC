"""
This is the main class for defining path-integral
simulations.
"""

import numpy as np

class Simulation(object):
	def __init__(self):
		self.t = 0.0
		self.nstep = 0

	def bind(self,ens,motion,prng,rp,pes,propa,therm):
		self.ens = ens
        self.motion = motion
        self.prng = prng
        self.rp = rp
        self.pes = pes
        self.propa = propa
        self.therm = therm
        # Bind the components
        self.rp.bind(self.ens, self.motion, self.prng)
        self.pes.bind(self.ens, self.rp)
        self.pes.update()
        self.propa.bind(self.ens, self.motion, self.rp,
                        self.pes, self.prng, therm)
        self.dt = self.motion.dt
        self.order = self.motion.order

	def NVE_pqstep(self):
		self.propa.pqstep()
		self.rp.mats2cart()	
		# q has already been converted at the A update step. 
		# So, p and M needs to be converted
	
	def NVE_Monodromystep(self):
		self.propa.Monodromy_step()
		self.rp.mats2cart()
		
	def NVT_pqstep(self,pmats):
		self.propa.O(pmats)
		self.propa.pqstep()
		self.propa.O(mats)
		self.rp.mats2cart()

	def NVT_Monodromystep(self,pmats):
		self.propa.O(pmats)
		self.propa.Monodromy_step()
		self.propa.O(pmats)
		self.rp.mats2cart()

	def step(self, ndt=1, mode="nvt",var='pq',pmats=None):

        if mode == "nve":
			if(var=='pq'):
				for i in range(ndt):
					self.NVE_pqstep()
			else:
				for i in range(ndt):
					self.NVE_Monodromystep()
        elif mode == "nvt":
			if(var=='pq'):
				for i in range(ndt):
					self.NVT_pqstep(pmats)
			else:
				for i in range(ndt):
					self.NVT_Monodromystep(pmats)
        self.t += ndt*self.dt
        self.nstep += ndt
