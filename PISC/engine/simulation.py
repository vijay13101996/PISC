"""
This is the main class for defining path-integral
simulations.
"""

import numpy as np

# Remove pmats and set it to pc; it is a redundant piece of code. 
# Create separate simulation classes for MF Matsubara and Matsubara 

class Simulation(object):
	def __init__(self):
		self.t = 0.0
		self.nstep = 0

	def bind(self,ens,motion,rng,rp,pes,propa,therm):
		self.ens = ens
		self.motion = motion
		self.rng = rng
		self.rp = rp
		self.pes = pes
		self.propa = propa
		self.therm = therm
        # Bind the components
		self.rp.bind(self.ens, self.motion, self.rng)
		self.pes.bind(self.ens, self.rp)
		self.pes.update()
		self.therm.bind(self.rp,self.motion,self.rng,self.ens)	
		self.propa.bind(self.ens, self.motion, self.rp,
                        self.pes, self.rng, therm)
		self.dt = self.motion.dt
		self.order = self.motion.order

class RP_Simulation(Simulation):
	# Base class for simulations using ring polymers 
	def __init__(self):
		super(RP_Simulation,self).__init__()
	
	def bind(self,ens,motion,rng,rp,pes,propa,therm):
		super(RP_Simulation,self).bind(ens,motion,rng,rp,pes,propa,therm)
		
	def NVE_pqstep(self):
		self.propa.pq_step()
		self.rp.mats2cart()	
		# q has already been converted at the A update step. 
		# So, p and M needs to be converted

	def NVE_pqstep_RSP(self):
		self.propa.pq_step_RSP()
		self.rp.mats2cart()	
			
	def NVE_Monodromystep(self):
		self.propa.Monodromy_step()
		self.rp.mats2cart()
		
	def NVT_pqstep(self,pc,centmove=True):
		self.propa.O(pc)
		self.propa.pq_step(centmove)
		self.propa.O(pc)
		self.rp.mats2cart()

	def NVT_pqstep_RSP(self,pc,centmove=True):
		self.propa.O(pc)
		self.propa.pq_step_RSP(centmove)
		self.propa.O(pc)
		self.rp.mats2cart()

	def NVT_Monodromystep(self,pc):
		self.propa.O(pc)
		self.propa.Monodromy_step()
		self.propa.O(pc)
		self.rp.mats2cart()

	def NVT_const_theta(self,pc):
		self.propa.O(pc)
		self.propa.b_cent()
		self.propa.Monodromy_step()
		self.propa.b_cent()
		self.propa.O(pc)
		self.rp.mats2cart()

	def step(self, ndt=1, mode="nvt",var='pq',RSP=False,pc=None,centmove=True):
		if mode == "nve":
			if(var=='pq'):
				if RSP is False:
					for i in range(ndt):
						self.NVE_pqstep()
				else:
					for i in range(ndt):
						self.NVE_pqstep_RSP()
			elif(var=='monodromy'):
				for i in range(ndt):
					self.NVE_Monodromystep()
		elif mode == "nvt":
			if(var=='pq'):
				if RSP is False:
					for i in range(ndt):
						self.NVT_pqstep(pc,centmove)
				else:
					for i in range(ndt):
						self.NVT_pqstep_RSP(pc,centmove)
			elif(var=='monodromy'):
				for i in range(ndt):
					self.NVT_Monodromystep(pc)
		elif mode == 'nvt_theta':
			self.NVT_const_theta(pc)
		self.t += ndt*self.dt
		self.nstep += ndt

class Matsubara_Simulation(Simulation):
	def __init__(self):
		super(Matsubara_Simulation,self).__init__()
	
	def bind(self,ens,motion,rng,rp,pes,propa,therm):
		super(Matsubara_Simulation,self).bind(ens,motion,rng,rp,pes,propa,therm)
	
	def NVE_pqstep(self):
		self.propa.pq_step_nosprings()
		self.rp.mats2cart()	
				
	def NVE_Monodromystep(self):
		self.propa.Monodromy_step_nosprings()
		self.rp.mats2cart()
		
	def NVT_pqstep(self,pc,centmove=True):
		self.propa.O(pc)
		self.propa.pq_step_nosprings(centmove)
		self.propa.O(pc)
		self.rp.mats2cart()

	def NVT_Monodromystep(self,pc):
		self.propa.O(pc)
		self.propa.Monodromy_step_nosprings()
		self.propa.O(pc)
		self.rp.mats2cart()

	def step(self, ndt=1, mode="nvt",var='pq',pc=None,centmove=True):
		if mode == "nve":
			if(var=='pq'):
				for i in range(ndt):
					self.NVE_pqstep()				
			elif(var=='monodromy'):
				for i in range(ndt):
					self.NVE_Monodromystep()

		elif mode == "nvt":
			if(var=='pq'):
				for i in range(ndt):
					self.NVT_pqstep(pc,centmove)
			elif(var=='monodromy'):
				for i in range(ndt):
					self.NVT_Monodromystep(pc)
		self.t += ndt*self.dt
		self.nstep += ndt


