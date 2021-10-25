"""
This module contains the code for implementing second 
and fourth order symplectic integrator.
"""

import numpy as np
### Work left to do:
# 1. Declare the order of the integrator and splitting inside the Integrator class object.  
# 2. Figure out how to do 4th order integration with the ring polymer springs.
# 3. Work out how to convert the Hessian to normal mode coordinates - straightforward
# 4. Make a list of possible places to optimize. 
# 5. Find ways to probe the accuracy of this integrator.
# 7. Include a separate spring force update step later, if required

class Integrator(object):
	def bind(self, ens, motion, rp, pes, prng, therm):
		# Declare variables that contain all details from the various classes in bind 
		self.motion = motion
		self.rp = rp
		self.pes = pes	
		self.ens = ens
		self.therm = therm
	 	self.therm.bind(self, ens, motion, prng, rp)
	
	def force_update(self):
		self.rp.mats2cart(self.rp.q,self.rp.qcart)
		self.pes.update()
	
class Symplectic_order_II(Integrator):	
	def O(self,pmats):
		self.therm.thalfstep(pmats)	
	
	def A(self):
		self.rp.q+=self.rp.p*self.qdt/self.rp.dynm3
		self.force_update()	

	def B(self):
		self.rp.p-=self.pes.dpot*self.pdt

	def b(self):
		self.rp.p-=self.rp.dpot*self.pdt

	def M1(self):
		self.Mpp-=(self.ddpot+self.rp.ddpot)*self.pdt*Mqp
	
	def M2(self):
		self.Mpq-=(self.ddpot+self.rp.ddpot)*self.pdt*Mqq

	def M3(self):
		self.Mqp+=self.qdt*Mpp/self.rp.dynm3
		
	def M4(self):
		self.Mqq+=self.qdt*Mpq/self.rp.dynm3

	def pq_step(self):
		self.B()
		self.b()
		self.A()
		self.B()
		self.b()

	def pq_step_RSP(self):
		self.B()
		self.rp.RSP_step()
		self.B()
	
	def Monodromy_step(self):
		self.B()
		self.b()
		self.M1()
		self.M2()
		self.M3()
		self.M4()
		self.A()
		self.B()
		self.b()
		self.M1()
		self.M2()
		self.M3()
		self.M4()	
 
class Symplectic_order_IV(Integrator):
"""
 Mark L. Brewer, Jeremy S. Hulme, and David E. Manolopoulos, 
"Semiclassical dynamics in up to 15 coupled vibrational degrees of freedom", 
 J. Chem. Phys. 106, 4832-4839 (1997)
"""
	def O(self,pmats):
		self.therm.thalfstep(pmats)

	def A(self,k):
		self.rp.q+=self.qdt[k]*self.rp.p/self.rp.dynm3
		self.force_update()
	
	def B(self,k):
		self.rp.p-=self.pes.dpot*self.pdt[k]

	def b(self,k)	
		self.rp.p-=self.rp.dpot*self.pdt[k]

	def M1(self,k):
		self.Mpp-=(self.ddpot+self.rp.ddpot)*self.pdt[k]*Mqp
	
	def M2(self,k):	
		self.Mpq-=(self.ddpot+self.rp.ddpot)*self.pdt[k]*Mqq
		
	def M3(self,k):
		self.Mqp+=self.qdt[k]*Mpp/self.rp.dynm3
		
	def M4(self,k):
		self.Mqq+=self.qdt[k]*Mpq/self.rp.dynm3
	
	def pq_kstep(self,k):
		self.B(k)
		self.b(k)
		self.A(k)

	def Monodromy_kstep(self,k):
		self.B(k)
		self.b(k)
		self.M1(k)
		self.M2(k)
		self.M3(k)
		self.M4(k)
		self.A(k)

	def pq_step(self):	
		for k in range(4):
			pq_kstep(k)

	def Monodromy_step(self):	
		for k in range(4):
			Monodromy_kstep(k)
