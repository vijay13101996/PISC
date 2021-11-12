"""
This module contains the code for implementing second 
and fourth order symplectic integrator.
"""

import numpy as np
import scipy
import scipy.integrate
from scipy.integrate import odeint, ode
### Work left to do:
# 1. Declare the order of the integrator and splitting inside the Integrator class object.  
# 2. Figure out how to do 4th order integration with the ring polymer springs.
# 3. Work out how to convert the Hessian to normal mode coordinates - straightforward
# 4. Make a list of possible places to optimize. 
# 5. Find ways to probe the accuracy of this integrator.
# 7. Include a separate spring force update step later, if required

class Integrator(object):
	def bind(self, ens, motion, rp, pes, rng, therm):
		# Declare variables that contain all details from the various classes in bind 
		self.motion = motion
		self.rp = rp
		self.pes = pes	
		self.ens = ens
		self.therm = therm
		self.therm.bind(rp,motion,rng,ens)
	
	def force_update(self):
		self.rp.mats2cart()
		self.pes.update()
	
class Symplectic_order_II(Integrator):	
	def O(self,pmats):
		self.therm.thalfstep(pmats)	
	
	def A(self):
		self.rp.q+=self.rp.p*self.motion.qdt/self.rp.dynm3
		self.force_update()	

	def B(self):
		self.rp.p-=self.pes.dpot*self.motion.pdt

	def b(self):
		self.rp.p-=self.rp.dpot*self.motion.pdt

	def M1(self):
		self.rp.Mpp-=(self.pes.ddpot+self.rp.ddpot)*self.motion.pdt*self.rp.Mqp
	
	def M2(self):
		self.rp.Mpq-=(self.pes.ddpot+self.rp.ddpot)*self.motion.pdt*self.rp.Mqq
		
	def M3(self):
		self.rp.Mqp+=self.motion.qdt*self.rp.Mpp/self.rp.dynm3[:,:,None,:,None]
	
	def M4(self):
		self.rp.Mqq+=self.motion.qdt*self.rp.Mpq/self.rp.dynm3[:,:,None,:,None]
		
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
		#self.M3()
		#self.M4()	
 
class Symplectic_order_IV(Integrator):
	"""
	 Mark L. Brewer, Jeremy S. Hulme, and David E. Manolopoulos, 
	"Semiclassical dynamics in up to 15 coupled vibrational degrees of freedom", 
	 J. Chem. Phys. 106, 4832-4839 (1997)
	"""

	def O(self,pmats):
		self.therm.thalfstep(pmats)

	def A(self,k):
		self.rp.q+=self.motion.qdt[k]*self.rp.p/self.rp.dynm3
		self.force_update()
	
	def B(self,k):
		self.rp.p-=self.pes.dpot*self.motion.pdt[k]

	def b(self,k):	
		self.rp.p-=self.rp.dpot*self.motion.pdt[k]

	def M1(self,k):
		self.rp.Mpp-=(self.pes.ddpot+self.rp.ddpot)*self.motion.pdt[k]*self.rp.Mqp
			
	def M2(self,k):
		self.rp.Mpq-=(self.pes.ddpot+self.rp.ddpot)*self.motion.pdt[k]*self.rp.Mqq
		
	def M3(self,k):
		self.rp.Mqp+=self.motion.qdt[k]*self.rp.Mpp/self.rp.dynm3[:,:,None,:,None]
		
	def M4(self,k):
		self.rp.Mqq+=self.motion.qdt[k]*self.rp.Mpq/self.rp.dynm3[:,:,None,:,None]
	
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
			self.pq_kstep(k)

	def Monodromy_step(self):	
		for k in range(4):
			self.Monodromy_kstep(k)

class Runge_Kutta_order_VIII(Integrator):
	def int_func(self,y,t):		
		N = self.rp.nsys*self.rp.nbeads*self.rp.ndim
		self.rp.q = y[:N].reshape(self.rp.nsys,self.rp.ndim,self.rp.nbeads)
		self.rp.p = y[N:2*N].reshape(self.rp.nsys,self.rp.ndim,self.rp.nbeads)
		self.force_update()
	
		n_mmat = self.rp.nsys*self.rp.ndim*self.rp.ndim*self.rp.nbeads*self.rp.nbeads
		self.rp.Mpp = y[2*N:2*N+n_mmat].reshape(self.rp.nsys,self.rp.ndim,self.rp.ndim,self.rp.nbeads,self.rp.nbeads)
		self.rp.Mpq = y[2*N+n_mmat:2*N+2*n_mmat].reshape(self.rp.nsys,self.rp.ndim,self.rp.ndim,self.rp.nbeads,self.rp.nbeads)
		self.rp.Mqp = y[2*N+2*n_mmat:2*N+3*n_mmat].reshape(self.rp.nsys,self.rp.ndim,self.rp.ndim,self.rp.nbeads,self.rp.nbeads)
		self.rp.Mqq = y[2*N+3*n_mmat:2*N+4*n_mmat].reshape(self.rp.nsys,self.rp.ndim,self.rp.ndim,self.rp.nbeads,self.rp.nbeads)
		d_mpp = -(self.pes.ddpot+self.rp.ddpot)*self.rp.Mqp
		d_mpq = -(self.pes.ddpot+self.rp.ddpot)*self.rp.Mqq
		d_mqp = self.rp.Mpp/self.rp.dynm3[:,:,None,:,None]
		d_mqq = self.rp.Mpq/self.rp.dynm3[:,:,None,:,None]
		
		dydt = np.concatenate(((self.rp.p/self.rp.dynm3).flatten(),-(self.pes.dpot+self.rp.dpot).flatten(),d_mpp.flatten(),d_mpq.flatten(),d_mqp.flatten(), d_mqq.flatten()  ))
		return dydt
    
	def integrate(self,tarr,mxstep=1000000):
		y0=np.concatenate((self.rp.q.flatten(), self.rp.p.flatten(),self.rp.Mpp.flatten(),self.rp.Mpq.flatten(),self.rp.Mqp.flatten(),self.rp.Mqq.flatten()))
		sol = scipy.integrate.odeint(self.int_func,y0,tarr,mxstep=mxstep)
		return sol

	def centroid_Mqq(self,sol):
		N = self.rp.nsys*self.rp.nbeads*self.rp.ndim
		n_mmat = self.rp.nsys*self.rp.ndim*self.rp.ndim*self.rp.nbeads*self.rp.nbeads

		Mqq = sol[:,2*N+3*n_mmat:2*N+4*n_mmat].reshape(len(sol),self.rp.nsys,self.rp.ndim,self.rp.ndim,self.rp.nbeads,self.rp.nbeads)
		return Mqq[...,0,0]

	def ret_q(self,sol):
		N = self.rp.nsys*self.rp.nbeads*self.rp.ndim
		q_arr = sol[:,:N].reshape(len(sol),self.rp.nsys,self.rp.ndim,self.rp.nbeads)
		return q_arr	
