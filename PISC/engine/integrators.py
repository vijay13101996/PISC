"""
This module contains the code for implementing second 
and fourth order symplectic integrator.
"""

import numpy as np
import scipy
import scipy.integrate
from scipy.integrate import odeint, ode
from PISC.utils.misc import hess_compress, hess_mul

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
	def O(self,pc):
		self.therm.thalfstep(pc)

	def b_cent(self):
		self.rp.p[...,:self.rp.nmats]-=self.pes.centrifugal_term()*self.motion.pdt
	
	def A(self):
		self.rp.q+=self.rp.p*self.motion.qdt/self.rp.dynm3
		self.force_update()	

	def Av(self):
		self.rp.dq+=self.rp.dp*self.motion.qdt/self.rp.dynm3
	
	def B(self,centmove=True):
		if(centmove):
			self.rp.p-=self.pes.dpot*self.motion.pdt	
		else:
			self.rp.p[...,1:]-=self.pes.dpot[...,1:]*self.motion.pdt

	def Bv(self,centmove=True):
		hess = hess_compress(self.pes.ddpot,self.rp)
		#self.pes.ddpot.swapaxes(2,3).reshape(-1,self.pes.ndim*self.rp.nbeads,self.pes.ndim*self.rp.nbeads)
		dq=self.rp.dq.reshape(-1,self.pes.ndim*self.rp.nbeads)
		dpc= np.einsum('ijk,ik->ij',hess,dq)
		dpc=dpc.reshape(-1,self.pes.ndim,self.rp.nbeads)
		if(centmove):
			self.rp.dp-=dpc*self.motion.pdt	
		else:
			self.rp.dp[...,1:]-=dpc[...,1:]*self.motion.pdt

	def b(self):
		if self.rp.nmats is None:
			self.rp.p-=self.rp.dpot*self.motion.pdt
		else:
			self.rp.p[...,self.rp.nmats:]-=self.rp.dpot[...,self.rp.nmats:]*self.motion.pdt

	def bv(self):
		hess = hess_compress(self.rp.ddpot,self.rp)
		#self.rp.ddpot.swapaxes(2,3).reshape(-1,self.rp.ndim*self.rp.nbeads,self.rp.ndim*self.rp.nbeads)
		dq=self.rp.dq.reshape(-1,self.pes.ndim*self.rp.nbeads)
		dpc= np.einsum('ijk,ik->ij',hess,dq)
		dpc=dpc.reshape(-1,self.pes.ndim,self.rp.nbeads)
		self.rp.dp-=dpc*self.motion.pdt
		if self.rp.nmats is None:	
			self.rp.dp-=dpc*self.motion.pdt
		else:
			self.rp.p[...,self.rp.nmats:]-=dpc[...,self.rp.nmats:]*self.motion.pdt

	def M1(self):
		#self.rp.Mpp-=(self.pes.ddpot)*self.motion.pdt*self.rp.Mqp
		hess_mul(self.pes.ddpot,self.rp.Mqp,self.rp.Mpp,self.rp,self.motion.pdt)		
	
	def m1(self):
		#self.rp.Mpp-=(self.rp.ddpot)*self.motion.pdt*self.rp.Mqp	
		hess_mul(self.rp.ddpot,self.rp.Mqp,self.rp.Mpp,self.rp,self.motion.pdt)		

	def M2(self):
		#self.rp.Mpq-=(self.pes.ddpot)*self.motion.pdt*self.rp.Mqq
		hess_mul(self.pes.ddpot,self.rp.Mqq,self.rp.Mpq,self.rp,self.motion.pdt)		

	def m2(self):
		#self.rp.Mpq-=(self.rp.ddpot)*self.motion.pdt*self.rp.Mqq
		hess_mul(self.rp.ddpot,self.rp.Mqq,self.rp.Mpq,self.rp,self.motion.pdt)		

	def M3(self):
		self.rp.Mqp+=self.motion.qdt*self.rp.Mpp/self.rp.dynm3[:,:,None,:,None]
	
	def M4(self):
		self.rp.Mqq+=self.motion.qdt*self.rp.Mpq/self.rp.dynm3[:,:,None,:,None]
		
	def pq_step(self,centmove=True):
		self.B(centmove)
		self.b()
		self.A()
		self.b()
		self.B(centmove)
		
	def pq_step_RSP(self,centmove=True):
		self.B(centmove)
		self.rp.RSP_step()
		self.force_update()
		self.B(centmove)

	def pq_step_nosprings(self,centmove=True):
		self.B(centmove)
		self.A()	
		self.B(centmove)
	
	def var_step(self,centmove=True):
		self.B(centmove)
		self.Bv(centmove)
		self.b()
		self.bv()
		
		self.A()
		self.Av()
		
		self.b()
		self.bv()
		self.B(centmove)
		self.Bv(centmove)
	
#	def var_step_nosprings (think about var_step_RSP)

	def Monodromy_step_nosprings(self):	
		self.B()
		self.M1()
		self.M2()
		self.M3()
		self.M4()
		self.A()
		self.B()
		self.M1()
		self.M2()
		
	def Monodromy_step(self):
		self.B()
		self.b()
		self.M1()
		self.m1()
		self.M2()
		self.m2()
		self.M3()
		self.M4()
		self.A()
		self.B()
		self.b()
		self.M1()
		self.m1()
		self.M2()
		self.m2()
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
	
	def B(self,k,centmove=True):
		if centmove:
			self.rp.p-=self.pes.dpot*self.motion.pdt[k]
		else:
			self.rp.p[...,1:]-=self.pes.dpot[...,1:]*self.motion.pdt[k]

	def b(self,k):	
		if self.rp.nmats is None:
			self.rp.p-=self.rp.dpot*self.motion.pdt[k]
		else:
			self.rp.p[...,self.rp.nmats:]-=self.pes.dpot[...,self.rp.nmats:]*self.motion.pdt[k]
	
	def Av(self,k):
		self.rp.dq+=self.rp.dp*self.motion.qdt[k]/self.rp.dynm3
	
	def Bv(self,k,centmove=True):
		hess = self.pes.ddpot.swapaxes(2,3).reshape(-1,self.pes.ndim*self.rp.nbeads,self.rp.ndim*self.rp.nbeads)
		dq=self.rp.dq.reshape(-1,self.pes.ndim*self.rp.nbeads)
		dpc= np.einsum('ijk,ik->ij',hess,dq) #Check once again!
		dpc=dpc.reshape(-1,self.pes.ndim,self.rp.nbeads)
		if(centmove):
			self.rp.dp-=dpc*self.motion.pdt[k]
		else:
			self.rp.dp[...,1:]-=dpc[...,1:]*self.motion.pdt[k]

	def bv(self,k):
		hess = self.rp.ddpot.swapaxes(2,3).reshape(-1,self.rp.ndim*self.rp.nbeads,self.rp.ndim*self.rp.nbeads)
		dq=self.rp.dq.reshape(-1,self.pes.ndim*self.rp.nbeads)
		dpc= np.einsum('ijk,ik->ij',hess,dq) #Check once again!
		dpc=dpc.reshape(-1,self.pes.ndim,self.rp.nbeads)
		self.rp.dp-=dpc*self.motion.pdt[k]
		if self.rp.nmats is None:	
			self.rp.dp-=dpc*self.motion.pdt[k]
		else:
			self.rp.p[...,self.rp.nmats:]-=dpc[...,self.rp.nmats:]*self.motion.pdt[k]
	
	def M1(self,k):
		#self.rp.Mpp -=(self.pes.ddpot)*self.motion.pdt[k]*self.rp.Mqp
	
		#print('Mpp before',self.rp.Mpp)	
		#hess = hess_compress(self.pes.ddpot,self.rp)
		#Mqp = hess_compress(self.rp.Mqp,self.rp)
		#Mpp = hess_compress(self.rp.Mpp,self.rp)
		#c2 = np.matmul(hess,Mqp)*self.motion.pdt[k]
		#Mpp-=c2
		#print('Mpp after',self.rp.Mpp,'\n')
	
		hess_mul(self.pes.ddpot,self.rp.Mqp,self.rp.Mpp,self.rp,self.motion.pdt[k])		
			
	def m1(self,k):
		#self.rp.Mpp-=(self.rp.ddpot)*self.motion.pdt[k]*self.rp.Mqp
		
		#hess = hess_compress(self.rp.ddpot,self.rp)
		#Mqp = hess_compress(self.rp.Mqp,self.rp)
		#Mpp = hess_compress(self.rp.Mpp,self.rp)
		#c2 = np.matmul(hess,Mqp)*self.motion.pdt[k]
		#Mpp-=c2

		hess_mul(self.rp.ddpot,self.rp.Mqp,self.rp.Mpp,self.rp,self.motion.pdt[k])		
		
	def M2(self,k):
		#self.rp.Mpq-=(self.pes.ddpot)*self.motion.pdt[k]*self.rp.Mqq
		
		#hess = hess_compress(self.pes.ddpot,self.rp)
		#Mqq = hess_compress(self.rp.Mqq,self.rp)
		#Mpq = hess_compress(self.rp.Mpq,self.rp)
		#c2 = np.matmul(hess,Mqq)*self.motion.pdt[k]
		#Mpq-=c2

		hess_mul(self.pes.ddpot,self.rp.Mqq,self.rp.Mpq,self.rp,self.motion.pdt[k])		

	def m2(self,k):
		#self.rp.Mpq-=(self.rp.ddpot)*self.motion.pdt[k]*self.rp.Mqq
		
		#hess = hess_compress(self.rp.ddpot,self.rp)
		#Mqq = hess_compress(self.rp.Mqq,self.rp)
		#Mpq = hess_compress(self.rp.Mpq,self.rp)
		#c2 = np.matmul(hess,Mqq)*self.motion.pdt[k]
		#Mpq-=c2

		hess_mul(self.rp.ddpot,self.rp.Mqq,self.rp.Mpq,self.rp,self.motion.pdt[k])		

	def M3(self,k):
		self.rp.Mqp+=self.motion.qdt[k]*self.rp.Mpp/self.rp.dynm3[:,:,None,:,None]
		
	def M4(self,k):
		self.rp.Mqq+=self.motion.qdt[k]*self.rp.Mpq/self.rp.dynm3[:,:,None,:,None]
	
	def pq_kstep(self,k,centmove=True):
		self.B(k,centmove)
		self.b(k)
		self.A(k)

	def pq_kstep_nosprings(self,k):
		self.B(k)
		self.A(k)

	def var_kstep(self,k,centmove=True):
		self.B(k,centmove)
		self.Bv(k,centmove)
		self.b(k)
		self.bv(k)
		self.A(k)
		self.Av(k)

	def Monodromy_kstep(self,k):
		self.B(k)
		self.b(k)
		self.M1(k)
		self.m1(k)
		self.M2(k)
		self.m2(k)
		self.M3(k)
		self.M4(k)
		self.A(k)

	def Monodromy_kstep_nosprings(self,k):
		self.B(k)
		self.M1(k)
		self.M2(k)
		self.M3(k)
		self.M4(k)
		self.A(k)

	def pq_step(self,centmove=True):	
		for k in range(4):
			self.pq_kstep(k,centmove=True)
	
	def pq_step_nosprings(self):	
		for k in range(4):
			self.pq_kstep_nosprings(k)

	def var_step(self):
		for k in range(4):
			self.var_kstep(k,centmove=True)

	def Monodromy_step(self):	
		for k in range(4):
			self.Monodromy_kstep(k)

	def Monodromy_step_nosprings(self):	
		for k in range(4):
			self.Monodromy_kstep_nosprings(k)
		
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
