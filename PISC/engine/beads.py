"""
This module defines the main class for treating 
discretized imaginary time path-integrals.
"""

import numpy as np
from PISC.utils import nmtrans,misc
### Work left:
# 1. Display error messages on input errors.
# 2. Augment code to simulate molecules: 
#	 a. Take care of labels initialization and
#		further usage.
#	 b. When molecular labels are passed, ensure 
#		that the masses are appropriately obtained
#		from the utils module.
# 3. Take care of the case with even number of 
#	 Matsubara modes: Not worth doing this.
# 4. Define momenta by default at class object 
#	 definition (Not necessary)
# 5. Figure out what happens to the free ring polymer
#	 propagator when you use fourth order symplectic 
#	 integrator (RSP cannot be done in 4th order code)
# 6. Comment out the code. 
# 7. Take care of the Matsubara mode ordering: It should
#	 be [0,-1,1,-2,2...] (Done)
# 8. Define M matrix as one of the Ring polymer dynamical
#	 variables in both cartesian and Matsubara coordinates
#	 (Not possible to do both simultaneously: Preferable to 
#	  propagate M in Matsubara coordinates)

# Ensure that the nmats, mode and scaling argument passed at 
# initialization is consistently defined. Any potential conflict 
# should be resolved and instructions to use the class object should
# be well-documented.  

class RingPolymer(object):

	def __init__(self,p=None,q=None,pcart=None,qcart=None,
		  Mpp=None,Mpq=None,Mqp=None,Mqq=None,
		  m=None,labels=None,nbeads=None,nmodes=None,
		  scaling=None,sgamma=None,sfreq=None,freqs=None,mode='rp',nmats = None):

		if qcart is None:
			if p is not None:
				self.p = p
				#self.pcart = nmtrans.mats2cart(p)
			else:
				self.p = None
			self.pcart = None
			if nmodes is None:
				nmodes = q.shape[-1]
			self.q = q
			self.qcart = None
			self.nbeads = nmodes
			self.nmodes = nmodes
			self.nsys = len(self.q)
			#self.qcart = nmtrans.mats2cart(q)
			
		if q is None:
			if pcart is not None:
				self.pcart = pcart
				#self.p = nmtrans.FFT.cart2mats(pcart)
			else: 
				self.pcart = None
			self.p=None
			if nbeads is None:
				nbeads = qcart.shape[-1]
			self.qcart = qcart
			self.q = None	
			self.nmodes = nbeads
			self.nbeads = nbeads
			self.nsys = len(self.qcart)	 
			#self.q = nmtrans.FFT.cart2mats(qcart)
			
		self.m = m	
		self.mode = mode 	
		if Mpp is None:
			self.Mpp = None
		else:
			self.Mpp = Mpp

		if Mpq is None:
			self.Mpq = None
		else:
			self.Mpq = Mpq

		if Mqp is None:
			self.Mqp = None
		else:
			self.Mqp = Mqp

		if Mqq is None:
			self.Mqq = None
		else:
			self.Mqq = Mqq

		if scaling is None:
			self.scaling="none"
			self._sgamma = None
			self._sfreq = None	
		elif (scaling=='MFmats'):
			self.nmats = nmats
			self.scaling = scaling
			self._sgamma = sgamma
			self._sfreq = sfreq
		else:
			self.scaling=scaling
			self._sgamma=sgamma
			self._sfreq=sfreq

		if(mode =='rp'):
			self.nmats = None
		elif (mode=='rp/mats' or mode=='mats'):	
			if nmats is None:
				raise ValueError('Number of Matsubara modes needs to be specified')	
			else:
				self.nmats = nmats
		self.freqs = freqs
		
	def bind(self, ens, motion, rng):
		self.ens = ens
		self.motion = motion
		self.rng = rng
		self.ndim = ens.ndim
		self.dt = self.motion.dt	
		
		self.omegan = self.nbeads/self.ens.beta
		self.nmtrans = nmtrans.FFT(self.ndim,self.nbeads,self.nmodes)
		self.nmtrans.compute_nm_matrix()
		self.nm_matrix = self.nmtrans.nm_matrix

		if(self.mode =='mats' or self.mode=='rp/mats'):
			self.freqs = self.get_rp_freqs()
			self.matsfreqs = self.get_mats_freqs()
		elif(self.mode == 'rp' or self.freqs is None):
			self.freqs = self.get_rp_freqs()

		if(self.qcart is None):
			self.qcart = self.nmtrans.mats2cart(self.q)
		elif(self.q is None):
			self.q = self.nmtrans.cart2mats(self.qcart)

		self.m3 = np.ones_like(self.q)*self.m	
		self.sqm3 = np.sqrt(self.m3)
		
		if self.Mpp is None and self.Mqq is None:
			self.Mpp = np.zeros((self.nsys,self.ndim,self.ndim,self.nmodes,self.nmodes))
			for d1 in range(self.ndim):
				for d2 in range(self.ndim):
					self.Mpp[:,d1,d2] = np.eye(self.nmodes,self.nmodes)
			self.Mqq = self.Mpp.copy()
		if self.Mqp is None and self.Mpq is None:
			self.Mqp = np.zeros_like(self.Mqq)
			self.Mpq = np.zeros_like(self.Mqq)		
		
		# The above definition is not foolproof, but works for now.
	
		self.freqs2 = self.freqs**2
		self.nmscale = self.get_dyn_scale()
		self.dynm3 = self.m3*self.nmscale
		self.sqdynm3 = np.sqrt(self.dynm3)
		self.dynfreqs = self.freqs/np.sqrt(self.nmscale)
		self.dynfreq2 = self.dynfreqs**2
		
		self.get_RSP_coeffs()	
		self.ddpot = np.zeros((self.nsys,self.ndim,self.ndim,self.nmodes,self.nmodes))
		for d1 in range(self.ndim):
			for d2 in range(self.ndim):
				self.ddpot[:,d1,d2] = np.eye(self.nmodes,self.nmodes)

		self.ddpot*=(self.dynm3*self.dynfreq2)[:,:,None,:,None]
		
		self.ddpot_cart = np.zeros((self.nsys,self.ndim,self.ndim,self.nbeads,self.nbeads))
		for d1 in range(self.ndim):
			for d2 in range(self.ndim):
				for k in range(self.nbeads-1):
					self.ddpot_cart[:,d1,d2,k,k] = 2
					self.ddpot_cart[:,d1,d2,k,k+1] = -1
					self.ddpot_cart[:,d1,d2,k,k-1] = -1
				self.ddpot_cart[:,d1,d2,self.nbeads-1,0] = -1
				self.ddpot_cart[:,d1,d2,self.nbeads-1,self.nbeads-1] = 2
				self.ddpot_cart[:,d1,d2,self.nbeads-1,self.nbeads-2] = -1
	
		self.ddpot_cart*=(self.dynm3*self.omegan**2)[:,:,None,:,None]	
	
		if self.p is None and self.pcart is None:
			self.p = self.rng.normal(size=self.q.shape,scale=1/np.sqrt(self.ens.beta))
			self.pcart = self.nmtrans.mats2cart(self.p)		
		elif(self.pcart is None):
			self.pcart = self.nmtrans.mats2cart(self.p)
		else:
			self.p = self.nmtrans.cart2mats(self.pcart)
	
	def get_dyn_scale(self):
		scale = np.ones(self.nmodes)
		if (self.scaling=="none"):
			return scale
		elif (self.scaling=="MFmats"):
			if(self._sgamma is None):
				scale = (self.freqs/self._sfreq)**2
			elif(self._sfreq is None):
				scale = (self.freqs/(self._sgamma*self.omegan))**2 
			scale[:self.nmats] = 1.0
			return scale
		elif (self.scaling=='cmd'):
			if(self._sgamma is None):
				scale = (self.freqs/self._sfreq)**2
			elif(self._sfreq is None):
				scale = (self.freqs/(self._sgamma*self.omegan))**2 
			scale[0] = 1.0
			return scale

	def RSP_step(self):
		qpvector = np.empty((self.nmodes,2,self.ndim,len(self.q)))
		qpvector[:,0] = (self.p/self.sqdynm3).T
		qpvector[:,1] = (self.q*self.sqdynm3).T

		qpvector[:] = np.einsum('ijk,ik...->ij...',self.RSP_coeffs,qpvector)#np.matmul(self.RSP_coeffs,qpvector)#
		#np.einsum is working perfectly, no issue there.		
		
		self.p[:] = qpvector[:,0].T*self.sqdynm3
		self.q[:] = qpvector[:,1].T/self.sqdynm3 

	def get_rp_freqs(self):
		n = [0]
		for i in range(1,self.nmodes//2+1):
			n.append(-i)
			n.append(i)
		if self.nmodes%2 == 0:
			n.pop(-2)
		freqs = np.sin(np.array(n)*np.pi/self.nbeads)*(2*self.omegan)
		return freqs

	def get_mats_freqs(self):
		n = [0]
		if(self.mode=='mats'):
			for i in range(1,self.nmodes//2+1):
				n.append(-i)
				n.append(i)
			freqs = 2*np.pi*np.array(n)/(self.ens.beta)
			return freqs			
		elif(self.mode=='rp/mats'):
			for i in range(1,self.nmats//2+1):
				n.append(-i)
				n.append(i)
			freqs = 2*np.pi*np.array(n)/(self.ens.beta)
			return freqs			
		
	def get_RSP_coeffs(self):
		self.RSP_coeffs = np.empty((self.nmodes, 2, 2))
		for n in range(self.nmodes):
			mat = np.eye(2)*np.cos(self.dynfreqs[n]*self.dt)		
			mat[0,1] = -self.dynfreqs[n]*np.sin(self.dynfreqs[n]*self.dt)		
			mat[1,0] = np.sinc(self.dynfreqs[n]*self.dt/np.pi)*self.dt
			self.RSP_coeffs[n] = mat
				
	def nm_matrix(self): # Delete this. It is redundant.
			narr = [0]
			for i in range(1,self.nmodes//2+1):
				narr.append(-i)
				narr.append(i)
			if self.nmodes%2 == 0:
				narr.pop(-1)
	
			self.nm_matrix = np.zeros((self.nmodes,self.nmodes))
			self.nm_matrix[:,0] = 1/np.sqrt(self.nmodes)
			for l in range(self.nbeads):
				for n in range(1,len(narr)):
					#print('n',narr,self.nmodes//2)
					if(narr[n]<0):
							self.nm_matrix[l,n] = np.sqrt(2/self.nmodes)*np.cos(2*np.pi*(l)*narr[n]/self.nmodes)
							#self.nm_matrix[narr[n],l] = np.sqrt(2/self.nbeads)*np.cos(2*np.pi*(l+1)*narr[n]/self.nmodes)
							#print('l,n',l,n,narr[n],self.nm_matrix[l,n])
					else:
							self.nm_matrix[l,n] = np.sqrt(2/self.nmodes)*np.sin(2*np.pi*(l)*narr[n]/self.nmodes)
							#self.nm_matrix[narr[n],l] = np.sqrt(2/self.nbeads)*np.sin(2*np.pi*(l+1)*narr[n]/self.nmodes)
							#print('l,n',l,n,narr[n],self.nm_matrix[l,n],2*l*narr[n]/self.nmodes)
				if(self.nmodes%2==0):
					self.nm_matrix[l,self.nmodes-1] =(-1)**l/np.sqrt(self.nmodes)
			#self.nm_matrix = self.nm_matrix.T	
		
	def mats_beads(self):
		if self.nmats is None: 
			return self.qcart
		else:
			ret= np.einsum('ij,...j',(self.nmtrans.nm_matrix)[:,:self.nmats],self.q[...,:self.nmats]) #Check this!
			return ret
	
	def mats2cart(self):
		self.qcart = self.nmtrans.mats2cart(self.q)
		self.pcart = self.nmtrans.mats2cart(self.p)
	
	def cart2mats(self):
		self.q = self.nmtrans.cart2mats(self.qcart)
		self.p = self.nmtrans.cart2mats(self.pcart)

	@property
	def theta(self): # Change for more than 1D
		ret = self.matsfreqs*misc.pairwise_swap(self.q[...,:self.nmats],self.nmats)*self.p[...,:self.nmats]
		#print(self.q[...,1]*self.p[...,2],self.q[...,2]*self.p[...,1])
		#print('ret', ret.shape)
		#print('freq', self.matsfreqs)
		#print('q',self.q[...,:self.nmats], misc.pairwise_swap(self.q[...,:self.nmats],self.nmats))
		return np.sum(ret,axis=2)
		
	@property
	def kin(self):
		return np.sum(0.5*(self.p/self.sqm3)**2)		

	@property
	def dynkin(self):
		return np.sum(0.5*(self.p/self.sqdynm3)**2)

	@property
	def pot(self):
		return np.sum(0.5*self.dynm3*self.dynfreq2*self.q**2)
	
	@property	
	def dpot(self):
		return self.dynm3*self.dynfreq2*self.q

	@property
	def dpot_cart(self):
		return self.dynm3*self.omegan**2*(2*self.qcart-np.roll(self.qcart,1,axis=-1)-np.roll(self.qcart,-1,axis=-1))
		
