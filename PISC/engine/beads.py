"""
This module defines the main class for treating 
discretized imaginary time path-integrals.
"""

import numpy as np
from PISC.utils import nmtrans

### Work left:
# 1. Display error messages on input errors.
# 2. Augment code to simulate molecules: 
#    a. Take care of labels initialization and
#    	further usage.
#	 b. When molecular labels are passed, ensure 
#		that the masses are appropriately obtained
# 		from the utils module.
# 3. Take care of the case with even number of 
#	 Matsubara modes: Not worth doing this.
# 4. Define momenta by default at class object 
#    definition (Not necessary)
# 5. Figure out what happens to the free ring polymer
#	 propagator when you use fourth order symplectic 
#    integrator (RSP cannot be done in 4th order code)
# 6. Comment out the code. 
# 7. Take care of the Matsubara mode ordering: It should
#	 be [0,-1,1,-2,2...] (Done)
# 8. Define M matrix as one of the Ring polymer dynamical
#	 variables in both cartesian and Matsubara coordinates
#    (Not possible to do both simultaneously: Preferable to 
#	  propagate M in Matsubara coordinates)

class RingPolymer(object):

	def __init__(self,p=None,q=None,pcart=None,qcart=None,
		  Mpp=None,Mpq=None,Mqp=None,Mqq=None,
          m=None,labels=None,nbeads=None,nmodes=None,
          scaling=None,sgamma=None,sfreq=None,freqs=None,mode='rp',nmats = None)

		if qcart is None and pcart is None:
			if nmodes is None:
				nmodes = q.shape[-1]
			self.q = q
			self.nbeads = nmodes
			self.nmodes = nmodes
			self.qcart = nmtrans.mats2cart(q)
			self.pcart = nmtrans.mats2cart(p)

		if q is None and p is None:
			if nbeads = None:
					nbeads = qcart.shape[-1]
			self.qcart = qcart	
			self.nmodes = nbeads
			self.nbeads = nbeads	 
			self.q = nmtrans.cart2mats(qcart)
			self.p = nmtrans.cart2mats(pcart)

		self.m = m
		self.nmtrans = nmtrans
	
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

		else:
			self.scaling = scaling.lower()
			self._sgamma = sgamma
			self._sfreq = sfreq
		
		self.freqs = freqs
		if (self.scaling == 'MFmats' or self.scaling == 'mats'):
			self.nmats = nmats #Add error statement here.

		self.nmtrans = nmtrans.FFT(self.ndim,self.nbeads,self.nmodes)

	def bind(self, ens, motion, prng):
		self.ens = ens
		self.motion = motion
		self.prng = prng
		self.nsys = ens.nsys
		self.ndim = ens.ndim
 		self.dt = self.motion.qdt	
		
		self.omegan = self.nbeads/self.ens.beta
		self.m3 = np.ones_like(self.q)*self.m	
	    self.sqm3 = np.sqrt(self.m3)
		
		if(self.mode =='mats'):
			self.freqs = self.get_mats_freqs()
		elif(self.mode == 'MFmats'):
			self.freqs = self.get_rp_freqs()
			self.matsfreqs = self.get_mats_freqs()
		elif(self.mode == 'rp' or self.freqs is None):
			self.freqs = self.get_rp_freqs()

		if self.Mpp is None and self.Mqq is None:
			self.Mpp = np.zeros(self.nsys,self.ndim,self.ndim,self.nmodes,self.nmodes)
			for d in range(self.ndim):
				self.Mpp[:,d,d] = np.eye(self.nmodes,self.nmodes)
			self.Mqq = self.Mpp.copy()
		if self.Mqp is None and self.Mpq is None:
			self.Mqp = np.zeros_like(Mqq)
			self.Mpq = np.zeros_like(Mqq)		
		
		# The above definition is not foolproof, but works for now.
	
		if self.p is None:
            sp = self.prng.normal(size=self.q.shape,
                                  scale=1/np.sqrt(self.ens.beta))
            self.p = self.sdynm3*sp
            self.pcart = self.nmtrans.mats2cart(self.p, self.pcart)	

		
		if(self.mode == 'rp' or self.mode == 'MFmats'):	
			self.freqs2 = self.freqs**2
			self.nmscale = self.get_dyn_scale()
			self.dynm3 = self.m3*self.nmscale
			self.sqdynm3 = np.sqrt(self.dynm3)
			self.dynfreqs = self.freqs/np.sqrt(self.nmscale)
			self.dynfreq2 = self.dynfreqs**2
			
			self.get_RSP_coeffs()	
			
			self.ddpot = np.zeros(self.nsys,self.ndim,self.ndim,self.nmodes,self.nmodes)
			for d in range(self.ndim):
				self.ddpot[:,d,d] = np.eye(self.nmodes,self.nmodes)
 
			self.ddpot*=(self.m3*self.freqs2)

		# Variables specific to Matsubara dynamics may need to be defined as well.

	def get_dyn_scale(self):
		scale = np.ones(self.nmodes)
		if (self.scaling=="none" or self.scaling=="mats"):
			return scale
		elif (self.scaling=="MFmats"):
			scale = (self.freqs/self._sfreq)**2
			scale[:self.nmats] = 1.0
			return scale

	def RSP_step(self):
		qpvector = np.zeros((self.nmodes,2,len(self.q)))
		qpvector[:,0,:] = (self.p/self.sdynm3).T
		qpvector[:,1,:] = (self.q*self.sdynm3).T

		qpvector[:] = np.matmul(self.RSP_coeffs,qpvector)
		self.p[:] = qpvector[:,0,:].T*self.dynm3
		self.q[:] = qpvector[:,1,:].T/self.dynm3 

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
		# Modify this for mean-field Matsubara
		n = [0]
        if(self.mode=='mats'):
			for i in range(1,self.nmodes//2+1):
				n.append(-i)
				n.append(i)
			freqs = 2*np.pi*np.array(n)/(self.ens.beta)
			return freqs			
		elif(self.mode=='MFmats'):
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
		
	def mats2cart(self):
		self.rp.qcart = self.nmtrans.mats2cart(self.rp.q)
		self.rp.pcart = self.nmtrans.mats2cart(self.rp.p)
	
	def cart2mats(self):
		self.rp.q = self.nmtrans.cart2mats(self.rp.qcart)
		self.rp.p = self.nmtrans.cart2mats(self.rp.pcart)

	@property	
	def dpot(self):
		return self.m3*self.freqs2*self.q
		
