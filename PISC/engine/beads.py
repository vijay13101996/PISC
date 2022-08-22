"""
This module defines the main 'RingPolymer' class for treating 
discretized imaginary time path-integrals. This class can be used
to run CMD, RPMD, Matsubara and mean-field Matsubara simulations
"""

from turtle import setundobuffer
import numpy as np
from PISC.utils import nmtrans,misc
### Work left:
# 1. Display error messages on input errors.
# 2. Comment out the code. 
  
class RingPolymer(object):
	"""
	Attributes:
	nbeads : Number of imaginary time points or 'beads'
	nmodes : Number of ring-polymer normal modes (Usually same as nbeads)
	nsys   : Number of parallel ring-polymer units 
	nmats  : Number of Matsubara modes (Differs from nbeads for Matsubara simulations)

	q/qcart 		: Ring-polymer bead positions in Cartesian/Matsubara coordinates
	p/pcart 		: Ring-polymer bead momenta in Cartesian/Matsubara coordinates
	Mpp,Mpq,Mqp,Mqq : Monodromy matrix elements of the ring-polymer beads in Matsubara coordinates
	
	m3	 : Vectorized ring-polymer bead masses
	sqm3 : Square root of m3
	"""

	def __init__(self,p=None,dp=None,q=None,dq=None,pcart=None,dpcart=None,qcart=None,dqcart=None,
		  Mpp=None,Mpq=None,Mqp=None,Mqq=None,
		  m=None,labels=None,nbeads=None,nmodes=None,
		  scaling=None,sgamma=None,sfreq=None,freqs=None,mode='rp',nmats = None):

		if qcart is None:
			if p is not None:
				self.p = p	
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
			
		if q is None:
			if pcart is not None:
				self.pcart = pcart	
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
		
		if dqcart is not None:
			self.dqcart=dqcart
			if dpcart is not None:
				self.dpcart=dpcart
			else:
				self.dpcart=0.0
		else:
			self.dqcart=None
			self.dpcart=None
		if dq is not None:
			self.dq=dq
			if dp is not None:
				self.dp=dp
			else:
				self.dp=0.0
		else:
			self.dq=None
			self.dp=None

		# Create variables dq,dp, dqcart, dpcart default set to None
		# If not None, they can be passed in cartesian/Matsubara coordinates.
		# Like with q/p, qcart/pcart, these variables need to be interconverted.

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
		
		if(self.dqcart is not None):
			self.dq = self.nmtrans.cart2mats(self.dqcart)
			self.dp = self.nmtrans.cart2mats(self.dpcart)
		elif(self.dq is not None):
			self.dqcart = self.nmtrans.mats2cart(self.dq)
			self.dpcart = self.nmtrans.mats2cart(self.dp)

		self.m3 = np.ones_like(self.q)*self.m	
		self.sqm3 = np.sqrt(self.m3)
		
		if self.Mpp is None and self.Mqq is None:
			self.Mpp = np.zeros((self.nsys,self.ndim,self.ndim,self.nmodes,self.nmodes))
			for d in range(self.ndim):
				self.Mpp[:,d,d] = np.eye(self.nmodes,self.nmodes)
			self.Mqq = self.Mpp.copy()
		if self.Mqp is None and self.Mpq is None:
			self.Mqp = np.zeros_like(self.Mqq)
			self.Mpq = np.zeros_like(self.Mqq)			
	
		self.freqs2 = self.freqs**2
		self.nmscale = self.get_dyn_scale()
		self.dynm3 = self.m3*self.nmscale
		self.sqdynm3 = np.sqrt(self.dynm3)
		self.dynfreqs = self.freqs/np.sqrt(self.nmscale)
		self.dynfreq2 = self.dynfreqs**2
		
		self.get_RSP_coeffs()	
		self.ddpot = np.zeros((self.nsys,self.ndim,self.ndim,self.nmodes,self.nmodes))
		self.ddpot_cart = np.zeros((self.nsys,self.ndim,self.ndim,self.nbeads,self.nbeads))
			
		##Check here when doing multi-D simulation with beads.
		if(self.nbeads>1):	
			for d in range(self.ndim):
				self.ddpot[:,d,d] = np.eye(self.nmodes,self.nmodes)
			#for d1 in range(self.ndim):  
			#	for d2 in range(self.ndim):
			#		self.ddpot[:,d1,d2] = np.eye(self.nmodes,self.nmodes)

			self.ddpot*=(self.dynm3*self.dynfreq2)[:,:,None,:,None]
			
			for d in range(self.ndim):
				for k in range(self.nbeads-1):
						self.ddpot_cart[:,d,d,k,k] = 2
						self.ddpot_cart[:,d,d,k,k+1] = -1
						self.ddpot_cart[:,d,d,k,k-1] = -1
				self.ddpot_cart[:,d,d,self.nbeads-1,0] = -1
				self.ddpot_cart[:,d,d,self.nbeads-1,self.nbeads-1] = 2
				self.ddpot_cart[:,d,d,self.nbeads-1,self.nbeads-2] = -1

			#for d1 in range(self.ndim):
			#	for d2 in range(self.ndim):
			#		for k in range(self.nbeads-1):
			#			self.ddpot_cart[:,d1,d2,k,k] = 2
			#			self.ddpot_cart[:,d1,d2,k,k+1] = -1
			#			self.ddpot_cart[:,d1,d2,k,k-1] = -1
			#		self.ddpot_cart[:,d1,d2,self.nbeads-1,0] = -1
			#		self.ddpot_cart[:,d1,d2,self.nbeads-1,self.nbeads-1] = 2
			#		self.ddpot_cart[:,d1,d2,self.nbeads-1,self.nbeads-2] = -1
		
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

		qpvector[:] = np.einsum('ijk,ik...->ij...',self.RSP_coeffs,qpvector)
		
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
				
	def nm_matrix(self):
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
					if(narr[n]<0):
							self.nm_matrix[l,n] = np.sqrt(2/self.nmodes)*np.cos(2*np.pi*(l)*narr[n]/self.nmodes)							
					else:
							self.nm_matrix[l,n] = np.sqrt(2/self.nmodes)*np.sin(2*np.pi*(l)*narr[n]/self.nmodes)							
				if(self.nmodes%2==0):
					self.nm_matrix[l,self.nmodes-1] =(-1)**l/np.sqrt(self.nmodes)
		
	def mats_beads(self):
		if self.nmats is None: 
			return self.qcart
		else:
			ret= np.einsum('ij,...j',(self.nmtrans.nm_matrix)[:,:self.nmats],self.q[...,:self.nmats]) #Check this!
			return ret
	
	def mats2cart(self):
		self.qcart = self.nmtrans.mats2cart(self.q)
		self.pcart = self.nmtrans.mats2cart(self.p)
		if(self.dq is not None):
			self.dqcart = self.nmtrans.mats2cart(self.dq)
			self.dpcart = self.nmtrans.mats2cart(self.dp)
		# If dq,dp are not None, convert them too
	
	def cart2mats(self):
		self.q = self.nmtrans.cart2mats(self.qcart)
		self.p = self.nmtrans.cart2mats(self.pcart)
		if(self.dqcart is not None):
			self.dq = self.nmtrans.cart2mats(self.dqcart)
			self.dp = self.nmtrans.cart2mats(self.dpcart)
		# If dqcart,dpcart are not None, convert them too

	@property
	def theta(self): # Check for more than 1D
		ret = self.matsfreqs*misc.pairwise_swap(self.q[...,:self.nmats],self.nmats)*self.p[...,:self.nmats]
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
	def pot_cart(self):
		return np.sum(0.5*self.m3*self.omegan**2*(self.qcart-np.roll(self.qcart,1,axis=-1))**2)

	@property	
	def dpot(self):
		return self.dynm3*self.dynfreq2*self.q

	@property
	def dpot_cart(self):
		return self.dynm3*self.omegan**2*(2*self.qcart-np.roll(self.qcart,1,axis=-1)-np.roll(self.qcart,-1,axis=-1))
		
