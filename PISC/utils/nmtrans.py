import numpy as np
import scipy.fftpack

# This code needs to be checked thoroughly. There are lots of room 
# for errors.

class FFT(object):
	def __init__(self, ndim, nbeads, nmodes=None):
		if(nmodes is None):
			self.nmodes = nbeads
		else:	
			self.nmodes = nmodes
		self.nbeads=nbeads
		self.ft_scale = [1.0,] + [j for i in range(1,self.nmodes//2+1)
						 for j in [1.0, -1.0]]	
		if self.nmodes%2 == 0:
			self.ft_scale.pop(-1)
			ub = -1
		else:
			ub = None

		self.ft_scale = np.array(self.ft_scale)
		self.ft_scale[1:ub] *= np.sqrt(2.0)
		self.ft_scale /= np.sqrt(nbeads)

		#print('ft_scale', self.ft_scale,np.sqrt(1/nbeads))
		#self.ft_scale = np.ones(nbeads)*(-(2.0/nbeads)**0.5)
		#self.ft_scale[0] = 1/nbeads**0.5
        #print('ft_rearrange',self.ft_rearrange)  
		#if(nbeads%2 == 0):
		#	self.ft_scale[nbeads-1]= 1/nbeads**0.5
		
	def cart2mats(self,cart,mats=None,axis=-1):
		mats = scipy.fftpack.rfft(cart,axis=axis)*self.ft_scale
		return mats
	
	def mats2cart(self,mats,cart=None,axis=-1):
		cart = scipy.fftpack.irfft(mats/self.ft_scale,axis=axis)
		return cart

	def compute_nm_matrix(self):
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

	def cart2mats_hessian(self,cart,mats=None):
		mats = np.einsum('ij,...jk',self.nm_matrix.T, np.einsum('...ij,jk',cart,self.nm_matrix))
		return mats

	def mats2cart_hessian(self,mats,cart=None):
		cart = np.einsum('ij,...jk',self.nm_matrix, np.einsum('...ij,jk',mats,self.nm_matrix.T))
		return cart
