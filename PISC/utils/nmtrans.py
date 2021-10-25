import numpy as np
import scipy

# This code needs to be checked thoroughly. There are lots of room 
# for errors.

class FFT(object):
	def __init__(self, ndim, nbeads, nmodes=None):
	 	if(nmodes is None):
			self.nmodes = nbeads
		else: 	
			self.nmodes = nmodes

		self.ft_scale = [1.0,] + [j for i in range(1,self.nmodes//2+1)
                         for j in [1.0, -1.0]]	
		if self.nmodes%2 == 0:
            self.ft_scale.pop(-1)
            ub = -1
        else:
            ub = None

        self.ft_scale = np.array(ft_scale)
        self.ft_scale[1:ub] *= np.sqrt(2.0)
        self.ft_scale /= nbeads

		def cart2mats(cart,mats=None,axis=-1):
			mats = scipy.fftpack.rfft(cart,axis=axis)*self.ft_scale
			return mats
	
		def mats2cart(mats,cart=None,axis=-1):
			cart = scipy.fftpack.irfft(mats/self.ft_scale,axis=axis)
			return cart
