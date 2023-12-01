import numpy as np
import scipy.fftpack
from PISC.utils.nmtrans_f import nmtrans
import time

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

    def cart2mats(self,cart,mats=None,axis=-1,fortran=False):
        if(fortran):
            nmtrans.cart2mats(cart,mats,self.nm_matrix_f)
        else:
            mats = scipy.fftpack.rfft(cart,axis=axis)*self.ft_scale
        return mats
    
    def mats2cart(self,mats,cart=None,axis=-1,fortran=False):
        start = time.time()
        if(fortran):
            nmtrans.mats2cart(mats,cart,self.nm_matrix_f)
        else:   
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
                if(narr[n]<0):
                        self.nm_matrix[l,n] = np.sqrt(2/self.nmodes)*np.cos(2*np.pi*(l)*narr[n]/self.nmodes)
                else:
                        self.nm_matrix[l,n] = np.sqrt(2/self.nmodes)*np.sin(2*np.pi*(l)*narr[n]/self.nmodes)
            if(self.nmodes%2==0):
                self.nm_matrix[l,self.nmodes-1] =(-1)**l/np.sqrt(self.nmodes)

        self.nm_matrix_f = self.nm_matrix.T

    def cart2mats_hessian(self,cart,mats=None,fortran=False):
        if(fortran):
            nmtrans.cart2mats_hessian(cart,mats,self.nm_matrix_f)
        else:
            mats = np.einsum('ij,...jnk',self.nm_matrix.T, np.einsum('...inj,jk',cart,self.nm_matrix))
        return mats

    def mats2cart_hessian(self,mats,cart=None,fortran=False):
        if(fortran):
            nmtrans.mats2cart_hessian(mats,cart,self.nm_matrix_f)
        else:
            cart = np.einsum('ij,...jnk',self.nm_matrix, np.einsum('...inj,jk',mats,self.nm_matrix.T))
        return cart


