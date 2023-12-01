import numpy as np
import PISC
from PISC.utils.misc import pairwise_swap
import time

"""
This is the base class for all PES definitions. Each PES
defined in this directory needs to have a dpot function and 
a ddpot function, and provision to convert from cartesian to 
Matsubara coordinates. 
"""

# Store all the Matsubara potentials with a 'Matsubara' tag 

class PES(object):
    def __init__(self):
        self.pot = None 
        self.dpot_cart = None
        self.dpot = None
        self.ddpot_cart = None
        self.ddpot = None

    def bind(self,ens,rp,pes_fort=False,transf_fort=False):
        # Bind the ensemble and ring polymer objects to the PES
        self.ens = ens
        self.rp = rp
        self.ndim = ens.ndim
        self.nsys = rp.nsys
        self.nmtrans = rp.nmtrans

        # Allocate memory for the potential, force and hessian
        # The potential, force and hessian is stored along both 
        # ring polymer beads and normal modes coordinates

        self.pot = np.zeros((self.nsys, self.rp.nbeads))    
        self.dpot_cart = np.zeros_like(rp.qcart)
        self.dpot = np.zeros_like(rp.q)
        self.ddpot_cart = np.zeros_like(rp.ddpot_cart)
        self.ddpot = np.zeros_like(rp.ddpot)
  
        self.ddpotmat = np.zeros((self.nsys,self.ndim,self.rp.nmodes,self.ndim,self.rp.nmodes))
        
        # ddpotmat is the multiplier required to expand the Hessian to the above dimension     
        for d1 in range(self.ndim):
            for d2 in range(self.ndim):
                self.ddpotmat[:,d1,:,d2] = np.eye(self.rp.nmodes,self.rp.nmodes)
        
        self._bind_fort()

        self.pes_fort = pes_fort
        self.transf_fort = transf_fort

    def _bind_fort(self):
        # Declare fortran variables as a 'view' of the python variables by transposing them  
        self.pot_f = self.pot.T
        self.dpot_cart_f = self.dpot_cart.T
        self.dpot_f = self.dpot.T

        self.ddpotmat_f = self.ddpotmat.T
        self.ddpot_cart_f = self.ddpot_cart.T
        self.ddpot_f = self.ddpot.T
   
    # The following functions need to be defined for each PES
    def potential(self,q):
        raise NotImplementedError("Potential function not defined")

    def potential_f(self,pot_f,q):
        raise NotImplementedError("Potential function not defined")

    def dpotential(self,q):
        raise NotImplementedError("Force function not defined")

    def dpotential_f(self,dpot_cart_f,q):
        raise NotImplementedError("Potential function not defined")

    def ddpotential(self,q):
        raise NotImplementedError("Hessian function not defined")
    
    def ddpotential_f(self,ddpotmat,ddpot_cart_f,q):
        raise NotImplementedError("Potential function not defined")

    def dpot_rearrange(self,dpot):
        # Rearrange the force vector to be consistent with the convention used in this code
        # Helpful to use this to compute bead forces from the PES 
        return np.transpose(dpot,axes=[1,0,2])

    def ddpot_rearrange(self,ddpot):
        # Rearrange the hessian to be consistent with the convention used in this code
        # Helpful to use this to compute bead hessians from the PES 
        return np.transpose(ddpot,axes=[2,0,3,1])

    def compute_potential(self,fortran=False):
        """ Update the potential energy (from the PES) for the current bead positions """
        if(fortran):
            self.potential_f(self.rp.qcart_f,self.pot_f)
        else:
            self.pot[:] = self.potential(self.rp.qcart)
        return self.pot
    
    def compute_force(self,fortran=False):
        """ Update the force (from the PES) for the current bead positions """
        if(fortran):
            self.dpotential_f(self.rp.qcart_f,self.dpot_cart_f)
        else:
            self.dpot_cart[:] = self.dpotential(self.rp.qcart)
        return self.dpot_cart

    def compute_hessian(self,fortran=False):
        """ Update the hessian (from the PES) for the current bead positions """
        if(fortran):
            self.ddpotential_f(self.rp.qcart_f,self.ddpot_cart_f)
        else:   
            # The hessian computed from the PES is multiplied by a matrix to expand it to the correct dimension
            # (This is necessary because the cross-bead terms in the hessian are zero)
            if(self.ndim==1):
                self.ddpot_cart[:] = self.ddpotmat*self.ddpotential(self.rp.qcart)[:,:,:,None,np.newaxis]
            else:
                self.ddpot_cart[:] = self.ddpotmat*self.ddpotential(self.rp.qcart)[...,np.newaxis]
        return self.ddpot_cart

    def compute_mats_potential(self):
        """
        Update the Matsubara potential energy (from the PES) when the 
        Matsubara potential is given in analytical form
        """
        self.pot[:] = self.potential(self.rp.mats_beads())
        return self.pot

    def compute_mats_force(self):
        """
        Update the Matsubara force (from the PES) when the
        Matsubara potential is given in analytical form
        """
        self.dpot_cart[:] = self.dpotential(self.rp.mats_beads())
        return self.dpot_cart

    def compute_mats_hessian(self):
        """  
        Update the Matsubara hessian (from the PES) when the
        Matsubara potential is given in analytical form
        (Valid only for 1D systems in the current implementation)
        """
        self.ddpot_cart[:] = self.ddpotmat*self.ddpotential(self.rp.mats_beads())[:,:,:,None,np.newaxis]
        return self.dpot_cart

    def centrifugal_term(self):
        """
        Compute the centrifugal term for the Matsubara potential energy 
        Used to sample constant phase Matsubara trajectories
        """
        # This is currently untested
        Rsq = np.sum(self.rp.matsfreqs**2*pairwise_swap(self.rp.q[...,:self.rp.nmats],self.rp.nmats)**2,axis=2)[:,None]
        const = self.ens.theta**2/self.rp.m
        dpot = (-const/Rsq**2)*pairwise_swap(self.rp.matsfreqs[...,:self.rp.nmats],self.rp.nmats)**2*self.rp.q[...,:self.rp.nmats]
        return dpot 

    def update(self,update_hess=False):
        #Currently fortran update step is implemented only for 'rp' mode.
        #Update hess is set to be True only for monodromy and variation mode
        if(self.rp.mode=='rp'):
            self.compute_potential(self.pes_fort)
            self.compute_force(self.pes_fort)
            self.dpot[:] = self.nmtrans.cart2mats(self.dpot_cart)
            if(update_hess):
                self.compute_hessian(self.pes_fort)
                if(self.transf_fort):
                    self.nmtrans.cart2mats_hessian(self.ddpot_cart_f,self.ddpot_f,fortran=True)
                else:
                    self.ddpot[:] = self.nmtrans.cart2mats_hessian(self.ddpot_cart,fortran=False)
        elif(self.rp.mode=='rp/mats'):
            # !!! The change in the axis ordering for Mxx's and Hessian is not implemented yet !!!
            self.compute_mats_potential()
            self.compute_mats_force()
            self.compute_mats_hessian()
            self.dpot[...,:self.rp.nmats] = (self.rp.nbeads)*self.nmtrans.cart2mats(self.dpot_cart)[...,:self.rp.nmats]
            self.ddpot[...,:self.rp.nmats,:self.rp.nmats] = (self.rp.nbeads)*self.nmtrans.cart2mats_hessian(self.ddpot_cart)[...,:self.rp.nmats,:self.rp.nmats]
        elif(self.rp.mode=='mats'):
            self.dpot = self.dpotential()
            self.ddpot = self.ddpotential()
            self.dpot_cart = self.nmtrans.mats2cart(self.dpot)
            self.ddpot_cart = self.nmtrans.mats2cart_hessian(self.ddpot)    
