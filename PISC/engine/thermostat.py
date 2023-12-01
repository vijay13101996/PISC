"""
This module defines classes for implementing various 
thermostats into the path-integral dynamics code
"""
import numpy as np
import PISC
from PISC.utils.lapack import Gram_Schmidt
from PISC.utils.misc import pairwise_swap
from PISC.engine.integrator_f import integrator

class Thermostat(object):
    """ Base class for thermostats """
    def bind(self,rp,motion,rng,ens):
        self.rp = rp
        self.motion = motion
        self.rng = rng
        self.ens = ens
        self.dt = abs(self.motion.dt)
        
class PILE_L(Thermostat):
    """ (Local) Path-Integral Langevin Equation thermostat """
    def __init__(self,tau0=1.0, pile_lambda=100.0):
        """
        Parameter convention is as in the following paper:
        
        Rossi, M., Ceriotti, M., & Manolopoulos, D. E. (2014). 
        How to remove the spurious resonances from ring polymer molecular dynamics. 
        The Journal of chemical physics, 140(23).

        Parameters:
        tau0: Friction coefficient for the centroid mode
        pile_lambda: Scaling factor for the friction coefficients of the other modes
        """

        self.tau0 = tau0 
        self.pile_lambda = pile_lambda      

    def bind(self,rp,motion,rng,ens):
        """ Bind the components and get the coefficients of the propagator """
        super(PILE_L,self).bind(rp,motion,rng,ens)
        self.get_propa_coeffs()

    def get_propa_coeffs(self):
        """
        Get the c1 and c2 coefficients for the propagator as in the following paper:
        
        Ceriotti, M., Parrinello, M., Markland, T. E., & Manolopoulos, D. E. (2010). 
        Efficient stochastic thermostatting of path integral molecular dynamics. 
        The Journal of chemical physics, 133(12)
        """
        if(self.rp.mode == 'mats'):
            # For Mean-Field Matsubara dynamics, only the Matsubara modes are thermostatted
            self.friction = (2.0*self.pile_lambda)*np.ones(self.rp.nmodes)
            self.friction[0] = 1/self.tau0
            self.c1 = np.exp(-self.dt*self.friction/2.0)
            self.c2 = np.sqrt((1.0 - self.c1**2)/self.ens.beta)         
        else:   
            # For RPMD and CMD, all the modes are thermostatted
            self.friction = 2.0*self.pile_lambda*np.abs(self.rp.dynfreqs)
            self.friction[0] = 1.0/self.tau0
            self.c1 = np.exp(-self.dt*self.friction/2.0)
            self.c2 = np.sqrt(self.rp.nbeads*(1.0 - self.c1**2)/self.ens.beta)  
            
    def thalfstep(self,pc=True,fort=False):
        """ Half-step of the thermostat """
        p = np.reshape(self.rp.p, (-1, self.rp.ndim, self.rp.nmodes))
        sm3 = np.reshape(self.rp.sqdynm3, p.shape)
        rng_normal = self.rng.normal(size=p.shape)

        # Same random numbers for both fortran and python
        if(fort):
            integrator.thalfstep(p.T, sm3.T, rng_normal.T, self.c1, self.c2, pc, self.rp.nbeads) 
            # To be checked for (MF) Matsubara dynamics
        else:
            sp = p/sm3
            sp *= self.c1
            sp += self.c2*rng_normal
            if pc:
                p[:] = sp*sm3
            else:
                if(self.rp.mode=='rp'):
                    p[...,1:] = sp[...,1:]*sm3[...,1:]
                else:
                    p[...,self.rp.nmats:] = sp[...,self.rp.nmats:]*sm3[...,self.rp.nmats:]
       
class Andersen(Thermostat):
    """ Andersen thermostat """
    def __init__(self,N): 
        self.N=N  

    def bind(self,rp,motion,rng,ens):
        super(Andersen,self).bind(rp,motion,rng,ens)

    def generate_p(self):
        # Generate random numbers for the momenta
        return self.rng.normal(0,np.sqrt(self.rp.m*self.rp.nbeads/(self.ens.beta)),(self.N,self.ens.ndim,self.rp.nbeads))
 
class Constrain_theta(Thermostat): #Need to be upgraded for 2D simulations
    """ Theta-constrained thermostat """
    # (Work in progress)
    # Write the code for the theta-constrained thermostat here at some point. 
    def __init__(self,theta=None):
        self.theta=theta

    def bind(self,rp,motion,rng,ens):
        super(Constrain_theta,self).bind(rp,motion,rng,ens)
        if self.theta is None:
            self.theta=self.ens.theta
        self.M = self.rp.nmats   
     
    def theta_constrained_randomize(self,dim_ind=0): 
        # Scaling factor for Pi_0 
        R = (np.sum(self.rp.matsfreqs**2*pairwise_swap(self.rp.q[:,dim_ind,:self.M],self.M)**2,axis=1)**0.5)[:,None] 
        
        # First column of the unitary transformation matrix
        T0 = self.rp.matsfreqs*pairwise_swap(self.rp.q[:,dim_ind,:self.M],self.M)/R   
        T = self.rng.uniform(size=(self.rp.nsys,self.M,self.M))   
        T[...,0] = T0
        
        # Obtain the unitary transformation matrix
        T = Gram_Schmidt(T)

        N = self.rp.nsys
        Pi = self.rng.normal(0,(self.rp.m/self.ens.beta)**0.5,(N,self.M))
        
        # Setting Pi_0 to theta/R, so that the resultant distribution has the required theta
        Pi[:,0] = self.theta/R[:,0]

        # Solving for the momentum distribution
        self.rp.p[:,dim_ind,:self.M] =np.einsum('ijk,ik->ij', T,Pi)
     
