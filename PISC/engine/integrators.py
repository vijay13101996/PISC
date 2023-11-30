"""
This module contains the code for implementing second 
and fourth order symplectic integrator.
"""

import numpy as np
import scipy
import scipy.integrate
from scipy.integrate import odeint, ode
from PISC.utils.misc import hess_compress, hess_mul
from PISC.utils.misc_f import misc
from PISC.engine.integrator_f import integrator

class Integrator(object):
    def __init__(self,fort=False):
        self.fort = fort

    def bind(self, ens, motion, rp, pes, rng, therm,fort=False):
        # Declare variables that contain all details from the various classes in bind 
        self.motion = motion
        self.rp = rp
        self.pes = pes  
        self.ens = ens
        self.therm = therm
        self.fort = fort
        self.therm.bind(rp,motion,rng,ens)
    
    def force_update(self,update_hess=False,fortran=False):
        """ Update the force and Hessian """
        self.rp.mats2cart()
        self.pes.update(update_hess=update_hess,fortran=fortran)

class Symplectic(Integrator):
    """
    This class implements the symplectic integrators of second and fourth order, and can be easily extended 
    to higher orders. The second order integrator is the usual Verlet algorithm, while the fourth order 
    integrator is adopted from the paper:
     
     Mark L. Brewer, Jeremy S. Hulme, and David E. Manolopoulos, 
    "Semiclassical dynamics in up to 15 coupled vibrational degrees of freedom", 
     J. Chem. Phys. 106, 4832-4839 (1997)

    The integrator has a fortran mode, which can be easily used to speed up the code.
    The test cases to compare python/fortran code are provided in PISC/test_files/test_fort_integrators.py
    """

    def O(self,pmats):
        """ Propagation of momenta due to the thermostat """
        self.therm.thalfstep(pmats,fort=self.fort)

    def A(self,k,update_hess=False):
        """ Propagation of coordinate """
        if(self.fort):
            integrator.a_f(self.rp.q_f,self.rp.p_f,self.motion.qdt[k],self.rp.dynm3_f)
        else:
            self.rp.q+=self.motion.qdt[k]*self.rp.p/self.rp.dynm3
        self.force_update(fortran=self.fort,update_hess=update_hess)

    def B(self,k,centmove=True):
        """ Propagation of momenta """
        if(self.fort):
            integrator.b_f(self.rp.p_f,self.pes.dpot_f,self.motion.pdt[k],centmove,self.rp.nbeads)
        else:   
            if centmove:
                self.rp.p-=self.pes.dpot*self.motion.pdt[k]
            else:
                self.rp.p[...,1:]-=self.pes.dpot[...,1:]*self.motion.pdt[k]

    def b(self,k):  
        """ Propagation of momenta due to spring potential """
        if(self.fort):
            integrator.b_f(self.rp.p_f,self.rp.dpot.T,self.motion.pdt[k],True,self.rp.nbeads)
        else:
            if self.rp.nmats is None:
                self.rp.p-=self.rp.dpot*self.motion.pdt[k]
            else:
                self.rp.p[...,self.rp.nmats:]-=self.pes.dpot[...,self.rp.nmats:]*self.motion.pdt[k] 
        #Note: Beware when running MF Matsubara simulation - it is untested as of yet.
    
    def Av(self,k):
        """ Propagation of coordinate in 'variation' mode """
        self.rp.dq+=self.rp.dp*self.motion.qdt[k]/self.rp.dynm3
    
    def Bv(self,k,centmove=True):
        """ Propagation of momenta in 'variation' mode """
        # Fortran mode not enabled here
        hess = self.pes.ddpot.swapaxes(2,3).reshape(-1,self.pes.ndim*self.rp.nbeads,self.rp.ndim*self.rp.nbeads)
        dq=self.rp.dq.reshape(-1,self.pes.ndim*self.rp.nbeads)
        dpc= np.einsum('ijk,ik->ij',hess,dq) #Check once again!
        dpc=dpc.reshape(-1,self.pes.ndim,self.rp.nbeads)
        if(centmove):
            self.rp.dp-=dpc*self.motion.pdt[k]
        else:
            self.rp.dp[...,1:]-=dpc[...,1:]*self.motion.pdt[k]

    def bv(self,k):
        """ Propagation of momenta in 'variation' mode due to spring potential """
        # Fortran mode not enabled here
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
        """ Propagation of the Mqp matrix elements (depends on the PES Hessian) """
        hess_mul(self.pes.ddpot,self.rp.Mqp,self.rp.Mpp,self.rp,self.motion.pdt[k],fort=self.fort)     
            
    def m1(self,k):
        """ Propagation of the Mqp matrix elements (depends on the ring-polymer Hessian) """
        hess_mul(self.rp.ddpot,self.rp.Mqp,self.rp.Mpp,self.rp,self.motion.pdt[k],fort=self.fort)      
        
    def M2(self,k):
        """ Propagation of the Mqq matrix elements (depends on the PES Hessian) """
        hess_mul(self.pes.ddpot,self.rp.Mqq,self.rp.Mpq,self.rp,self.motion.pdt[k],fort=self.fort)

    def m2(self,k):
        """ Propagation of the Mqq matrix elements (depends on the ring-polymer Hessian) """
        hess_mul(self.rp.ddpot,self.rp.Mqq,self.rp.Mpq,self.rp,self.motion.pdt[k],fort=self.fort)

    def M3(self,k):
        """ Propagation of the Mqp matrix elements (independent of the Hessian) """
        if(self.fort):
            integrator.m_upd_f(self.rp.Mqp_f,self.rp.Mpp_f,self.rp.dynm3_f,self.motion.qdt[k])
        else:
            self.rp.Mqp+=self.motion.qdt[k]*self.rp.Mpp/self.rp.dynm3[:,:,:,None,None]
        
    def M4(self,k):
        """ Propagation of the Mqq matrix elements (independent of the Hessian) """
        if(self.fort):
            integrator.m_upd_f(self.rp.Mqq_f,self.rp.Mpq_f,self.rp.dynm3_f,self.motion.qdt[k])
        else:
            self.rp.Mqq+=self.motion.qdt[k]*self.rp.Mpq/self.rp.dynm3[:,:,:,None,None]
    
    def pq_kstep(self,k,centmove=True):
        """ Propagation of the coordinates and momenta for one 'k' step (k varies from 0 to order-1) """
        self.B(k,centmove)
        self.b(k)
        self.A(k)

    def pq_kstep_nosprings(self,k):
        """ Propagation of the coordinates and momenta for one 'k' step when there are no springs """
        self.B(k)
        self.A(k)

    def var_kstep(self,k,centmove=True):
        """ Propagation of the coordinates and momenta for one 'k' step in 'variation' mode """
        self.B(k,centmove)
        self.Bv(k,centmove)
        self.b(k)
        self.bv(k)
        self.A(k,update_hess=True)
        self.Av(k)

    def Monodromy_kstep(self,k):
        """ Propagation of the coordinates, momenta and monodromy matrix elements for one 'k' step """
        self.B(k)
        self.b(k)
        self.M1(k)
        self.m1(k)
        self.M2(k)
        self.m2(k)
        self.M3(k)
        self.M4(k)
        self.A(k,update_hess=True)

    def Monodromy_kstep_nosprings(self,k):
        """ Propagation of the coordinates, momenta and monodromy matrix elements for one 'k' step 
        when there are no springs """
        self.B(k)
        self.M1(k)
        self.M2(k)
        self.M3(k)
        self.M4(k)
        self.A(k,update_hess=True)

    def pq_step(self,centmove=True):
        """ Propagation of the coordinates and momenta for one full step """
        for k in range(self.motion.order):
            self.pq_kstep(k,centmove=True)
    
    def pq_step_RSP(self,centmove=True):
        """ Propagation of the coordinates and momenta for one full step using 'Reference-System Propagation' """
        if(self.motion.order==2):
            self.B(0,centmove)
            self.rp.RSP_step()
            self.force_update()
            self.B(1,centmove)
        else:
            raise ValueError("RSP step only implemented for second order integrator")
    
    def pq_step_nosprings(self):
        """ Propagation of the coordinates and momenta for one full step when there are no springs """
        for k in range(self.motion.order):
            self.pq_kstep_nosprings(k)

    def var_step(self):
        """ Propagation of the coordinates and momenta for one full step in 'variation' mode """
        for k in range(self.motion.order):
            self.var_kstep(k,centmove=True)

    def Monodromy_step(self):
        """ Propagation of the coordinates, momenta and monodromy matrix elements for one full step """
        for k in range(self.motion.order):
            self.Monodromy_kstep(k)

    def Monodromy_step_nosprings(self): 
        """ Propagation of the coordinates, momenta and monodromy matrix elements for one full step
        when there are no springs """
        for k in range(self.motion.order):
            self.Monodromy_kstep_nosprings(k)
    
class Runge_Kutta_order_VIII(Integrator):
    """ Primitive code used to propagate the monodromy matrix elements using scipy.integrate.odeint """
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
