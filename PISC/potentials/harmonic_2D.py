import numpy as np
from PISC.potentials.base import PES
from PISC.potentials.harmonic_2D_f import harmonic_2d as harmonic_2D_f

class Harmonic(PES):
        def __init__(self,omega,m=0.5):
            super(Harmonic).__init__()
            """
             Define the mass m and frequency w of the harmonic oscillator
            (Mass is set to 0.5 by default, to compare with Hashimoto's OTOC papers)
            
            """

            self.omega = omega
            self.m=m
            self.param_list = np.array([self.m,self.omega])
            
            # The parameter list is used to pass parameters to the fortran functions 

        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(Harmonic,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y)

        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            dVx = self.m*self.omega**2*x
            dVy = self.m*self.omega**2*y
            return self.dpot_rearrange([dVx,dVy])

        def ddpotential(self,q):
            x = q[:,0]
            y = q[:,1]
            ddVxx = self.m*self.omega**2*np.ones_like(x)
            ddVxy = np.zeros_like(x)
            ddVyy = self.m*self.omega**2*np.ones_like(x)
            return self.ddpot_rearrange([[ddVxx,ddVxy],[ddVxy,ddVyy]])

        def potential_xy(self,x,y):
            # Calculate the potential energy when x and y are specified
            # Most useful for generating contour plots
            return np.array((self.omega**2/4.0)*(x**2+y**2))

        # Functions to compute the potential energy, force and hessian using fortran
        def potential_f(self,qcart_f,pot_f):
            harmonic_2D_f.potential_f(qcart_f,self.param_list,pot_f)            

        def dpotential_f(self,qcart_f,dpot_cart_f):
            harmonic_2D_f.dpotential_f(qcart_f,self.param_list,dpot_cart_f)

        def ddpotential_f(self,qcart_f,ddpot_cart_f):
            harmonic_2D_f.ddpotential_f(qcart_f,self.param_list,ddpot_cart_f)
