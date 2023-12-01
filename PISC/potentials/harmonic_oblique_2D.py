import numpy as np
from PISC.potentials.base import PES
from PISC.potentials.harmonic_oblique_f import harmonic_oblique as harmonic_oblique_f

class Harmonic_oblique(PES):
        def __init__(self,T,m,omega1,omega2):
            super(Harmonic_oblique).__init__()
            self.T = T
            self.m = m
            self.omega1 = omega1
            self.omega2 = omega2
            self.T11 = T[0,0]
            self.T12 = T[0,1]
            self.T21 = T[1,0]
            self.T22 = T[1,1]

            self.param_list = [self.m, self.omega1, self.omega2, self.T11, self.T12, self.T21, self.T22]

        def bind(self,ens,rp,fort=False):
            super(Harmonic_oblique,self).bind(ens,rp,fort=fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 

        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            xi  = self.T11*x + self.T12*y
            eta = self.T21*x + self.T22*y
            dVx = self.T11*self.m*self.omega1**2*xi + self.T21*self.m*self.omega2**2*eta      
            dVy = self.T12*self.m*self.omega1**2*xi + self.T22*self.m*self.omega2**2*eta
            return self.dpot_rearrange([dVx,dVy])

        def ddpotential(self,q):
            x = q[:,0]
            y = q[:,1]
            ones = np.ones_like(x)
            ddVxx = self.T11**2*self.m*self.omega1**2 + self.T21**2*self.m*self.omega2**2
            ddVxy = self.T11*self.T12*self.m*self.omega1**2 + self.T21*self.T22*self.m*self.omega2**2
            ddVyy = self.T12**2*self.m*self.omega1**2 + self.T22**2*self.m*self.omega2**2
            return self.ddpot_rearrange([[ddVxx*ones,ddVxy*ones],[ddVxy*ones,ddVyy*ones]])

        def potential_xy(self,x,y):
            xi  = self.T11*x + self.T12*y
            eta = self.T21*x + self.T22*y
            return np.array(0.5*self.m*self.omega1**2*xi**2 + 0.5*self.m*self.omega2**2*eta**2)

        # Functions to compute the potential energy, force and hessian using fortran
        def potential_f(self,qcart_f,pot_f):
            harmonic_oblique_f.potential_f(qcart_f,self.param_list,pot_f)            

        def dpotential_f(self,qcart_f,dpot_cart_f):
            harmonic_oblique_f.dpotential_f(qcart_f,self.param_list,dpot_cart_f)

        def ddpotential_f(self,qcart_f,ddpot_cart_f):
            harmonic_oblique_f.ddpotential_f(qcart_f,self.param_list,ddpot_cart_f)

