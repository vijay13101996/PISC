import numpy as np
from PISC.potentials.base import PES
from PISC.potentials.coupled_quartic_f import coupled_quartic as coupled_quartic_f

class Morse_harm_2D(PES):
        def __init__(self,m,omega,D,alpha,z):
            super(Morse_harm_2D).__init__()
            self.m = m
            self.omega = omega
            self.D = D
            self.alpha = alpha
            self.z = z
            self.param_list = [m,omega,D,alpha,z]

        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(Morse_harm_2D,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 

        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]      
            return self.dpot_rearrange([Vx,Vy]) # TO BE CHECKED!!!

        def ddpotential(self,q):
            x = q[:,0]
            y = q[:,1]
            return self.ddpot_rearrange([[Vxx,Vxy],[Vxy,Vyy]]) # TO BE CHECKED!!!

        def potential_xy(self,x,y):
            return np.array(0.5*self.m*self.omega**2*x**2 + self.D*(1-np.exp(-self.alpha*y))**2 + self.z*x**2*y**2)

        def potential_f(self,qcart_f,pot_f):
            return coupled_quartic_f.potential_f(qcart_f,self.param_list,pot_f)

        def dpotential_f(self,qcart_f,dpot_f):
            return coupled_quartic_f.dpotential_f(qcart_f,self.param_list,dpot_f)

        def ddpotential_f(self,qcart_f,ddpot_f):
            return coupled_quartic_f.ddpotential_f(qcart_f,self.param_list,ddpot_f)

