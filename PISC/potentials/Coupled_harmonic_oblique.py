import numpy as np
from PISC.potentials.base import PES

## This potential is known as 'Pullen-Edmonds Hamiltonian' in 
##  James L. Anchell, J. Chem. Phys. 92, 4342-4350 (1990) 

class coupled_harmonic_oblique(PES):
        def __init__(self,m,w1,w2,g0):
            super(coupled_harmonic_oblique).__init__()
            self.m = m
            self.w1 = w1
            self.w2 = w2
            self.g0 = g0
                
        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(coupled_harmonic_oblique,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 

        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]      
            return self.dpot_rearrange([Vx,Vy])

        def ddpotential(self,q):
            x = q[:,0]
            y = q[:,1]
            Vxx = self.omega**2/2 + 2*self.g0*y**2
            Vxy = 4*self.g0*x*y
            Vyy = self.omega**2/2 + 2*self.g0*x**2
            return self.ddpot_rearrange([[Vxx,Vxy],[Vxy,Vyy]])

        def potential_xy(self,x,y):
            return np.array(self.m/2*(self.w1**2*x**2 + self.w2**2*y**2) + self.g0*x**2*y**2)
