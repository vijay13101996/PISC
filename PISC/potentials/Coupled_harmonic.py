import numpy as np
from PISC.potentials.base import PES

## This potential is known as 'Pullen-Edmonds Hamiltonian' in 
##  James L. Anchell, J. Chem. Phys. 92, 4342-4350 (1990) 

class coupled_harmonic(PES):
        def __init__(self,omega,g0):
            super(coupled_harmonic).__init__()
            self.omega = omega
            self.g0 = g0
                
        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(coupled_harmonic,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 

        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]      
            Vx = self.omega**2*x/2 + 2*self.g0*x*y**2
            Vy = self.omega**2*y/2 + 2*self.g0*y*x**2
            return self.dpot_rearrange([Vx,Vy])
            #return np.transpose([(self.omega**2/2.0)*x + 2*self.g0*x*y**2,(self.omega**2/2.0)*y + 2*self.g0*y*x**2],axes=[1,0,2])
            #return np.transpose([(self.omega**2/2.0)*x + self.g0*y,(self.omega**2/2.0)*y + self.g0*x],axes=[1,0,2])

        def ddpotential(self,q):
            x = q[:,0]
            y = q[:,1]
            Vxx = self.omega**2/2 + 2*self.g0*y**2
            Vxy = 4*self.g0*x*y
            Vyy = self.omega**2/2 + 2*self.g0*x**2
            return self.ddpot_rearrange([[Vxx,Vxy],[Vxy,Vyy]])
            #return np.transpose([[(self.omega**2/2.0) + 2*self.g0*y**2, 4*self.g0*x*y],[4*self.g0*x*y, (self.omega**2/2.0) + 2*self.g0*x**2]],axes=[2,0,1,3]) 
            #return np.transpose([[(self.omega**2/2.0)*np.ones_like(x), self.g0*np.ones_like(x)],[self.g0*np.ones_like(x), \
            #       (self.omega**2/2.0)*np.ones_like(x)]],axes=[2,0,1,3]) 

        def potential_xy(self,x,y):
            return np.array((self.omega**2/4.0)*(x**2+y**2) + self.g0*x**2*y**2)
            #return np.array((self.omega**2/4.0)*(x**2+y**2) + self.g0*x*y)
            #return np.array((self.omega**2/4.0)*(x**2+y**2) + self.g0*(x**2*y)) 
