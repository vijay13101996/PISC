import numpy as np
from PISC.potentials.base import PES
#from PISC.potentials.coupled_quartic_f import coupled_quartic as coupled_quartic_f

class coupled_quartic(PES):
        def __init__(self,g1,g2):
            super(coupled_quartic).__init__()
            self.g1 = g1
            self.g2 = g2
            self.param_list = [g1,g2]

        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(coupled_quartic,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 

        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]      
            Vx = 4*self.g2*x**3 + 2*self.g1*x*y**2
            Vy = 4*self.g2*y**3 + 2*self.g1*y*x**2
            return self.dpot_rearrange([Vx,Vy]) # TO BE CHECKED!!!

        def ddpotential(self,q):
            x = q[:,0]
            y = q[:,1]
            Vxx = 12*self.g2*x**2 + 2*self.g1*y**2
            Vxy = 4*self.g1*x*y
            Vyy = 12*self.g2*y**2 + 2*self.g1*x**2
            return self.ddpot_rearrange([[Vxx,Vxy],[Vxy,Vyy]]) # TO BE CHECKED!!!

        def potential_xy(self,x,y):
            #return np.array( self.g2*(x**4+y**4) + self.g1*x**4*y**4 )
            return np.array( self.g1*(x**2+y**2) + self.g2*x**4*y**4 )

        def potential_f(self,qcart_f,pot_f):
            return coupled_quartic_f.potential_f(qcart_f,self.param_list,pot_f)

        def dpotential_f(self,qcart_f,dpot_f):
            return coupled_quartic_f.dpotential_f(qcart_f,self.param_list,dpot_f)

        def ddpotential_f(self,qcart_f,ddpot_f):
            return coupled_quartic_f.ddpotential_f(qcart_f,self.param_list,ddpot_f)

