import numpy as np
from PISC.potentials.base import PES
#from PISC.potentials.henon_heiles_f import henon_heiles as henon_heiles_f

class henon_heiles(PES):
        def __init__(self,lamda,g):
            super(henon_heiles).__init__()
            self.g = g
            self.lamda = lamda  
            self.param_list = [lamda,g]

        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(henon_heiles,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 
        
        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            Vx = 4*self.g*x*(x**2 + y**2) + 2*self.lamda*x*y + x
            Vy = 4*self.g*y*(x**2 + y**2) + self.lamda*(x**2 - y**2) + y
            return self.dpot_rearrange([Vx,Vy]) 

        def ddpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            Vxx = 8*self.g*x**2 + 4*self.g*(x**2 + y**2) + 2*self.lamda*y + 1.0
            Vxy = 2*x*(4*self.g*y + self.lamda)
            Vyx = 2*x*(4*self.g*y + self.lamda)
            Vyy = 8*self.g*y**2 + 4*self.g*(x**2 + y**2) - 2.0*self.lamda*y + 1.0

            return self.ddpot_rearrange([[Vxx,Vxy],[Vxy,Vyy]])

        def potential_xy(self,x,y):
            vh = 0.5*(x**2+y**2)
            vp = self.lamda*(x**2*y - y**3/3.0)
            vs = self.g*(x**2+y**2)**2  
            return (vh + vp + vs)

        def potential_f(self,qcart_f,pot_f):
            return henon_heiles_f.potential_f(qcart_f,self.param_list,pot_f) 
            
        def dpotential_f(self,qcart_f,dpot_f):
            return henon_heiles_f.dpotential_f(qcart_f,self.param_list,dpot_f)

        def ddpotential_f(self,qcart_f,ddpot_f):
            return henon_heiles_f.ddpotential_f(qcart_f,self.param_list,ddpot_f)
