import numpy as np
from PISC.potentials.base import PES

"""
The potential used in this file is the one described in the paper:
    "Quantum dynamical tunneling in bound states."
    Davis, Michael J., and Eric J. Heller.  
    The Journal of Chemical Physics 75.1 (1981): 246-254.

This demonstrates the concept of dynamical tunneling in a 2D model system
and is given as:
    V(u,s; ws, wu, lambda) = 1/2*ws^2*u^2 + 1/2*wu^2*s^2 + lambda*u^2*s

We set u:=x and s:=y for convenience.

"""

class heller_davis(PES):
        def __init__(self,ws=1.0,wu=1.1,lamda=-0.11):
            super(heller_davis).__init__()
            self.ws = ws
            self.wu = wu

            self.lamda = lamda

        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(heller_davis,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 
        
        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            Vx= 2*self.lamda*x*y  + self.wu**2*x
            Vy= self.lamda*x**2 + self.ws**2*y  
            return self.dpot_rearrange([Vx,Vy])

        def ddpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            Vxx= self.wu**2*np.ones_like(x) + 2*self.lamda*y
            Vxy= 2*self.lamda*x
            Vyy= self.ws**2*np.ones_like(y)
            return self.ddpot_rearrange([[Vxx,Vxy],[Vxy,Vyy]])
        
        def potential_xy(self,x,y):
            vx = 0.5*self.wu**2*x**2
            vy = 0.5*self.ws**2*y**2
            vxy = self.lamda*x**2*y
            return vx + vy + vxy 
