import numpy as np
from PISC.potentials.base import PES
from numpy import exp
#from PISC.potentials.DW_harm_f import dw_harm as dw_harm_f

class DW_harm(PES):
        def __init__(self,m,w,lamda,g,z,T=1.0):
            super(DW_harm).__init__()
            """
            This is the two-dimensional double well potential with a 1D double well potential along the x-axis 
             a harmonic oscillator along the y-axis coupled with a nonlinear term 

             m: Mass of the particle
             w: Frequency of the harmonic oscillator

             g: Double well parameter g as in Hashimoto's paper
             lamda: Double well parameter lambda as in Hashimoto's paper

             z: Coupling strength
            """
            self.m = m
            self.w = w
            
            self.g = g
            self.lamda = lamda
            
            self.z = z

            self.T = T

            self.param_list = [m,w,lamda,g,z,T]

        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(DW_harm,self).bind(ens,rp,pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 
        
        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
           
            Vx = 4*self.g*x*(x**2 - self.lamda**2/(8*self.g)) + 2*self.z*x*y**2
            Vy = self.T**2*self.m*self.w**2*y + 2*self.z*x**2*y
            
            return self.dpot_rearrange([Vx,Vy])

        def ddpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            
            Vxx =  2*(4*self.g*x**2 + self.g*(8*x**2 - self.lamda**2/self.g)/4 + self.z*y**2)
            Vxy = 4*self.z*x*y
            Vyy = self.T**2*self.m*self.w**2 + 2*self.z*x**2
            
            return self.ddpot_rearrange([[Vxx,Vxy],[Vxy,Vyy]])

        def potential_xy(self,x,y): 
            quartx = (x**2 - self.lamda**2/(8*self.g))
            vy = 0.5*self.m*self.w**2*y**2*self.T**2
            vx = self.g*quartx**2
            vxy = self.z*y**2*x**2
            
            self.alpha = 0.382
            self.D = 9.3
            
            return vx + vy + vxy

        def potential_f(self,qcart_f,pot_f):
            return dw_harm_f.potential_f(qcart_f,self.param_list,pot_f) 
            
        def dpotential_f(self,qcart_f,dpot_f):
            return dw_harm_f.dpotential_f(qcart_f,self.param_list,dpot_f)

        def ddpotential_f(self,qcart_f,ddpot_f):
            return dw_harm_f.ddpotential_f(qcart_f,self.param_list,ddpot_f)


