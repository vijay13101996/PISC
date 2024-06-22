import numpy as np
from PISC.potentials.base import PES
from numpy import exp
#from PISC.potentials.DW_Morse_harm_f import dw_morse_harm as dw_morse_harm_f

class DW_Morse_harm(PES):
        def __init__(self,alpha,D,lamda,g,z):
            super(DW_Morse_harm).__init__()
            """
            This is the two-dimensional double well potential with a 1D double well potential along the x-axis 
             a harmonic oscillator along the y-axis coupled with a nonlinear term 

             m: Mass of the particle
             w: Frequency of the harmonic oscillator

             g: Double well parameter g as in Hashimoto's paper
             lamda: Double well parameter lambda as in Hashimoto's paper

             z: Coupling strength
            """
        
            self.alpha = alpha
            self.D = D

            self.g = g
            self.lamda = lamda
            
            self.z = z

            self.param_list = [alpha,D,lamda,g,z]

        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(DW_Morse_harm,self).bind(ens,rp,pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 
        
        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            
            Vx = self.alpha**2*self.z**2*x*y**2 + 4*self.g*x*(x**2 - self.lamda**2/(8*self.g))
            Vy = 2*self.D*self.alpha*(1 - exp(-self.alpha*y))*exp(-self.alpha*y) + self.alpha**2*self.z**2*x**2*y
            
            return self.dpot_rearrange([Vx,Vy])

        def ddpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            
            Vxx =  self.alpha**2*self.z**2*y**2 + 8*self.g*x**2 + self.g*(8*x**2 - self.lamda**2/self.g)/2
            Vxy = 2*self.alpha**2*self.z**2*x*y
            Vyy = self.alpha**2*(-2*self.D*(1 - exp(-self.alpha*y))*exp(-self.alpha*y) + 2*self.D*exp(-2*self.alpha*y) + self.z**2*x**2)
            
            return self.ddpot_rearrange([[Vxx,Vxy],[Vxy,Vyy]])

        def potential_xy(self,x,y): 
            quartx = (x**2 - self.lamda**2/(8*self.g))
            eay = np.exp(-self.alpha*y)
            
            vx = self.g*quartx**2
            vy = self.D*(1-eay)**2
            vxy = self.z**2*self.alpha**2*x**2*y**2/2
             
            return vx + vy + vxy

        def potential_f(self,qcart_f,pot_f):
            return dw_harm_f.potential_f(qcart_f,self.param_list,pot_f) 
            
        def dpotential_f(self,qcart_f,dpot_f):
            return dw_harm_f.dpotential_f(qcart_f,self.param_list,dpot_f)

        def ddpotential_f(self,qcart_f,ddpot_f):
            return dw_harm_f.ddpotential_f(qcart_f,self.param_list,ddpot_f)


