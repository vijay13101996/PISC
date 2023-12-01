import numpy as np
from PISC.potentials.base import PES
from numpy import exp
from PISC.potentials.quartic_bistable_f import quartic_bistable as quartic_bistable_f

class quartic_bistable(PES):
        def __init__(self,alpha,D,lamda,g,z,invert_pes=1.0):
            super(quartic_bistable).__init__()
            """
            This is the two-dimensional double well potential with a 1D double well potential along the x-axis 
             a 1D Morse potential along the y-axis coupled with a nonlinear term 

             D: Dissociation energy of the Morse potential
             alpha: Morse potential parameter (determines the width of the potential well, appears in the exponent)
             
             g: Double well parameter g as in Hashimoto's paper
             lamda: Double well parameter lambda as in Hashimoto's paper

             z: Coupling strength
            """
            
            self.D = D
            self.alpha = alpha

            self.g = g
            self.lamda = lamda
            
            self.z = z  
    
            self.param_list = [alpha,D,lamda,g,z]

        def bind(self,ens,rp,pes_fort=False,transf_fort=False):
            super(quartic_bistable,self).bind(ens,rp,pes_fort,transf_fort=transf_fort)
 
        def potential(self,q):
            x = q[:,0] 
            y = q[:,1]
            return self.potential_xy(x,y) 
        
        def dpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            Vx = 4*self.g*x*(-1 + np.exp(-self.alpha*y*self.z))*(x**2 - self.lamda**2/(8*self.g)) + 4*self.g*x*(x**2 - self.lamda**2/(8*self.g))
            Vy = 2*self.D*self.alpha*(1 - np.exp(-self.alpha*y))*np.exp(-self.alpha*y) \
                    - self.alpha*self.z*(self.g*(x**2 - self.lamda**2/(8*self.g))**2 - self.lamda**4/(64*self.g))*np.exp(-self.alpha*y*self.z)       
            
            return self.dpot_rearrange([Vx,Vy])

        def ddpotential(self,q):
            x = q[:,0] 
            y = q[:,1]
            Vxx = 4*self.g*(-2*x**2*(1 - np.exp(-self.alpha*y*self.z)) + 3*x**2 - (1 - np.exp(-self.alpha*y*self.z))*(8*x**2 - self.lamda**2/self.g)/8 - self.lamda**2/(8*self.g))
            Vxy = -4*self.alpha*self.g*x*self.z*(x**2 - self.lamda**2/(8*self.g))*np.exp(-self.alpha*y*self.z)
            Vyy = self.alpha**2*(-2*self.D*(1 - np.exp(-self.alpha*y))*np.exp(-self.alpha*y) + 2*self.D*np.exp(-2*self.alpha*y) \
                        + self.z**2*(self.g*(8*x**2 - self.lamda**2/self.g)**2 - self.lamda**4/self.g)*np.exp(-self.alpha*y*self.z)/64)

            return self.ddpot_rearrange([[Vxx,Vxy],[Vxy,Vyy]])

        def potential_xy(self,x,y):
            eay = np.exp(-self.alpha*y)
            quartx = (x**2 - self.lamda**2/(8*self.g))
            vy = self.D*(1-eay)**2
            vx = self.g*quartx**2
            vxy = (vx-self.lamda**4/(64*self.g))*(np.exp(-self.z*self.alpha*y) - 1) 
            return vx + vy + vxy

        def potential_f(self,qcart_f,pot_f):
            return quartic_bistable_f.potential_f(qcart_f,self.param_list,pot_f) 
            
        def dpotential_f(self,qcart_f,dpot_f):
            return quartic_bistable_f.dpotential_f(qcart_f,self.param_list,dpot_f)

        def ddpotential_f(self,qcart_f,ddpot_f):
            return quartic_bistable_f.ddpotential_f(qcart_f,self.param_list,ddpot_f)


