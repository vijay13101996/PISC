import numpy as np
from PISC.potentials.base import PES

class morse_fit(PES):
    def __init__(self,coeff,eq):
        super(morse_fit).__init__()
        self.coeff = coeff
        self.eq = eq
        
    def bind(self,ens,rp):
        super(morse_fit,self).bind(ens,rp)
    
    def potential(self,q):
        qd=q-self.eq
        return np.polyval(self.coeff,qd)  
    
    def dpotential(self,q):
        qd=q-self.eq
        return np.polyval(np.polyder(self.coeff),qd)

    def ddpotential(self,q):
        qd=q-self.eq
        return np.polyval(np.polyder(self.coeff,2),qd)




