import numpy as np
from PISC.potentials.base import PES

class morse(PES):
    def __init__(self,D,alpha,eq=0.0):
        super(morse).__init__()
        self.alpha = alpha
        self.D = D
        self.eq = eq
        
    def bind(self,ens,rp):
        super(morse,self).bind(ens,rp)
    
    def potential(self,q):
        qd=q-self.eq
        eaq = np.exp(-self.alpha*(qd))
        return self.D*(1-eaq)**2    
    
    def dpotential(self,q):
        qd=q-self.eq
        return 2*self.D*self.alpha*(1 - np.exp(-self.alpha*qd))*np.exp(-self.alpha*qd)

    def ddpotential(self,q):
        qd=q-self.eq
        return self.alpha**2*(-2*self.D*(1 - np.exp(-self.alpha*qd))*np.exp(-self.alpha*qd) + 2*self.D*np.exp(-2*self.alpha*qd))

