import numpy as np
from PISC.potentials.base import PES

class Matsubara_Quartic(PES):
	def __init__(self):
		super(Matsubara_Quartic).__init__()
		
	def bind(self,ens,rp):
		super(Matsubara_Quartic,self).bind(ens,rp)
	
	def potential(self):
		Q_m1 = self.rp.q[...,1]
		Q_0  = self.rp.q[...,0]
		Q_1  = self.rp.q[...,2]
		return 0.25*(1.5*Q_m1**4 + 6.0*Q_m1**2*Q_0**2 + 3.0*Q_m1**2*Q_1**2 + 1.0*Q_0**4 \
				+ 6.0*Q_0**2*Q_1**2 + 1.5*Q_1**4)

	def dpotential(self):
		Q_m1 = self.rp.q[...,1].T
		Q_0  = self.rp.q[...,0].T
		Q_1  = self.rp.q[...,2].T
		
		return 0.25*np.array([4.0*Q_0**3 + 12.0*Q_0*Q_1**2 + 12.0*Q_0*Q_m1**2,12.0*Q_0**2*Q_m1 + 6.0*Q_1**2*Q_m1 + 6.0*Q_m1**3, \
				12.0*Q_0**2*Q_1 + 6.0*Q_1**3 + 6.0*Q_1*Q_m1**2]).T
	  
	def ddpotential(self):
		Q_m1 = self.rp.q[...,1].T
		Q_0  = self.rp.q[...,0].T
		Q_1  = self.rp.q[...,2].T
		
		ret =  0.25*np.array([ [12.0*Q_0**2 + 12.0*Q_1**2 + 12.0*Q_m1**2,24.0*Q_0*Q_m1,24.0*Q_0*Q_1],
						 [24.0*Q_0*Q_m1,  12.0*Q_0**2 + 6.0*Q_1**2 + 18.0*Q_m1**2, 12.0*Q_1*Q_m1],
							[24.0*Q_0*Q_1,12.0*Q_1*Q_m1,12.0*Q_0**2 + 18.0*Q_1**2 + 6.0*Q_m1**2]])

		return ret.T


