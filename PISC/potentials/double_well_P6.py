import numpy as np
from PISC.potentials.base import PES

class double_well_P6(PES):
	def __init__(self,a,b,c):
		super(double_well_P6).__init__()
		self.a = a
		self.b = b
		self.c = c

	def bind(self,ens,rp,pes_fort=False,transf_fort=False):
		super(double_well_P6,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
	
	def potential(self,q):
		#q = q[:,0,:]
		return self.a*q**2 + self.b*q**4 + self.c*q**6

	def dpotential(self,q):
		return 2*self.a*q + 4*self.b*q**3 + 6*self.c*q**5

	def ddpotential(self,q):
		return 2*self.a + 12*self.b*q**2 + 30*self.c*q**4

	
