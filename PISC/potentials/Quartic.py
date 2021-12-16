import numpy as np
from PISC.potentials.base import PES

class Quartic(PES):
	def __init__(self):
		super(Quartic).__init__()	
		
	def bind(self,ens,rp):
		super(Quartic,self).bind(ens,rp)
		
	def potential(self,q):
		return 0.25*q**4 
	
	def dpotential(self,q):
		return q**3

	def ddpotential(self,q):
		return 3*q**2
