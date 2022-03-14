import numpy as np
from PISC.potentials.base import PES

class henon_heiles(PES):
		def __init__(self,lamda,g):
			super(henon_heiles).__init__()
			self.g = g
			self.lamda = lamda	
				
		def bind(self,ens,rp):
			super(henon_heiles,self).bind(ens,rp)
 
		def potential(self,q):
			x = q[:,0] 
			y = q[:,1]
			return self.potential_xy(x,y) 
		
		def dpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			Vx = 4*self.g*x*(x**2 + y**2) + 2*self.lamda*x*y + x
			Vy = 4*self.g*y*(x**2 + y**2) + self.lamda*(x**2 - y**2) + y

			return np.transpose([Vx,Vy],axes=[1,0,2])
	
		def ddpotential(self,q):
			x = q[:,0] 
			y = q[:,1]
			Vxx = 8*self.g*x**2 + 4*self.g*(x**2 + y**2) + 2*self.lamda*y + 1.0
			Vxy = 2*x*(4*self.g*y + self.lamda)
			Vyx = 2*x*(4*self.g*y + self.lamda)
			Vyy = 8*self.g*y**2 + 4*self.g*(x**2 + y**2) - 2.0*self.lamda*y + 1.0

			return np.transpose([[Vxx, Vxy],[Vxy, Vyy]],axes=[2,0,1,3]) 
			
		def potential_xy(self,x,y):
			vh = 0.5*(x**2+y**2)
			vp = self.lamda*(x**2*y - y**3/3.0)
			vs = self.g*(x**2+y**2)**2	
			return (vh + vp + vs) 
